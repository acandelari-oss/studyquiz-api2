from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Dict, Iterable, Optional

from sqlalchemy import text


DEFAULT_MIN_SEMANTIC_SIMILARITY = 0.55
DEFAULT_MIN_SHARED_CHUNKS = 1
DEFAULT_TOP_K_PER_TOPIC = 5


@dataclass(frozen=True)
class TopicRelationshipFilters:
    min_semantic_similarity: float = DEFAULT_MIN_SEMANTIC_SIMILARITY
    min_shared_chunks: int = DEFAULT_MIN_SHARED_CHUNKS
    top_k_per_topic: int = DEFAULT_TOP_K_PER_TOPIC
    focus_topic_id: Optional[str] = None
    focus_topic: Optional[str] = None


def _row_value(row, key, index):
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "_mapping"):
        return row._mapping[key]
    return row[index]


def semantic_similarity_from_negative_inner_product(value):
    if value is None:
        return None
    return -float(value)


def _clean_text(value):
    return str(value or "").strip()


def _topic_pair_key(topic_a_id, topic_b_id):
    left = str(topic_a_id)
    right = str(topic_b_id)
    return (left, right) if left < right else (right, left)


def _edge_sort_key(edge):
    return (
        -int(edge["shared_chunks"] or 0),
        -float(edge["chunk_jaccard"] or 0),
        -float(edge["semantic_similarity"] or -1),
        edge["topic_a_id"],
        edge["topic_b_id"],
    )


def _matches_focus(edge, focus_topic_id):
    if not focus_topic_id:
        return True
    return (
        edge["topic_a_id"] == focus_topic_id
        or edge["topic_b_id"] == focus_topic_id
    )


def _resolve_focus_topic_id(nodes, focus_topic_id=None, focus_topic=None):
    if focus_topic_id:
        candidate = str(focus_topic_id)
        return candidate if candidate in nodes else None

    normalized_focus = _clean_text(focus_topic).lower()
    if not normalized_focus:
        return None

    for topic_id, node in nodes.items():
        if _clean_text(node["topic"]).lower() == normalized_focus:
            return topic_id

    return None


def build_topic_relationship_graph(
    topic_rows: Iterable,
    topic_chunk_rows: Iterable,
    semantic_pair_rows: Iterable,
    filters: Optional[TopicRelationshipFilters] = None,
):
    filters = filters or TopicRelationshipFilters()
    started = perf_counter()

    nodes: Dict[str, dict] = {}
    ordered_topic_ids = []

    for row in topic_rows:
        topic_id = str(_row_value(row, "id", 0))
        if topic_id in nodes:
            continue

        ordered_topic_ids.append(topic_id)
        nodes[topic_id] = {
            "id": topic_id,
            "topic": _row_value(row, "topic", 1),
            "category": _row_value(row, "category", 2),
            "source_section": _row_value(row, "source_section", 3),
            "associated_chunk_count": 0,
            "document_count": 0,
            "section_count": 0,
            "total_associated_text_length": 0,
        }

    links_by_chunk = defaultdict(dict)
    topic_chunk_ids = defaultdict(set)
    topic_documents = defaultdict(set)
    topic_sections = defaultdict(set)
    topic_text_lengths = defaultdict(int)

    for row in topic_chunk_rows:
        topic_id = str(_row_value(row, "topic_id", 0))
        chunk_id = _row_value(row, "chunk_id", 1)
        if topic_id not in nodes or chunk_id is None:
            continue

        chunk_key = str(chunk_id)
        if chunk_key in topic_chunk_ids[topic_id]:
            continue

        document_id = _row_value(row, "document_id", 2)
        doc_title = _row_value(row, "doc_title", 3)
        section = _clean_text(_row_value(row, "section", 4))
        text_length = _row_value(row, "text_length", 5) or 0

        document_key = _clean_text(document_id) or _clean_text(doc_title)

        topic_chunk_ids[topic_id].add(chunk_key)
        if document_key:
            topic_documents[topic_id].add(document_key)
        if section:
            topic_sections[topic_id].add(section)
        try:
            topic_text_lengths[topic_id] += int(text_length)
        except (TypeError, ValueError):
            pass

        links_by_chunk[chunk_key][topic_id] = {
            "document": document_key,
            "section": section,
        }

    for topic_id, node in nodes.items():
        node["associated_chunk_count"] = len(topic_chunk_ids[topic_id])
        node["document_count"] = len(topic_documents[topic_id])
        node["section_count"] = len(topic_sections[topic_id])
        node["total_associated_text_length"] = topic_text_lengths[topic_id]

    pair_signals = defaultdict(
        lambda: {
            "shared_chunks": 0,
            "shared_sections": set(),
            "shared_documents": set(),
            "semantic_similarity": None,
        }
    )

    for chunk_topics in links_by_chunk.values():
        topic_ids = sorted(chunk_topics)
        for topic_a_id, topic_b_id in combinations(topic_ids, 2):
            key = _topic_pair_key(topic_a_id, topic_b_id)
            pair_signals[key]["shared_chunks"] += 1

            section = (
                chunk_topics[topic_a_id].get("section")
                or chunk_topics[topic_b_id].get("section")
            )
            if section:
                pair_signals[key]["shared_sections"].add(section)

            document = (
                chunk_topics[topic_a_id].get("document")
                or chunk_topics[topic_b_id].get("document")
            )
            if document:
                pair_signals[key]["shared_documents"].add(document)

    candidate_pairs_evaluated = 0
    for row in semantic_pair_rows:
        topic_a_id = str(_row_value(row, "topic_a_id", 0))
        topic_b_id = str(_row_value(row, "topic_b_id", 1))
        if topic_a_id not in nodes or topic_b_id not in nodes:
            continue

        candidate_pairs_evaluated += 1
        key = _topic_pair_key(topic_a_id, topic_b_id)
        semantic_similarity = _row_value(row, "semantic_similarity", 2)
        if semantic_similarity is None:
            semantic_similarity = semantic_similarity_from_negative_inner_product(
                _row_value(row, "negative_inner_product", 3)
            )
        pair_signals[key]["semantic_similarity"] = semantic_similarity

    candidate_edges = []
    for (topic_a_id, topic_b_id), signals in pair_signals.items():
        chunks_a = len(topic_chunk_ids[topic_a_id])
        chunks_b = len(topic_chunk_ids[topic_b_id])
        shared_chunks = int(signals["shared_chunks"] or 0)
        denominator = chunks_a + chunks_b - shared_chunks
        chunk_jaccard = (
            round(shared_chunks / denominator, 4)
            if denominator > 0
            else 0
        )
        semantic_similarity = signals["semantic_similarity"]

        passes_semantic_filter = (
            semantic_similarity is not None
            and semantic_similarity >= filters.min_semantic_similarity
        )
        passes_shared_filter = shared_chunks >= filters.min_shared_chunks

        if not (passes_semantic_filter or passes_shared_filter):
            continue

        candidate_edges.append({
            "topic_a_id": topic_a_id,
            "topic_b_id": topic_b_id,
            "topic_a": nodes[topic_a_id]["topic"],
            "topic_b": nodes[topic_b_id]["topic"],
            "category_a": nodes[topic_a_id]["category"],
            "category_b": nodes[topic_b_id]["category"],
            "semantic_similarity": (
                round(float(semantic_similarity), 4)
                if semantic_similarity is not None
                else None
            ),
            "shared_chunks": shared_chunks,
            "chunks_a": chunks_a,
            "chunks_b": chunks_b,
            "chunk_jaccard": chunk_jaccard,
            "shared_sections": len(signals["shared_sections"]),
            "shared_documents": len(signals["shared_documents"]),
        })

    focus_topic_id = _resolve_focus_topic_id(
        nodes,
        focus_topic_id=filters.focus_topic_id,
        focus_topic=filters.focus_topic,
    )

    if focus_topic_id:
        selected_edges = [
            edge
            for edge in candidate_edges
            if _matches_focus(edge, focus_topic_id)
        ]
        selected_edges.sort(key=_edge_sort_key)
        selected_edges = selected_edges[:filters.top_k_per_topic]
    else:
        selected_keys = set()
        edges_by_topic = defaultdict(list)
        for edge in candidate_edges:
            edges_by_topic[edge["topic_a_id"]].append(edge)
            edges_by_topic[edge["topic_b_id"]].append(edge)

        for topic_id in ordered_topic_ids:
            topic_edges = edges_by_topic.get(topic_id, [])
            topic_edges.sort(key=_edge_sort_key)
            for edge in topic_edges[:filters.top_k_per_topic]:
                selected_keys.add(_topic_pair_key(
                    edge["topic_a_id"],
                    edge["topic_b_id"],
                ))

        selected_edges = [
            edge
            for edge in candidate_edges
            if _topic_pair_key(edge["topic_a_id"], edge["topic_b_id"])
            in selected_keys
        ]
        selected_edges.sort(key=_edge_sort_key)

    connected_topic_ids = set()
    for edge in selected_edges:
        connected_topic_ids.add(edge["topic_a_id"])
        connected_topic_ids.add(edge["topic_b_id"])

    return {
        "node_count": len(nodes),
        "edge_count": len(selected_edges),
        "candidate_pairs_evaluated": candidate_pairs_evaluated,
        "candidate_edge_count_before_top_k": len(candidate_edges),
        "isolated_node_count": len(nodes) - len(connected_topic_ids),
        "filters": {
            "min_semantic_similarity": filters.min_semantic_similarity,
            "min_shared_chunks": filters.min_shared_chunks,
            "top_k_per_topic": filters.top_k_per_topic,
            "focus_topic_id": focus_topic_id,
            "focus_topic": filters.focus_topic,
        },
        "nodes": [
            nodes[topic_id]
            for topic_id in ordered_topic_ids
        ],
        "edges": selected_edges,
        "execution_time_ms": round((perf_counter() - started) * 1000, 2),
    }


def get_topic_relationship_graph(
    db,
    project_id: str,
    filters: Optional[TopicRelationshipFilters] = None,
):
    topic_rows = db.execute(
        text("""
            select
                id,
                topic,
                category,
                source_section
            from topics
            where project_id = :project_id
            and is_display_topic = true
            and topic is not null
            order by category asc, topic asc, id asc
        """),
        {"project_id": project_id},
    ).fetchall()

    topic_chunk_rows = db.execute(
        text("""
            select distinct
                tc.topic_id,
                tc.chunk_id,
                c.document_id,
                c.doc_title,
                c.section,
                char_length(c.chunk_text) as text_length
            from topic_chunks tc
            join topics t on t.id = tc.topic_id
            join chunks c on c.id = tc.chunk_id
            where t.project_id = :project_id
            and t.is_display_topic = true
            and t.topic is not null
        """),
        {"project_id": project_id},
    ).fetchall()

    semantic_pair_rows = db.execute(
        text("""
            select
                a.id as topic_a_id,
                b.id as topic_b_id,
                -(a.embedding <#> b.embedding) as semantic_similarity,
                null as negative_inner_product
            from topics a
            join topics b on b.project_id = a.project_id
                and a.id < b.id
            where a.project_id = :project_id
            and a.is_display_topic = true
            and b.is_display_topic = true
            and a.topic is not null
            and b.topic is not null
            and a.embedding is not null
            and b.embedding is not null
        """),
        {"project_id": project_id},
    ).fetchall()

    graph = build_topic_relationship_graph(
        topic_rows,
        topic_chunk_rows,
        semantic_pair_rows,
        filters=filters,
    )
    graph["project_id"] = project_id
    return graph
