import json
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import text


MAX_EVIDENCE_CHUNKS = 6
MAX_CHUNK_CHARS = 900


def generate_relationship_explanation(
    db,
    openai_client,
    project_id: str,
    topic_a_id: str,
    topic_b_id: str,
    study_language: str = "English",
) -> Dict[str, Any]:
    topics = _load_topics(db, project_id, topic_a_id, topic_b_id)
    topic_a = topics.get(topic_a_id)
    topic_b = topics.get(topic_b_id)

    if not topic_a or not topic_b:
        raise HTTPException(status_code=404, detail="Relationship topics not found")

    evidence_chunks = _load_relationship_evidence(
        db,
        project_id,
        topic_a_id,
        topic_b_id,
    )

    if not evidence_chunks:
        raise HTTPException(
            status_code=404,
            detail="No source evidence found for this relationship",
        )

    prompt = _build_relationship_explanation_prompt(
        topic_a,
        topic_b,
        evidence_chunks,
        _normalize_study_language(study_language),
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DOUNO, an experienced university tutor. "
                    "Explain relationships using only the supplied study-material evidence. "
                    "Do not invent facts or sources."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(
            raw_content.replace("```json", "").replace("```", "").strip()
        )
    except Exception:
        parsed = {}

    why_connected = _clean_text(parsed.get("why_connected"))
    study_relevance = _clean_text(parsed.get("study_relevance"))
    evidence_summary = _clean_text(parsed.get("evidence_summary"))

    if not why_connected:
        why_connected = (
            f"{topic_a['topic']} and {topic_b['topic']} are connected in the "
            "uploaded material, but the explanation could not be generated safely."
        )
    if not study_relevance:
        study_relevance = (
            "Study these concepts together by identifying what each one explains "
            "and then checking where the source material places them in relation."
        )
    if not evidence_summary:
        evidence_summary = _fallback_evidence_summary(evidence_chunks)

    return {
        "topic_a": topic_a,
        "topic_b": topic_b,
        "why_connected": why_connected,
        "study_relevance": study_relevance,
        "evidence_summary": evidence_summary,
        "evidence": [_serialize_evidence_chunk(chunk) for chunk in evidence_chunks],
    }


def _load_topics(db, project_id: str, topic_a_id: str, topic_b_id: str) -> Dict[str, Dict[str, Any]]:
    rows = db.execute(
        text(
            """
            select id, topic, category, description
            from topics
            where project_id = :project_id
            and id in (:topic_a_id, :topic_b_id)
            """
        ),
        {
            "project_id": project_id,
            "topic_a_id": topic_a_id,
            "topic_b_id": topic_b_id,
        },
    ).mappings().fetchall()

    return {
        str(row["id"]): {
            "id": str(row["id"]),
            "topic": row["topic"] or "",
            "category": row["category"],
            "description": row["description"] or "",
        }
        for row in rows
    }


def _load_relationship_evidence(
    db,
    project_id: str,
    topic_a_id: str,
    topic_b_id: str,
) -> List[Dict[str, Any]]:
    shared_rows = db.execute(
        text(
            """
            with topic_a_chunks as (
                select chunk_id from topic_chunks where topic_id = :topic_a_id
            ),
            topic_b_chunks as (
                select chunk_id from topic_chunks where topic_id = :topic_b_id
            )
            select
                c.id,
                c.chunk_text,
                c.doc_title,
                c.page,
                c.section,
                c.topic,
                'shared' as evidence_type
            from chunks c
            join topic_a_chunks a on a.chunk_id = c.id
            join topic_b_chunks b on b.chunk_id = c.id
            where c.project_id = :project_id
            and c.chunk_text is not null
            order by char_length(c.chunk_text) desc, c.id asc
            limit :limit
            """
        ),
        {
            "project_id": project_id,
            "topic_a_id": topic_a_id,
            "topic_b_id": topic_b_id,
            "limit": MAX_EVIDENCE_CHUNKS,
        },
    ).mappings().fetchall()

    if len(shared_rows) >= 2:
        return [dict(row) for row in shared_rows]

    supporting_rows = db.execute(
        text(
            """
            select distinct
                c.id,
                c.chunk_text,
                c.doc_title,
                c.page,
                c.section,
                c.topic,
                case
                    when tc.topic_id = :topic_a_id then 'topic_a'
                    when tc.topic_id = :topic_b_id then 'topic_b'
                    else 'supporting'
                end as evidence_type
            from topic_chunks tc
            join chunks c on c.id = tc.chunk_id
            where c.project_id = :project_id
            and tc.topic_id in (:topic_a_id, :topic_b_id)
            and c.chunk_text is not null
            order by c.id asc
            limit :limit
            """
        ),
        {
            "project_id": project_id,
            "topic_a_id": topic_a_id,
            "topic_b_id": topic_b_id,
            "limit": MAX_EVIDENCE_CHUNKS,
        },
    ).mappings().fetchall()

    combined: Dict[str, Dict[str, Any]] = {
        str(row["id"]): dict(row)
        for row in shared_rows
    }
    for row in supporting_rows:
        combined.setdefault(str(row["id"]), dict(row))

    return list(combined.values())[:MAX_EVIDENCE_CHUNKS]


def _build_relationship_explanation_prompt(
    topic_a: Dict[str, Any],
    topic_b: Dict[str, Any],
    evidence_chunks: List[Dict[str, Any]],
    study_language: str,
) -> str:
    evidence_text = "\n\n".join(
        _format_evidence_for_prompt(index + 1, chunk)
        for index, chunk in enumerate(evidence_chunks)
    )

    return f"""
    Explain the relationship between two study topics using only the evidence below.

    OUTPUT LANGUAGE:
    {study_language}

    The output language above is mandatory. Do not infer the answer language
    from topic titles, category names, document language, source chunks, or
    examples. If the uploaded study material is in a different language, still
    write the explanation entirely in {study_language}.

    TOPIC A:
    Title: {topic_a.get("topic")}
    Category: {topic_a.get("category") or "Uncategorized"}
    Description: {topic_a.get("description") or "Not available"}

    TOPIC B:
    Title: {topic_b.get("topic")}
    Category: {topic_b.get("category") or "Uncategorized"}
    Description: {topic_b.get("description") or "Not available"}

    SOURCE EVIDENCE:
    {evidence_text}

    Requirements:
    - Be concise and student-oriented.
    - Write every returned field entirely in {study_language}.
    - Explain the conceptual connection, not the relationship metrics.
    - Do not say the topics are connected merely because they are semantically similar.
    - Do not add facts that are not supported by the source evidence.
    - If the evidence is limited, acknowledge that the connection is based on limited evidence.

    Return valid JSON with exactly these fields:
    {{
      "why_connected": "...",
      "study_relevance": "...",
      "evidence_summary": "..."
    }}
    """


def _format_evidence_for_prompt(index: int, chunk: Dict[str, Any]) -> str:
    source = _format_source(chunk.get("doc_title"), chunk.get("page"))
    section = chunk.get("section") or "Section not available"
    text_value = _clean_text(chunk.get("chunk_text"))[:MAX_CHUNK_CHARS]
    return (
        f"EVIDENCE {index}\n"
        f"Source: {source}\n"
        f"Section: {section}\n"
        f"Evidence type: {chunk.get('evidence_type')}\n"
        f"Text: {text_value}"
    )


def _serialize_evidence_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    text_value = _clean_text(chunk.get("chunk_text"))
    return {
        "chunk_id": str(chunk.get("id")),
        "document": chunk.get("doc_title"),
        "page": chunk.get("page"),
        "section": chunk.get("section"),
        "topic": chunk.get("topic"),
        "evidence_type": chunk.get("evidence_type"),
        "preview": text_value[:260],
    }


def _format_source(document: Optional[str], page: Any) -> str:
    if document and page not in (None, ""):
        return f"{document} — page {page}"
    return document or "Unknown source"


def _fallback_evidence_summary(evidence_chunks: List[Dict[str, Any]]) -> str:
    sources = [_format_source(chunk.get("doc_title"), chunk.get("page")) for chunk in evidence_chunks]
    unique_sources = []
    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)
    return "Evidence used: " + "; ".join(unique_sources[:4])


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_study_language(value: Optional[str]) -> str:
    normalized = _clean_text(value).lower()
    if normalized.startswith("it"):
        return "Italian"
    return "English"
