import json
import re
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import text


MAX_EVIDENCE_CHUNKS = 8
MAX_SHARED_EVIDENCE_CHUNKS = 4
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

    support_level = _normalize_support_level(parsed.get("support_level"))
    relationship_form = _clean_text(parsed.get("relationship_form"))
    explanation = _clean_explanation_text(parsed.get("explanation"))
    why_connected = _clean_explanation_text(parsed.get("why_connected"))
    study_relevance = _clean_explanation_text(parsed.get("study_relevance"))
    evidence_summary = _clean_text(parsed.get("evidence_summary"))

    is_italian = _normalize_study_language(study_language) == "Italian"

    if not support_level:
        support_level = "PARTIAL" if (explanation or why_connected or study_relevance) else "INSUFFICIENT"

    if not relationship_form:
        relationship_form = "insufficient" if support_level == "INSUFFICIENT" else "undetermined"

    if support_level == "INSUFFICIENT" and not explanation:
        explanation = _fallback_insufficient_explanation(is_italian)
    elif not explanation and why_connected and study_relevance:
        explanation = f"{why_connected}\n\n{study_relevance}"
    elif not explanation:
        explanation = _fallback_explanation(topic_a, topic_b, is_italian)

    if not why_connected:
        why_connected = explanation
    if not study_relevance:
        study_relevance = explanation
    if not evidence_summary:
        evidence_summary = _fallback_evidence_summary(evidence_chunks, is_italian)

    return {
        "topic_a": topic_a,
        "topic_b": topic_b,
        "support_level": support_level,
        "relationship_form": relationship_form,
        "explanation": explanation,
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
            "limit": MAX_SHARED_EVIDENCE_CHUNKS,
        },
    ).mappings().fetchall()

    supporting_rows = db.execute(
        text(
            """
            select
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
            order by
                case
                    when tc.topic_id = :topic_a_id then 0
                    when tc.topic_id = :topic_b_id then 1
                    else 2
                end asc,
                char_length(c.chunk_text) desc,
                c.id asc
            limit :limit
            """
        ),
        {
            "project_id": project_id,
            "topic_a_id": topic_a_id,
            "topic_b_id": topic_b_id,
            "limit": MAX_EVIDENCE_CHUNKS * 2,
        },
    ).mappings().fetchall()

    combined: Dict[str, Dict[str, Any]] = {}
    selected: List[Dict[str, Any]] = []

    def append_unique(row: Any) -> None:
        row_dict = dict(row)
        row_id = str(row_dict["id"])
        if row_id in combined:
            return
        combined[row_id] = row_dict
        selected.append(row_dict)

    for row in shared_rows:
        append_unique(row)

    for desired_type in ("topic_a", "topic_b"):
        for row in supporting_rows:
            if row["evidence_type"] == desired_type:
                append_unique(row)
                break

    for row in supporting_rows:
        append_unique(row)
        if len(selected) >= MAX_EVIDENCE_CHUNKS:
            break

    return selected[:MAX_EVIDENCE_CHUNKS]


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
    Teach a student how two study topics are connected using only the evidence
    below.

    OUTPUT LANGUAGE:
    {study_language}

    The output language above is mandatory for explanation and
    evidence_summary. Do not infer the answer language from topic titles,
    category names, document language, source chunks, or examples. If the
    uploaded study material is in a different language, still write the
    student-facing text entirely in {study_language}.

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

    Student intent:
    "I understand these two topics individually, but I do not understand why
    DOUNO connected them. Teach me the connection."

    Critical diagnostic task:
    The Relationship Engine has proposed this relationship as a candidate.
    Your job is NOT to justify that decision. Your job is to independently
    determine whether the supplied study evidence supports a pedagogically
    meaningful connection between Topic A and Topic B.

    First assess evidence support:
    - STRONG: the supplied evidence clearly establishes a meaningful
      relationship between Topic A and Topic B and allows that relationship to
      be explained without inventing missing reasoning.
    - PARTIAL: the supplied evidence supports a useful relationship, but it is
      indirect, comparative, contextual, or only partially establishes the
      bridge. Explain only what is supported and do not imply a stronger direct
      causal or mechanistic relationship than the evidence demonstrates.
    - INSUFFICIENT: the supplied evidence does not provide enough information
      to explain a meaningful relationship without speculation, generic
      textbook knowledge, or forced reasoning.

    Important distinctions:
    - Distinguish same concept from related concept.
    - Distinguish part-of from causes.
    - Distinguish correlation from mechanism.
    - Distinguish shared context from direct dependency.
    - Distinguish similar phenotype from same genetic mechanism.
    - Distinguish variant heterogeneity from locus heterogeneity.
    - Distinguish homozygosity / identity by descent from compound
      heterozygosity.
    Do not merge related but distinct concepts merely to justify an edge.

    Before writing, reason internally about the pedagogical form of the
    relationship. Possible forms include causal, mechanistic, structural /
    part-of, prerequisite, functional, regulatory, comparative, sequential,
    shared process, shared principle, and contextual. Use this internal choice
    only to decide how to explain the connection. Do not expose it as a label.

    Pedagogical goal:
    - Start directly with the key connection. Avoid generic openings such as
      "Understanding the relationship between...", "The relationship can be
      seen...", or "Both topics are important...".
    - Explain the bridge between the concepts, not two separate definitions.
    - Build a clear reasoning path from Topic A to Topic B.
    - Adapt the structure to the relationship form supported by the evidence:
      causal, mechanistic, structural, comparative, prerequisite,
      shared-principle, shared-process, sequential, functional, regulatory, or
      contextual.
    - Do not force causal or mechanistic language when the relationship is not
      causal or mechanistic.
    - For cross-category relationships, make the conceptual bridge across the
      categories explicit. Do not merely say they belong to different
      categories.
    - Naturally include how understanding this connection helps the student
      reason about the subject, but do not create a separate study-relevance
      section and do not append generic phrases.

    Avoid this failure mode:
    "Topic A is... Topic B is... Therefore they are related."

    Definitions are allowed only briefly when they are necessary to explain the
    bridge. Most of the answer must explain the mechanism, dependency,
    comparison, shared principle, sequence, or context that connects them.

    Style:
    - Write 2-4 short paragraphs when the evidence supports it.
    - Aim for roughly 120-220 words when that gives enough room to teach the
      connection clearly.
    - Each paragraph must have one clear teaching purpose.
    - Be explanatory, reasoned, and suitable for a university student.
    - Prefer progressive explanation over dense academic prose.
    - Avoid repetition, rhetorical padding, generic academic conclusions, and
      phrases like "enhancing comprehension as a whole" or "critical for
      appreciating" unless truly necessary.
    - When useful, end with one concise synthesis sentence beginning naturally
      with "In other words:" in {study_language}. Do not force it.
    - Write explanation and evidence_summary entirely in {study_language}.
    - Return support_level as one of the exact uppercase enum values.
    - Return relationship_form as one of the exact lowercase taxonomy tokens
      listed in the JSON shape.
    - Do not expose or mention relationship metrics.
    - Use evidence internally, but never cite internal evidence identifiers in
      the explanation.
    - Do not output "(EVIDENCE 1)", "(EVIDENCE 1, EVIDENCE 2)", retrieval IDs,
      chunk IDs, or similar internal references.
    - Do not say the topics are connected merely because they are semantically similar.
    - Do not add facts that are not supported by the source evidence.
    - If the evidence is limited, acknowledge that the connection is based on
      limited evidence.
    - If support is INSUFFICIENT, do not generate a plausible explanation.
      Return a concise student-facing message equivalent to: "The available
      study material does not provide enough evidence to explain a direct
      relationship between these topics." Adapt it naturally to
      {study_language}. Do not mention embeddings, semantic similarity,
      thresholds, or the Relationship Engine.

    Return valid JSON with exactly these fields:
    {{
      "support_level": "STRONG | PARTIAL | INSUFFICIENT",
      "relationship_form": "structural | part-of | comparative | causal | mechanistic | prerequisite | regulatory | functional | sequential | shared-process | shared-principle | contextual | indirect | insufficient",
      "explanation": "...",
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


def _fallback_evidence_summary(
    evidence_chunks: List[Dict[str, Any]],
    is_italian: bool,
) -> str:
    sources = [_format_source(chunk.get("doc_title"), chunk.get("page")) for chunk in evidence_chunks]
    unique_sources = []
    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)
    prefix = "Evidenze utilizzate: " if is_italian else "Evidence used: "
    return prefix + "; ".join(unique_sources[:4])


def _fallback_explanation(
    topic_a: Dict[str, Any],
    topic_b: Dict[str, Any],
    is_italian: bool,
) -> str:
    if is_italian:
        return (
            f"Il materiale caricato contiene evidenze che collegano {topic_a['topic']} "
            f"e {topic_b['topic']}, ma non è stato possibile generare in modo sicuro "
            "una spiegazione pedagogica completa dal contesto disponibile."
        )

    return (
        f"The uploaded material contains evidence connecting {topic_a['topic']} "
        f"and {topic_b['topic']}, but a safe pedagogical explanation could not "
        "be generated from the available context."
    )


def _fallback_insufficient_explanation(is_italian: bool) -> str:
    if is_italian:
        return (
            "Il materiale di studio disponibile non fornisce evidenze sufficienti "
            "per spiegare un collegamento diretto tra questi due topic."
        )

    return (
        "The available study material does not provide enough evidence to "
        "explain a direct relationship between these topics."
    )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clean_explanation_text(value: Any) -> str:
    text_value = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text_value = re.sub(
        r"\s*\(\s*EVIDENCE\s+\d+(?:\s*,\s*EVIDENCE\s+\d+)*\s*\)",
        "",
        text_value,
        flags=re.IGNORECASE,
    )
    paragraphs = [
        " ".join(paragraph.split()).strip()
        for paragraph in re.split(r"\n\s*\n+", text_value)
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs)


def _normalize_support_level(value: Any) -> str:
    normalized = _clean_text(value).upper()
    if normalized in {"STRONG", "PARTIAL", "INSUFFICIENT"}:
        return normalized
    return ""


def _normalize_study_language(value: Optional[str]) -> str:
    normalized = _clean_text(value).lower()
    if normalized.startswith("it"):
        return "Italian"
    return "English"
