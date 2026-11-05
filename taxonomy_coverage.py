"""Bounded taxonomy requests and explicit, run-local coverage accounting."""
import json
import math
import os
import time
from collections import Counter


REQUEST_ATTEMPTS = 3
REQUEST_MAX_CHARS = 48000


def discovery_budget(batch_count):
    """Bound elapsed discovery work by its preplanned request count."""
    per_batch = float(os.getenv("DOUNO_DISCOVERY_SECONDS_PER_BATCH", "30"))
    minimum = float(os.getenv("DOUNO_DISCOVERY_MIN_SECONDS", "600"))
    maximum = float(os.getenv("DOUNO_DISCOVERY_MAX_SECONDS", "3600"))
    if (not all(math.isfinite(v) and v > 0 for v in (per_batch, minimum, maximum))
            or minimum > maximum or batch_count < 0):
        raise ValueError("Invalid discovery budget configuration")
    return min(maximum, max(minimum, batch_count * per_batch))


def remaining_discovery_seconds(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Topic generation timeout; discovery incomplete")
    return remaining


def permanent_request_error(error):
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 429}


def request_error_diagnostic(error):
    """Never log raw API messages: they can echo credentials or source excerpts."""
    import re
    if isinstance(error, DiscoveryEvidenceError):
        return {"exception": type(error).__name__, **error.diagnostic, "permanent": False}
    def label(value):
        return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,79}", value) else None
    body = getattr(error, "body", None)
    body = body if isinstance(body, dict) else {}
    if isinstance(body.get("error"), dict):
        body = body["error"]
    message = str(body.get("message", "")).lower()
    explanation = "Request failed; raw response omitted to protect credentials and study material."
    if "json" in message and "messages" in message and ("contain" in message or "include" in message):
        explanation = "JSON mode requires messages to explicitly request JSON."
    elif "unsupported" in message and ("parameter" in message or "value" in message):
        explanation = "The API rejected an unsupported request parameter or value."
    elif "context_length" in str(body.get("code", "")) or "maximum context" in message:
        explanation = "Request exceeds the model context limit."
    elif getattr(error, "status_code", None) == 429:
        explanation = "API rate or quota limit reached."
    elif isinstance(getattr(error, "status_code", None), int) and error.status_code >= 500:
        explanation = "API service failure."
    return {"exception": type(error).__name__, "http_status": getattr(error, "status_code", None),
            "api_type": label(body.get("type")), "api_code": label(body.get("code")),
            "explanation": explanation, "permanent": permanent_request_error(error)}


def discovery_source_text(item):
    """Exact source payload used in discovery; internal provenance stays local."""
    return (
        f"SOURCE: {item.get('document') or 'Document'}"
        f" | PAGE: {item.get('page') if item.get('page') is not None else 'n/a'}"
        f" | SOURCE_BLOCK: {item.get('chunk_id')}\n"
        f"{item.get('text') or ''}"
    )


def bounded_batches(items, size=20, max_chars=24000, prompt_builder=None):
    batch, chars = [], 0
    for item in items:
        item_chars = len(discovery_source_text(item))
        if item_chars > max_chars or (prompt_builder is not None and
                                      len(prompt_builder([item])) > REQUEST_MAX_CHARS):
            raise ValueError("Taxonomy input item exceeds request budget; not discarded")
        if batch and (len(batch) >= size or chars + 2 + item_chars > max_chars or
                      (prompt_builder is not None and
                       len(prompt_builder(batch + [item])) > REQUEST_MAX_CHARS)):
            yield batch
            batch, chars = [], 0
        chars += item_chars + (2 if batch else 0)
        batch.append(item)
    if batch:
        yield batch


def validate_topics(data):
    if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
        raise ValueError("Taxonomy response must contain categories")
    count = 0
    for category in data["categories"]:
        if not isinstance(category, dict) or not isinstance(category.get("name"), str) or not category["name"].strip():
            raise ValueError("Invalid taxonomy category")
        if not isinstance(category.get("topics"), list):
            raise ValueError("Invalid taxonomy topics")
        for topic in category["topics"]:
            if not isinstance(topic, dict) or not isinstance(topic.get("title"), str) or not topic["title"].strip():
                raise ValueError("Invalid taxonomy topic title")
            if not isinstance(topic.get("description"), str):
                raise ValueError("Invalid taxonomy topic description")
            count += 1
    if not count:
        raise ValueError("Teaching-material batch returned no topics")
    return data


class DiscoveryEvidenceError(ValueError):
    """Safe structured diagnostics: no source text or provider payload in logs."""
    def __init__(self, code, category_index, topic_index, citation_index=None):
        self.diagnostic = dict(validation_code=code, category_index=category_index,
                               topic_index=topic_index, citation_index=citation_index)
        super().__init__(f"Discovery evidence validation failed: {code} "
                         f"(category={category_index}, topic={topic_index}, citation={citation_index})")


def validate_discovery_evidence(data, batch):
    """Keep exact citations; quarantine invalid claims without retrying discovery.

    Topic structure remains mandatory. Rejected citations cannot establish links;
    candidates must still pass independent assignment and coverage/source review.
    """
    validate_topics(data)
    sources = {str(c["chunk_id"]): c["text"] for c in batch}
    for ci, category in enumerate(data["categories"]):
        for ti, topic in enumerate(category["topics"]):
            # Diagnostics are validator-owned, never trusted from model output.
            topic.pop("source_evidence_rejections", None)
            if "source_evidence" not in topic:
                continue
            accepted, rejected = [], []

            def reject(code, citation_index=None):
                diagnostic = DiscoveryEvidenceError(code, ci, ti, citation_index).diagnostic
                rejected.append(diagnostic)
                print("DISCOVERY_EVIDENCE_REJECTED", json.dumps(diagnostic, sort_keys=True))

            evidence = topic["source_evidence"]
            if not isinstance(evidence, list) or not evidence:
                reject("invalid_evidence_list")
                evidence = []
            for ei, item in enumerate(evidence):
                if not isinstance(item, dict):
                    reject("malformed_citation", ei)
                    continue
                identity, quote = item.get("chunk_id"), item.get("quote")
                # Resolve selected source references locally; never ask the model
                # to recopy or translate evidence. Legacy quotes remain strict.
                if set(item) == {"chunk_id", "support"} and item.get("support") == "direct":
                    if not isinstance(identity, bool) and isinstance(identity, (str, int)):
                        quote = sources.get(str(identity))
                        if isinstance(quote, str) and quote.strip():
                            item = {"chunk_id": identity, "quote": quote.strip()}
                            quote = item["quote"]
                if isinstance(identity, bool) or not isinstance(identity, (str, int)):
                    reject("invalid_source_id_type", ei)
                elif str(identity) not in sources:
                    reject("unknown_source_id", ei)
                elif not isinstance(quote, str) or not quote.strip():
                    reject("invalid_quote", ei)
                elif quote != quote.strip() or quote not in sources[str(identity)]:
                    reject("quote_not_in_source", ei)
                else:
                    # Preserve exact quotes and IDs, including repeated excerpts.
                    accepted.append(item)
            topic["source_evidence"] = accepted
            if rejected:
                topic["source_evidence_rejections"] = rejected
    return data


def request_json(create, prompt, validate=validate_topics, attempts=REQUEST_ATTEMPTS,
                 deadline=None, diagnostic_context=None):
    if len(prompt) > REQUEST_MAX_CHARS:
        raise ValueError("Taxonomy prompt exceeds bounded request budget")
    last_error = None
    for attempt in range(attempts):
        if deadline is not None:
            remaining_discovery_seconds(deadline)
        started = time.monotonic()
        outcome = "failed"
        try:
            response = create(prompt)
            if deadline is not None:
                remaining_discovery_seconds(deadline)
            content = response.choices[0].message.content
            data = json.loads(content.replace("```json", "").replace("```", "").strip())
            validate(data)
            if deadline is not None:
                remaining_discovery_seconds(deadline)
            outcome = "completed"
            return data
        except Exception as error:
            last_error = error
            print("TAXONOMY_REQUEST_ERROR", json.dumps({"attempt": attempt + 1,
                  **request_error_diagnostic(error)}, sort_keys=True))
            if deadline is not None:
                remaining_discovery_seconds(deadline)
            if permanent_request_error(error):
                break
        finally:
            if diagnostic_context is not None:
                print("DISCOVERY_ATTEMPT", json.dumps({**diagnostic_context,
                    "attempt": attempt + 1, "outcome": outcome,
                    "elapsed_seconds": time.monotonic() - started,
                    "remaining_seconds": max(0, deadline - time.monotonic())
                    if deadline is not None else None}, sort_keys=True))
    raise ValueError(f"Taxonomy request failed after {attempt + 1} attempts") from last_error


def candidate_ids(data):
    return [candidate_id for category in data["categories"]
            for topic in category["topics"] for candidate_id in topic.get("candidate_ids", [])]


def validate_candidate_coverage(data, expected):
    actual = candidate_ids(data)
    if Counter(actual) != Counter(expected):
        raise ValueError("Incomplete or duplicate taxonomy candidate accounting")


REVIEW_PAIRS_PER_CANDIDATE = 8
# More independent decisions can share the same evidence payload. Keep the
# unique-topic and context bounds unchanged; do not widen nomination or merges.
REVIEW_PAIRS_PER_REQUEST = 48
REVIEW_CANDIDATES_PER_REQUEST = 24


def normalized_title(value):
    import re
    import unicodedata
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold()))


def nominate_relationships(candidates, protected_categories=False, threshold=.82):
    """Screen all pairs locally, retaining a bounded, prioritized review queue.

    No top-K neighbour exclusion: all threshold/title matches are counted.
    Review capacity is linear in candidate count. Deferred relationships are
    explicitly counted by nomination reason, never reported as reviewed.
    """
    import heapq
    import numpy as np
    if not candidates:
        return [], {"nominated": 0, "deferred": {}, "protected": 0}
    import math
    normalized, invalid = {}, Counter()
    for i, candidate in enumerate(candidates):
        try:
            raw = candidate.get("embedding")
            if not isinstance(raw, (list, tuple, np.ndarray)) or np.ndim(raw) != 1 or not len(raw):
                raise ValueError("malformed")
            values = [float(value) for value in raw]
            if not all(math.isfinite(value) for value in values):
                invalid["non_finite"] += 1
                continue
            scale = max(map(abs, values))
            if not scale:
                invalid["zero_norm"] += 1
                continue
            # Scale first: squaring huge/tiny finite values must not overflow/underflow.
            scaled = [value / scale for value in values]
            norm = math.sqrt(math.fsum(value * value for value in scaled))
            normalized[i] = [value / norm for value in scaled]
        except (TypeError, ValueError, OverflowError):
            invalid["malformed"] += 1
    # A malformed dimension must not disable the remaining valid embeddings.
    dimension = Counter(map(len, normalized.values())).most_common(1)
    dimension = dimension[0][0] if dimension else 0
    vectors = np.zeros((len(candidates), dimension), dtype=np.float64)
    valid = np.zeros(len(candidates), dtype=bool)
    for i, vector in normalized.items():
        if len(vector) != dimension:
            invalid["dimension_mismatch"] += 1
        else:
            vectors[i] = vector
            valid[i] = True
    titles = [normalized_title(c["topic"]) for c in candidates]
    families = {}
    for i, title in enumerate(titles):
        if title:
            families.setdefault(title, []).append(i)
    capacity = REVIEW_PAIRS_PER_CANDIDATE * len(candidates)
    queue, nominated, protected = [], Counter(), 0
    for start in range(0, len(candidates), 128):
        # Avoid BLAS matmul, which can report spurious floating-point exceptions
        # on finite unit vectors on some NumPy/platform builds.
        with np.errstate(under="ignore"):
            scores = np.einsum("ik,jk->ij", vectors[start:start + 128], vectors, optimize=False)
        usable = valid[start:start + 128, None] & valid[None, :] & np.isfinite(scores)
        scores = np.where(usable, np.clip(scores, -1., 1.), -1.)
        for local, row in enumerate(scores):
            i = start + local
            eligible = set(np.flatnonzero(usable[local] & (row >= threshold))) | set(families.get(titles[i], []))
            for j in sorted(j for j in eligible if j > i):
                if (candidates[i].get("chapter_id") != candidates[j].get("chapter_id")
                        or (protected_categories and candidates[i]["category"] != candidates[j]["category"])):
                    protected += 1
                    continue
                exact = candidates[i]["topic"] == candidates[j]["topic"]
                title_match = bool(titles[i]) and titles[i] == titles[j]
                priority = 2 if exact else 1 if title_match else 0
                nominated[priority] += 1
                entry = (priority, float(row[j]), -i, -int(j))
                if len(queue) < capacity:
                    heapq.heappush(queue, entry)
                elif entry > queue[0]:
                    heapq.heapreplace(queue, entry)
    selected = Counter(entry[0] for entry in queue)
    names = {2: "exact_title", 1: "normalized_title", 0: "semantic"}
    relationships = [{"pair": (-a, -b), "reason": names[priority]}
                     for priority, _, a, b in sorted(queue, reverse=True)]
    return relationships, {"nominated": sum(nominated.values()), "protected": protected,
                          "invalid_embeddings": dict(invalid),
                          "deferred": {names[k]: count - selected[k] for k, count in nominated.items()
                                       if count > selected[k]}}


def select_review_evidence(candidate, source_text):
    """Rank windows from validated links only; no assignment changes or AI calls."""
    import re
    title_words = set(normalized_title(candidate["topic"]).split())
    words = set(normalized_title(candidate["description"]).split()) | title_words
    words = {w for w in words if len(w) > 3}
    ranked = []
    for chunk_id in sorted(candidate["evidence_ids"]):
        body = source_text.get(chunk_id, "")
        if not body.strip():
            continue
        best = None
        for offset in range(0, max(1, len(body) - 399), 400):
            excerpt = body[offset:offset + 800]
            tokens = re.findall(r"\w+", excerpt.casefold())
            found = set(tokens) & words
            score = len(found) + 2 * len(found & title_words)
            density = sum(1 + 2 * (w in title_words) for w in tokens if w in words) / max(1, len(tokens))
            choice = (score, density, -offset, excerpt)
            if best is None or choice[:3] > best[:3]:
                best = choice
        ranked.append((best[0], best[1], chunk_id, -best[2], best[3]))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [{"chunk_id": cid, "offset": offset, "excerpt": excerpt}
            for _, _, cid, offset, excerpt in ranked[:2]]


REVIEW_INSTRUCTIONS = """
DUPLICATE PAIR REVIEW. Judge each requested pair independently for any academic discipline.
Titles and similarity only nominate review: they do NOT authorize a merge.
Confirm duplicate only if the COMPLETE learning objectives are substantially equivalent,
supported by BOTH descriptions and validated evidence samples. An existing description
must be sufficient to represent the common objective without combining distinct objectives.
Keep different aspects, theory/application, structure/pathology and complementary objectives
separate. If evidence is insufficient choose uncertain. Never infer equivalence transitively.
Excerpts are samples; unsampled source associations will all be preserved.
Return valid JSON: an object with a "decisions" array containing exactly one decision
for each REQUESTED pair. Every decision must include:
- "input_ids": the two actual input_id strings from that requested pair.
- "decision": exactly one of "duplicate", "distinct", "complementary", "uncertain".
- "reason": a nonempty explanation of the semantic decision.
For "duplicate", also require BOTH of these fields:
- "category": exactly one of this pair's existing category strings.
- "evidence_checks": an object whose keys are EXACTLY the two input_id strings
  in "input_ids", with no additional keys. Each value must be a nonempty array
  of integer chunk_id values copied from THAT input's validated_evidence_samples.
  Cite only excerpts that support the equivalent objective for that input.
Do not use titles, chunk IDs, or literal placeholders such as "a" and "b" as
evidence_checks keys. Do not cite the other input's excerpts or invent IDs.
If either input lacks sufficient supporting evidence, choose "uncertain", not
"duplicate". Non-duplicate decisions do not require category or evidence_checks.
The following is a FORMAT EXAMPLE ONLY: use the actual requested input IDs,
categories and supplied chunk IDs, not these example values:
{"decisions":[{"input_ids":["0","1"],"decision":"duplicate",
"reason":"Both supplied excerpts support the same complete learning objective.",
"category":"Example category","evidence_checks":{"0":[101],"1":[202]}}]}
Do not decide unrequested pairs.
REVIEW INPUT:
"""


def pack_review_batch(ledger, prompt_for):
    """Reuse candidate context, preserving nomination order as a stable tie-break.

    Only scheduling changes: every selected pair keeps its independent decision.
    Reconciliation still traverses the original ledger, not request order.
    """
    pending = [e for e in ledger if e["status"] == "pending"]
    batch, ids = [], set()
    while pending and len(batch) < REVIEW_PAIRS_PER_REQUEST:
        # Prefer pairs needing fewer additional descriptions/evidence excerpts.
        ordered = sorted(range(len(pending)), key=lambda i: len(set(pending[i]["pair"]) - ids))
        chosen = None
        for index in ordered:
            entry = pending[index]
            union = ids | set(entry["pair"])
            if len(union) > REVIEW_CANDIDATES_PER_REQUEST:
                continue
            if len(prompt_for(batch + [entry])) > REQUEST_MAX_CHARS:
                continue
            chosen = index
            ids = union
            break
        if chosen is None:
            break
        batch.append(pending.pop(chosen))
    return batch


def consolidate_candidates(candidates, create, instructions, *, protected_categories=False,
                           source_text=None, diagnostics=None, log_context=None):
    """Bounded pair review, isolated retries, complete-link reconciliation.

    A merged set requires an independently confirmed edge for EVERY member pair.
    No similarity chain, title match, missing decision or exhausted budget merges
    candidates. Original evidence and objective lineage stay local and lossless.
    """
    started = time.monotonic()
    source_text = source_text or {}
    ledger, screening = nominate_relationships(candidates, protected_categories)
    summary = {"candidates_in": len(candidates), **screening, "relationships_selected": len(ledger),
               "pair_budget": REVIEW_PAIRS_PER_CANDIDATE * len(candidates),
               "review_group_budget": len(candidates),
               "review_groups": 0, "ai_calls": 0, "validation_failures": 0,
               "request_characters": 0, "input_tokens": 0, "output_tokens": 0,
               "calls_with_usage": 0, "request_seconds": 0.0,
               "first_pass_calls": 0, "retry_calls": 0,
               "first_pass_seconds": 0.0, "retry_seconds": 0.0}
    def emit(event, **values):
        print("TAXONOMY_CONSOLIDATION", json.dumps({**(log_context or {}), "event": event, **values}, sort_keys=True))
    emit("screening", **summary)
    inputs = {i: {"input_id": str(i), "title": c["topic"], "description": c["description"],
                  "category": c["category"], "validated_evidence_samples": select_review_evidence(c, source_text)}
              for i, c in enumerate(candidates)}
    def prompt_for(entries):
        ids = sorted({i for entry in entries for i in entry["pair"]})
        return instructions + REVIEW_INSTRUCTIONS + json.dumps({
            "candidates": [inputs[i] for i in ids],
            "relationships": [[str(i) for i in entry["pair"]] for entry in entries]}, ensure_ascii=False)
    for entry in ledger:
        entry.update(status="pending", attempts=0)
        if any(not inputs[i]["validated_evidence_samples"] for i in entry["pair"]):
            entry.update(status="unresolved", code="no_validated_excerpt")
        elif len(prompt_for([entry])) > REQUEST_MAX_CHARS:
            entry.update(status="unresolved", code="context_budget")

    # Retry rounds pool only unresolved pairs across first-pass batches. Valid
    # decisions are never requested again; reconciliation keeps ledger order.
    group_budget = len(candidates)
    summary["retry_groups"] = 0
    for review_round in range(REQUEST_ATTEMPTS):
        while True:
            eligible = [e for e in ledger if e["status"] == "pending"
                        and e["attempts"] == review_round]
            if not eligible or (review_round == 0 and summary["review_groups"] >= group_budget):
                break
            batch = pack_review_batch(eligible, prompt_for)
            if not batch:
                break
            if review_round == 0:
                summary["review_groups"] += 1
            else:
                summary["retry_groups"] += 1
            pending = batch[:]
            group_started = time.monotonic()
            group_calls_before = summary["ai_calls"]
            summary["ai_calls"] += 1
            phase = "first_pass" if review_round == 0 else "retry"
            summary[phase + "_calls"] += 1
            for entry in pending:
                entry["attempts"] += 1
            response_failure = None
            permanent_failure = False
            try:
                def validate_shape(data):
                    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
                        raise ValueError("Missing pair decisions")
                def measured_create(prompt):
                    summary["request_characters"] += len(prompt)
                    call_started = time.monotonic()
                    try:
                        response = create(prompt)
                        usage = getattr(response, "usage", None)
                        if usage is not None:
                            summary["calls_with_usage"] += 1
                            summary["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                            summary["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                        return response
                    finally:
                        elapsed = time.monotonic() - call_started
                        summary["request_seconds"] += elapsed
                        summary[phase + "_seconds"] += elapsed
                data = request_json(measured_create, prompt_for(pending), validate_shape, attempts=1)
                by_pair = {}
                requested = {entry["pair"] for entry in pending}
                for record in data["decisions"]:
                    if not isinstance(record, dict):
                        continue
                    pair = record.get("input_ids")
                    if (not isinstance(pair, list) or len(pair) != 2 or
                            any(not isinstance(i, str) or not i.isdecimal() for i in pair)):
                        continue
                    key = tuple(sorted(map(int, pair)))
                    if key in requested:
                        by_pair.setdefault(key, []).append(record)
            except Exception as error:
                by_pair = {}
                permanent_failure = permanent_request_error(error.__cause__ or error)
                response_failure = ("invalid_response" if isinstance(error.__cause__, (ValueError, TypeError))
                                    else "request_failure")
            retry = []
            for entry in pending:
                pair = entry["pair"]
                records = by_pair.get(pair, [])
                code = None
                if len(records) != 1:
                    code = (response_failure or "missing_decision") if not records else "ambiguous_decisions"
                else:
                    record = records[0]
                    decision = record.get("decision")
                    if not isinstance(decision, str) or decision not in {"duplicate", "distinct", "complementary", "uncertain"}:
                        code = "invalid_decision"
                    elif not isinstance(record.get("reason"), str) or not record["reason"].strip():
                        code = "missing_reason"
                    elif decision == "duplicate":
                        evidence = record.get("evidence_checks")
                        if not isinstance(record.get("category"), str) or record["category"] not in {candidates[i]["category"] for i in pair}:
                            code = "invalid_category"
                        elif not isinstance(evidence, dict) or set(evidence) != {str(i) for i in pair}:
                            code = "missing_evidence_review"
                        else:
                            for i in pair:
                                cited = evidence[str(i)]
                                allowed = {s["chunk_id"] for s in inputs[i]["validated_evidence_samples"]}
                                if (not isinstance(cited, list) or not cited or
                                        any(type(cid) is not int or cid not in allowed for cid in cited)):
                                    code = "unsupported_evidence"
                                    break
                if code:
                    entry["code"] = code
                    summary["validation_failures"] += 1
                    retry.append(entry)
                else:
                    entry.update(status=record["decision"], code=record["decision"])
                    if record["decision"] == "duplicate":
                        entry["category"] = record["category"]
            pending = retry
            for entry in pending:
                entry["status"] = ("unresolved" if permanent_failure or entry["attempts"] >= REQUEST_ATTEMPTS
                                   else "pending")
            emit("review", group=summary["review_groups"], review_round=review_round + 1,
                 elapsed_seconds=round(time.monotonic() - group_started, 3),
                 ai_calls=summary["ai_calls"] - group_calls_before,
                 remaining_pairs=sum(e["status"] == "pending" for e in ledger), relationships=[
                {k: e[k] for k in ("pair", "reason", "status", "code", "attempts")} for e in batch])
    for entry in ledger:
        if entry["status"] == "pending":
            entry.update(status="unresolved", code="review_budget")

    confirmed = {e["pair"] for e in ledger if e["status"] == "duplicate"}
    clusters = {i: {i} for i in range(len(candidates))}
    owner = {i: i for i in range(len(candidates))}
    for entry in ledger:
        if entry["status"] != "duplicate":
            continue
        a, b = entry["pair"]
        left, right = owner[a], owner[b]
        if left == right:
            continue
        if not all(tuple(sorted((x, y))) in confirmed for x in clusters[left] for y in clusters[right]):
            continue
        if len({candidates[i].get("chapter_id") for i in clusters[left] | clusters[right]}) != 1:
            raise ValueError("Cross-chapter consolidation rejected")
        members = clusters.pop(right)
        clusters[left].update(members)
        for i in members:
            owner[i] = left

    categories = {}
    for members in sorted(clusters.values(), key=min):
        ids = sorted(members)
        # Equivalent objectives need no extra synthesis call. Reuse the most
        # detailed original wording, avoiding concatenation and invented content.
        representative = max(ids, key=lambda i: (len(candidates[i]["description"]), -i))
        c = candidates[representative]
        votes = Counter(e["category"] for e in ledger if e["status"] == "duplicate"
                        and set(e["pair"]) <= members)
        category = max(votes, key=lambda name: (votes[name], name == c["category"], name)) if votes else c["category"]
        topic = {"title": c["topic"], "description": c["description"], "importance": max(candidates[i]["importance"] for i in ids),
                 "candidate_ids": [str(i) for i in ids], "assignment_id": c["assignment_id"],
                 "embedding": c["embedding"], "original_objectives": [candidates[i]["description"] for i in ids],
                 "evidence_ids": sorted(set().union(*(set(candidates[i]["evidence_ids"]) for i in ids)))}
        if c.get("chapter_id"):
            topic["chapter_id"] = c["chapter_id"]
        if category != c["category"]:
            topic.pop("embedding")  # recompute only when the embedding text changed
        categories.setdefault(category, []).append(topic)
    result = {"categories": [{"name": name, "topics": topics} for name, topics in categories.items()]}
    validate_candidate_coverage(result, [str(i) for i in range(len(candidates))])
    summary.update(candidates_out=sum(len(c["topics"]) for c in result["categories"]),
                   confirmed_relationships=len(confirmed),
                   confirmed_not_coalesced=sum(owner[a] != owner[b] for a, b in confirmed),
                   merged_groups=sum(len(m) > 1 for m in clusters.values()),
                   outcomes=dict(Counter(e["code"] for e in ledger)))
    merged_sets = [sorted(m) for m in clusters.values() if len(m) > 1]
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    emit("summary", **summary, merged_sets=merged_sets)
    if diagnostics is not None:
        diagnostics.update(summary=summary, relationships=ledger,
                           merged_sets=merged_sets)
    return result
