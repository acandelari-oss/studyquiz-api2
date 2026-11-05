"""Read-only assignment planning; publication remains owned by the upload transaction.

Use PostgreSQL's vector operator (not a Python approximation) and the legacy
score callback. Topic IDs are allocated before planning so ranking ties use the
same IDs that will be published for unmerged candidates.
"""
import hashlib
import json

from sqlalchemy import text


def initial_source_read(db, load):
    """Retry only the initial read transaction, once, on a classified disconnect.

    Call only after status writes have committed. No AI work or publication is
    permitted in load; PostgreSQL enforces read-only access for each attempt.
    """
    import time
    from sqlalchemy.exc import DBAPIError

    if db.in_transaction():
        raise ValueError("Initial source read requires a clean transaction boundary")
    for attempt in (1, 2):
        started = time.monotonic()
        previous = started

        def checkpoint(phase):
            nonlocal previous
            now = time.monotonic()
            print("TAXONOMY SOURCE READ:", {"attempt": attempt, "phase": phase,
                  "seconds": round(now - previous, 3)})
            previous = now

        try:
            if db.get_bind().dialect.name == "postgresql":
                db.execute(text("SET TRANSACTION READ ONLY"))
            result = load(checkpoint)
            db.rollback()  # release reads before any model calls
            checkpoint("complete")
            return result
        except DBAPIError as error:
            db.rollback()
            print("TAXONOMY SOURCE READ FAILED:", {
                "attempt": attempt, "seconds": round(time.monotonic() - started, 3),
                "disconnect": bool(error.connection_invalidated),
                "retry": bool(error.connection_invalidated and attempt == 1)})
            if not error.connection_invalidated or attempt == 2:
                raise
        except Exception:
            db.rollback()
            raise


def source_scope(project_id, module_id=None, document_id=None):
    params = {"project_id": project_id}
    join, clause = "", ""
    if module_id:
        join = "join documents d on d.id=c.document_id and d.project_id=c.project_id"
        clause = "and d.module_id=:module_id"
        params["module_id"] = module_id
    if document_id:
        clause += " and c.document_id=:document_id"
        params["document_id"] = str(document_id)
    return join, clause, params


def source_snapshot(db, project_id, module_id=None, document_id=None):
    # Inspect only column names, retaining compatibility before the optional
    # provenance migration. Never serialize entire document/chunk records.
    from sqlalchemy import inspect
    if db.get_bind().dialect.name == "postgresql":
        columns = set(db.execute(text("""
            select c.relname, a.attname
            from pg_attribute a join pg_class c on c.oid=a.attrelid
            where a.attrelid in (to_regclass('chunks'), to_regclass('documents'))
              and a.attnum > 0 and not a.attisdropped
              and a.attname in ('structural_provenance', 'chapter_structure')
        """)))
        chunk_columns = {name for table, name in columns if table == "chunks"}
        document_columns = {name for table, name in columns if table == "documents"}
    else:
        inspector = inspect(db.connection())
        chunk_columns = {c["name"] for c in inspector.get_columns("chunks")}
        document_columns = {c["name"] for c in inspector.get_columns("documents")}
    provenance = "c.structural_provenance" if "structural_provenance" in chunk_columns else "NULL"
    structure = "d.chapter_structure" if "chapter_structure" in document_columns else "NULL"
    join, clause, params = source_scope(project_id, module_id, document_id)
    document_filter = ""
    if module_id:
        document_filter += " and d.module_id=:module_id"
    if document_id:
        document_filter += " and d.id=:document_id"
    if db.get_bind().dialect.name == "postgresql":
        # Versioned, internal fingerprints: JSONB arrays preserve field boundaries
        # and nulls. Hash large values on the server, never transfer them merely
        # to hash them. All comparisons in an attempt use this same format.
        def digest(expression):
            return "encode(sha256(convert_to((" + expression + ")::text,'UTF8')),'hex')"

        metadata = {str(row[0]): row[1] for row in db.execute(text(f"""
            select d.id, {digest('jsonb_build_array(' + structure + ')')}
            from documents d where d.project_id=:project_id {document_filter}
        """), params)}
        fields = "c.id,c.document_id,c.page,c.chunk_text,c.section,c.embedding::text," + provenance
        rows = db.execute(text(f"""
            select c.id, c.document_id, {digest('jsonb_build_array(' + fields + ')')}
            from chunks c {join} where c.project_id=:project_id {clause}
            and c.chunk_role='teaching' order by c.id
        """), params)
        return {row[0]: hashlib.sha256(json.dumps(
                    ["pg-source-v2", row[2], metadata.get(str(row[1]))],
                    ensure_ascii=False).encode()).hexdigest() for row in rows}

    # One metadata row per document, not a correlated whole-row conversion for
    # every chunk. IDs are normalized only as Python lookup keys.
    metadata = {str(row[0]): row[1] for row in db.execute(text(f"""
        select d.id, {structure} from documents d
        where d.project_id=:project_id {document_filter}
    """), params)}
    rows = db.execute(text(f"""
        select c.id, c.document_id, c.page, c.chunk_text, c.section, c.embedding::text,
               {provenance}
        from chunks c {join} where c.project_id=:project_id {clause}
        and c.chunk_role='teaching' order by c.id
    """), params)
    return {row[0]: hashlib.sha256(json.dumps(
                list(row) + [metadata.get(str(row[1]))], ensure_ascii=False).encode()).hexdigest()
            for row in rows}


def plan_assignments(db, candidates, project_id, score, *, module_id=None,
                     document_id=None, primary=.54, max_topics=3, rescue=.46,
                     pair_budget=30000, chapter_ownership=None):
    """Return candidate->chunk sets, without writing topics, links, or chunk roles.

    Primary top-k is global across candidate batches. Rescue selects the best
    eligible chunk for each orphan, including the legacy lowest-chunk-ID tie.
    A planning error fails the replacement instead of publishing partial links.
    """
    import time
    planning_started = time.monotonic()
    snapshot = source_snapshot(db, project_id, module_id, document_id)
    if not candidates or not snapshot:
        raise ValueError("Assignment planning requires candidates and teaching chunks")
    ids = [c["assignment_id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate assignment identities")
    join, clause, scope = source_scope(project_id, module_id, document_id)
    if chapter_ownership is not None:
        if set(chapter_ownership) != set(snapshot) or any(
                not c.get("chapter_id") or c["chapter_id"] not in set(chapter_ownership.values())
                for c in candidates):
            raise ValueError("Incomplete chapter assignment scope")
    candidate_chapters = {c["assignment_id"]: c.get("chapter_id") for c in candidates}
    # Rescue both sides of the relation: a topic's best chunk alone does not
    # guarantee coverage of its other supporting source chunks.
    source_batches = {c["assignment_id"]: set(c.get("discovery_chunk_ids") or ()) for c in candidates}
    citations = {}
    for candidate in candidates:
        cited = {}
        for item in candidate.get("source_evidence") or []:
            if not isinstance(item, dict):
                raise ValueError("Malformed discovery evidence at assignment")
            identity, quote = item.get("chunk_id"), item.get("quote")
            matches = [i for i in snapshot if str(i) == str(identity)]
            if (isinstance(identity, bool) or not isinstance(identity, (str, int)) or len(matches) != 1
                    or matches[0] not in source_batches[candidate["assignment_id"]]
                    or not isinstance(quote, str) or not quote.strip()
                    or quote != quote.strip()):
                raise ValueError("Invalid discovery citation scope")
            chunk_id = matches[0]
            if chapter_ownership is not None and chapter_ownership[chunk_id] != candidate.get("chapter_id"):
                raise ValueError("Discovery citation crosses chapter boundary")
            cited.setdefault(chunk_id, []).append(quote)
        citations[candidate["assignment_id"]] = cited
    source_texts = {}
    candidate_by_id = {c["assignment_id"]: c for c in candidates}
    score_queries = 0
    best_for_chunk = {}
    best = {}
    links = {i: set() for i in ids}
    chunk_ids = sorted(snapshot)
    for offset in range(0, len(chunk_ids), 200):
        chunk_batch = chunk_ids[offset:offset + 200]
        params = dict(scope)
        chunks = []
        for i, chunk_id in enumerate(chunk_batch):
            params[f"c{i}"] = chunk_id
            chunks.append(f":c{i}")
        source_rows = db.execute(text(f"""
            select c.id, c.chunk_text, c.section from chunks c {join}
            where c.project_id=:project_id {clause} and c.chunk_role='teaching'
            and c.id in ({','.join(chunks)}) order by c.id
        """), params).fetchall()
        sources = {cid: (body, section) for cid, body, section in source_rows}
        if len(source_rows) != len(chunk_batch) or set(sources) != set(chunk_batch):
            raise ValueError("Incomplete assignment planning sources")
        ranked = {i: [] for i in chunk_batch}
        width = max(1, min(100, pair_budget // len(chunk_batch)))
        for start in range(0, len(candidates), width):
            group = candidates[start:start + width]
            params = dict(scope)
            chunks = []
            for i, chunk_id in enumerate(chunk_batch):
                params[f"c{i}"] = chunk_id
                chunks.append(f":c{i}")
            values = []
            for i, candidate in enumerate(group):
                params.update({f"id{i}": candidate["assignment_id"],
                               f"v{i}": json.dumps(list(candidate["embedding"]))})
                values.append(f"(CAST(:id{i} AS text), CAST(:v{i} AS vector))")
            rows = db.execute(text(f"""
                with candidate(id, embedding) as (values {','.join(values)})
                select c.id, t.id,
                       c.embedding <#> t.embedding
                from chunks c {join} cross join candidate t
                where c.project_id=:project_id {clause} and c.chunk_role='teaching'
                and c.id in ({','.join(chunks)}) order by c.id,t.id
            """), params).fetchall()
            score_queries += 1
            if len(rows) != len(chunk_batch) * len(group):
                raise ValueError("Incomplete assignment planning scores")
            for chunk_id, topic_id, distance in rows:
                body, section = sources[chunk_id]
                candidate = candidate_by_id[topic_id]
                title, topic_section = candidate["topic"], candidate["source_section"]
                if chapter_ownership is not None and chapter_ownership[chunk_id] != candidate_chapters[topic_id]:
                    continue
                source_texts[chunk_id] = body
                quotes = citations[topic_id].get(chunk_id)
                if quotes:
                    if any(quote not in body for quote in quotes):
                        raise ValueError("Discovery citation changed before assignment")
                    links[topic_id].add(chunk_id)
                value = score(body, section, title, topic_section, distance)
                if value is None:
                    continue
                if (chapter_ownership is not None and chunk_id in source_batches[topic_id]
                        and value >= rescue):
                    previous = best_for_chunk.get(chunk_id)
                    if previous is None or (-value, str(topic_id)) < (-previous[0], str(previous[1])):
                        best_for_chunk[chunk_id] = (value, topic_id)
                if value >= primary:
                    ranked[chunk_id].append((value, topic_id))
                if topic_id not in best or (value, -int(chunk_id)) > best[topic_id][0]:
                    best[topic_id] = ((value, -int(chunk_id)), chunk_id)
            # Bound retained primary scores without changing global ranking.
            for chunk_id in chunk_batch:
                ranked[chunk_id] = sorted(ranked[chunk_id], key=lambda x: (-x[0], str(x[1])))[:max_topics]
        for chunk_id, matches in ranked.items():
            for _, topic_id in matches:
                links[topic_id].add(chunk_id)
    for topic_id in ids:
        if not links[topic_id] and topic_id in best and best[topic_id][0][0] >= rescue:
            links[topic_id].add(best[topic_id][1])
    if chapter_ownership is not None:
        covered = set().union(*links.values())
        for chunk_id, (_, topic_id) in best_for_chunk.items():
            if chunk_id not in covered:
                links[topic_id].add(chunk_id)
    if chapter_ownership is not None:
        inherit_exact_source_overlaps(links, candidates, source_texts, chapter_ownership)
    if source_snapshot(db, project_id, module_id, document_id) != snapshot:
        raise ValueError("Source changed during assignment planning")
    print("TAXONOMY ASSIGNMENT PLANNING:", {"candidates": len(candidates),
          "chunks": len(chunk_ids), "score_queries": score_queries,
          "seconds": round(time.monotonic() - planning_started, 3)})
    return links, snapshot


def inherit_exact_source_overlaps(links, candidates, source_texts, chapter_ownership):
    """Carry substantial exact overlap from an already verified quotation only.

    No page/section proximity inference; tiny or multiply attributable fragments
    remain unresolved. The 40-character floor rejects weak substring matches.
    """
    covered = set().union(*links.values()) if links else set()
    for chunk_id, body in source_texts.items():
        if chunk_id in covered or not isinstance(body, str) or len(body.strip()) < 40:
            continue
        matches = set()
        for candidate in candidates:
            identity = candidate["assignment_id"]
            if (chunk_id not in candidate.get("discovery_chunk_ids", []) or
                    chapter_ownership.get(chunk_id) != candidate.get("chapter_id")):
                continue
            for item in candidate.get("source_evidence") or []:
                cited = next((i for i in links[identity] if str(i) == str(item["chunk_id"])), None)
                quote = item["quote"]
                if (cited is not None and quote in source_texts.get(cited, "")
                        and body.strip() in quote):
                    matches.add(identity)
        if len(matches) == 1:
            links[next(iter(matches))].add(chunk_id)


def save_discovery_checkpoint(candidates, snapshot, organization, project_id,
                              module_id, document_id, chapter_ownership):
    """Opt-in local recovery artifact; never automatically trusts or applies it.

    Contains study content. Keep outside public/static folders. Atomic mode-0600
    files retain source fingerprints/settings for explicit future validation.
    """
    import os
    import tempfile
    from pathlib import Path
    directory = os.environ.get("DOUNO_DISCOVERY_CHECKPOINT_DIR")
    if not directory:
        return None
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {"version": 1, "project_id": str(project_id),
               "module_id": str(module_id) if module_id else None,
               "document_id": str(document_id) if document_id else None,
               "source_snapshot": snapshot, "organization": organization,
               "chapter_ownership": chapter_ownership, "candidates": candidates}
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # Content addressing avoids collisions across attempts and untrusted path IDs.
    destination = root / (hashlib.sha256(content).hexdigest() + ".json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".discovery-", delete=False) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    print("DISCOVERY_CHECKPOINT_SAVED", str(destination))
    return str(destination)


def validate_chapter_coverage(candidates, chunks, chapter_ownership):
    """Report missing evidence separately from uncovered source; never invent links."""
    sources = {c["chunk_id"]: c for c in chunks}
    covered = set()
    unsupported = []
    invalid = []
    for index, candidate in enumerate(candidates):
        evidence = set(candidate.get("evidence_ids") or ())
        covered.update(evidence)
        if not evidence:
            unsupported.append({"candidate_index": index,
                "assignment_id": candidate.get("assignment_id"), "title": candidate.get("topic"),
                "chapter_id": candidate.get("chapter_id"),
                "discovery_chunk_ids": candidate.get("discovery_chunk_ids", [])})
        for chunk_id in evidence:
            if chunk_id not in sources or chapter_ownership.get(chunk_id) != candidate.get("chapter_id"):
                invalid.append({"candidate_index": index, "chunk_id": chunk_id})
    uncovered = [{"chunk_id": chunk_id, "page": source.get("page"),
                  "chapter_id": chapter_ownership.get(chunk_id)}
                 for chunk_id, source in sources.items() if chunk_id not in covered]
    report = {"candidate_count": len(candidates), "eligible_chunks": len(sources),
              "covered_chunks": len(covered & set(sources)),
              "unsupported_candidates": unsupported, "uncovered_chunks": uncovered,
              "invalid_links": invalid}
    print("CHAPTER_EVIDENCE_COVERAGE", json.dumps(report, sort_keys=True))
    if unsupported or uncovered or invalid:
        raise ValueError("Chapter assignment has incomplete validated source coverage: "
                         f"{len(unsupported)} unsupported candidates, {len(uncovered)} uncovered chunks, "
                         f"{len(invalid)} invalid links")
    return report


def validate_evidence(data, candidates, eligible_ids):
    """Exact lineage and union validation, including unchanged candidates."""
    from taxonomy_coverage import validate_candidate_coverage
    validate_candidate_coverage(data, [str(i) for i in range(len(candidates))])
    before, after, assignment_ids = set(), set(), set()
    eligible_ids = set(eligible_ids)
    for candidate in candidates:
        before.update(candidate["evidence_ids"])
    for category in data["categories"]:
        for topic in category["topics"]:
            expected = set().union(*(set(candidates[int(i)]["evidence_ids"])
                                    for i in topic["candidate_ids"]))
            if set(topic["evidence_ids"]) != expected or not expected <= eligible_ids:
                raise ValueError("Consolidation changed validated source evidence")
            valid_ids = {candidates[int(i)]["assignment_id"] for i in topic["candidate_ids"]}
            if topic["assignment_id"] not in valid_ids or topic["assignment_id"] in assignment_ids:
                raise ValueError("Invalid consolidated assignment identity")
            assignment_ids.add(topic["assignment_id"])
            objectives = [candidates[int(i)]["description"] for i in topic["candidate_ids"]]
            if topic["original_objectives"] != objectives:
                raise ValueError("Consolidation lost original objectives")
            after.update(expected)
    if before != after:
        raise ValueError("Consolidation introduced uncovered source chunks")


def publish_prepared_topic_links(db, project_id, module_id, document_id, links, deadline):
    """Insert a validated plan into newly inserted topics, never commit here."""
    import time
    def check():
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Taxonomy publication deadline exceeded")
    check()
    params = {"project_id": project_id}
    clause = ""
    if module_id:
        params["module_id"] = module_id
        clause = "and module_id=:module_id"
    if document_id:
        params["document_id"] = str(document_id)
        clause += " and document_id=:document_id"
    topic_ids = {str(r[0]) for r in db.execute(text(
        f"select id from topics where project_id=:project_id {clause}"), params)}
    if set(links) != topic_ids:
        raise ValueError("Prepared assignments do not match publication scope")
    join, source_clause, source_params = source_scope(project_id, module_id, document_id)
    eligible = {row[0] for row in db.execute(text(f"""
        select c.id from chunks c {join} where c.project_id=:project_id
        {source_clause} and c.chunk_role='teaching'
    """), source_params)}
    if any(not set(chunk_ids) <= eligible for chunk_ids in links.values()):
        raise ValueError("Prepared assignments contain out-of-scope chunks")
    # Explicit multi-row VALUES avoids driver executemany issuing one INSERT
    # per link over a remote connection. Keep the caller's atomic transaction.
    batch = []
    def insert_batch(rows):
        check()
        values, parameters = [], {}
        for i, (topic_id, chunk_id) in enumerate(rows):
            values.append(f"(:topic_{i},:chunk_{i})")
            parameters[f"topic_{i}"] = topic_id
            parameters[f"chunk_{i}"] = chunk_id
        db.execute(text("insert into topic_chunks(topic_id,chunk_id) values "
                        + ",".join(values)), parameters)
        check()
    for topic_id, chunk_ids in links.items():
        for chunk_id in sorted(chunk_ids):
            batch.append((topic_id, chunk_id))
            if len(batch) == 400:
                insert_batch(batch)
                batch = []
    if batch:
        insert_batch(batch)
    actual = {(str(row[0]), row[1]) for row in db.execute(text(f"""
        select tc.topic_id,tc.chunk_id from topic_chunks tc
        join topics t on t.id=tc.topic_id where t.project_id=:project_id
        {clause.replace('and module_id', 'and t.module_id').replace('and document_id', 'and t.document_id')}
    """), params)}
    expected = {(topic_id, chunk_id) for topic_id, chunk_ids in links.items() for chunk_id in chunk_ids}
    if actual != expected:
        raise ValueError("Published evidence differs from validated plan")
    check()
