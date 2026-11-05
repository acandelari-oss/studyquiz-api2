# Original-source retention for targeted coverage recovery

Status: proposal only. No migration, bucket, upload, backfill or production changes.

## Confirmed boundary

`main.py:ingest_stream` decodes `IngestDocument.file_bytes` and calls
`acquire_document`. Background processing and retry read persisted chunks. No
original-file storage reference is passed through that path. The local Downloads
PDF used for the audit is not an available source for a deployed worker.

Stored-text recovery can repair unpublished candidates and propose missing topics.
It cannot reconstruct diagram geometry or unreadable handwriting from missing text.
A source link also cannot establish factual correctness of a generated description.

## Minimum durable contract

Use a private object-storage bucket (Supabase Storage is a candidate; configuration
and credentials have not been verified). Retain original bytes for every supported
format, not PDFs only. Do not put large binary data in PostgreSQL.

Propose one nullable `documents.source_file` JSONB field, subject to migration review:

- `version`: metadata contract version, initially 1.
- `object_key`: immutable random document-qualified object key; never a public URL.
- `sha256`: verify bytes before extraction and bind recovery to the original source.
- `byte_length`: validate complete upload/download and enforce existing resource limits.
- `media_type`: validated format for routing through existing extraction adapters.

Reuse document ID, project/module ownership, original title, taxonomy run ID, source
snapshot checks and existing hierarchy revision where available. Do not overload
`chapter_structure` with an unrelated storage lifecycle. Do not assume TEXT/UUID
joins are interchangeable. A new JSONB field does not change any identifier type.

## Lifecycle

1. After authentication/ownership checks, retain immutable bytes before acknowledging
   recoverable ingestion. Upload with overwrite disabled. Validate length and hash.
2. Commit the object reference with its document record. If the transaction fails,
   delete the unreferenced object; log cleanup failures for deterministic orphan
   cleanup. Object storage and PostgreSQL are not a shared atomic transaction.
3. Background recovery checks document/run ownership, downloads only the intended
   source to a bounded temporary file, verifies its hash, and uses existing format
   extraction. PDFs can render selected physical pages. DOCX/PPTX use native blocks
   or slides; do not pretend their positions are PDF pages.
4. Re-extract only the unresolved source units. Keep recovered results separate from
   persisted chunk text until verified against the original source and provenance.
   Do not attach a new quote to an old chunk whose text does not contain that quote.
5. Any necessary chunk/provenance replacement requires an explicit revision and
   source-snapshot update under existing document locks, followed by scoped evidence
   validation. Preserve sibling documents, existing topic IDs and learning records.
   Do not publish recovered material against a stale snapshot.
6. Deletion makes the document unavailable using existing ownership/Study Mode checks,
   then deletes its object idempotently. Cleanup failure must not resurrect a deleted
   document or leave it available to another user. Never log signed URLs or keys.

Retention follows document lifetime, with explicit deletion. No automatic transfer
or backfill of existing files. Legacy documents remain usable; original-source
recovery is unavailable until the original is explicitly supplied and verified.

## Existing components to reuse / files affected when approved

- `main.py`: upload, background recovery and document deletion integration.
- Existing `document_acquisition` / `document_extractors.py`: format extraction.
- Existing PDF source alignment and chapter provenance: verify source coordinates.
- `taxonomy_source_review.py`: targeted repair outcomes and unresolved diagnostics.
- A reviewed migration adding only the nullable source-file reference field.
- Focused tests with mocked storage: isolation, missing objects, hash mismatches,
  interrupted uploads, cleanup failures, stale runs and publication rollback.

No new heading engine, study engine, category algorithm or full-document AI pass.
Storage retention is not itself OCR recovery: format-specific extraction must still
be validated. Models must not infer missing diagram relationships or column order.

## Remaining coverage work

- Source-span accounting must distinguish mixed syllabus/body chunks. Do not mark
  a whole chunk non-teaching because part is an index.
- Short overlap tails need verified positional containment, not weak substring or
  page-neighbor matching. Existing exact substantial-overlap logic remains intact.
- The current bounded repair call is not a factual correctness proof. Original-page
  comparison is needed when text has lost table or diagram relationships.
- No live recovery is authorized by this document. Deployment/storage/migration and
  any paid recovery test are separate actions.
