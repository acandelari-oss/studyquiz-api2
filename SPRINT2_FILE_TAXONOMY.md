# Sprint 2 — File-level taxonomy integration

Implemented locally. No paid API calls, real ingestion, production database changes or deployment.

## Behavior and API

- Single-file modules use the existing category/topic view with a filename/status row. Multi-file modules have separately ordered document sections and independent editors. Legacy NULL-document topics remain in a separate labelled group.
- GET topics adds document name/order/status and a document-qualified `category_key`; GET documents adds document order. Module-wide topic retrieval remains intact.
- Category rename requests now include document ID. Topic reassignment and merging retain backend document-scope guards. Identical category names remain independent.
- DELETE `/projects/{project_id}/documents/{identifier}` resolves document ID first; unique old filenames remain supported. Ambiguous filenames are rejected. It locks the scope and atomically deletes only the document's topics, links and chunks. Legacy/shared source dependencies block deletion.
- POST `/projects/{project_id}/documents/{document_id}/taxonomy/retry` explicitly retries failed or queued/building work. Ready and legacy/untracked documents are rejected. Authorization and Study Mode guards apply. Acquired chunks are reused; ingestion is not rerun.
- Recovery assigns a new attempt token; superseded workers cannot publish or overwrite the replacement attempt's status. Discovery/consolidation algorithms are unchanged.
- File states are polled and aggregated; pending/processing takes precedence over terminal failure. Successful file topics stay visible if a sibling fails. Study Mode approval remains module-scoped and requires the existing readiness safeguards.
- Planner and priorities use document-qualified category keys. Existing plan records are untouched. Title-only historical analytics cannot safely identify repeated titles across different categories, so ambiguous events are not attributed to an arbitrary file. Unambiguous old priorities resolve locally; ambiguous priorities show a reselection notice.
- English and Italian labels cover statuses, recovery, deletion, legacy groups and priority review.

## Database prerequisite

`migrations/20261021_document_taxonomy_attempt.sql` adds nullable UUID `documents.taxonomy_run_id`. It is **not applied**. Apply before running this backend version. This is additional to the Sprint 1 migration already applied by the user. No backfill, data reassignment or taxonomy regeneration is included.

## Verification

- 132 backend offline tests passed: document isolation/publication, recovery fencing, deletion rollback, ownership and Study Mode, discovery deadlines, duplicate consolidation, category organization, Planner context/engine/activity allocation/preferences.
- 29 frontend tests passed: single/multi-file rendering, repeated names, mixed outcomes, legacy groups, localized recovery/deletion controls, module topic IDs, priority compatibility, category review and upload state behavior.
- Focused strict TypeScript validation passed for new document controls, polling hook and taxonomy/category utilities.
- Full TypeScript validation executed: 178 existing diagnostic occurrences remain (initial check: 182); no new diagnostic signatures. Existing unrelated missing modules and typing errors prevent a clean repository-wide result.
- Changed Python modules parse successfully.

## Remaining limitations

- PostgreSQL row-lock/concurrent-worker behavior has not been exercised against a live database; offline tests use SQLite and mocked services.
- Interrupted jobs require explicit recovery; the UI does not infer worker death. Superseded workers are fenced from publication, not forcibly terminated.
- Deleting a file needed by legacy mixed taxonomy is deliberately blocked rather than deleting protected evidence.
- Existing Study Mode approval remains module-wide; successful file taxonomy is visible during sibling failure, but a failed module must be recovered or its failed file safely removed before approval.
- No chapter-based organization changes.

## Modified files

Backend:
- `/Users/MOKA/DOUNO/studyquiz-api2/main.py`
- `/Users/MOKA/DOUNO/studyquiz-api2/taxonomy_document_state.py`
- `/Users/MOKA/DOUNO/studyquiz-api2/category_identity.py`
- `/Users/MOKA/DOUNO/studyquiz-api2/planner/context_builder.py`
- `/Users/MOKA/DOUNO/studyquiz-api2/migrations/20261021_document_taxonomy_attempt.sql`
- `/Users/MOKA/DOUNO/studyquiz-api2/tests/test_document_taxonomy_isolation.py`
- `/Users/MOKA/DOUNO/studyquiz-api2/tests/test_planner_context_builder.py`

Frontend:
- `/Users/MOKA/DOUNO/quiz-ui/components/views/ModuleTopicsView.tsx`
- `/Users/MOKA/DOUNO/quiz-ui/components/views/TopicsView.tsx`
- `/Users/MOKA/DOUNO/quiz-ui/components/DocumentTaxonomyControls.tsx`
- `/Users/MOKA/DOUNO/quiz-ui/components/useTaxonomyDocuments.ts`
- `/Users/MOKA/DOUNO/quiz-ui/components/Workspace.tsx`
- `/Users/MOKA/DOUNO/quiz-ui/components/ui/CategoryLabel.tsx`
- `/Users/MOKA/DOUNO/quiz-ui/components/views/PlannerView.tsx`
- `/Users/MOKA/DOUNO/quiz-ui/components/views/planner/plannerWeekAdapter.ts`
- `/Users/MOKA/DOUNO/quiz-ui/utils/topicModules.ts`
- `/Users/MOKA/DOUNO/quiz-ui/utils/topics.ts`
- `/Users/MOKA/DOUNO/quiz-ui/utils/categoryIdentity.ts`
- `/Users/MOKA/DOUNO/quiz-ui/public/locales/en/common.json`
- `/Users/MOKA/DOUNO/quiz-ui/public/locales/it/common.json`
- `/Users/MOKA/DOUNO/quiz-ui/tests/topicModules.test.cjs`
- `/Users/MOKA/DOUNO/quiz-ui/tests/documentTaxonomy.test.cjs`

This report: `/Users/MOKA/DOUNO/studyquiz-api2/SPRINT2_FILE_TAXONOMY.md`.
