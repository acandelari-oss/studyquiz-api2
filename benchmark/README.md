# Local AI cost benchmark

Temporary, opt-in diagnostics. No database schema, quotas, authentication, prompts, model choices or retry limits are changed. No student cost UI. Raw provider responses are never stored by this tool.

## Enable for local use

Use the same test ID and **absolute** output directory in both terminals. Keep the default directory name `benchmark-output`; generated directories also receive a `.gitignore` so outputs are not accidentally committed.

Backend terminal:

```bash
cd /Users/MOKA/DOUNO/studyquiz-api2
export DOUNO_AI_COST_TEST=1
export DOUNO_AI_COST_TEST_ID=quiz-medium-balanced-15-r1
export DOUNO_AI_COST_TEST_OUTPUT_DIR=/Users/MOKA/DOUNO/studyquiz-api2/benchmark-output
venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend terminal:

```bash
cd /Users/MOKA/DOUNO/quiz-ui
export DOUNO_AI_COST_TEST=1
export DOUNO_AI_COST_TEST_ID=quiz-medium-balanced-15-r1
export DOUNO_AI_COST_TEST_OUTPUT_DIR=/Users/MOKA/DOUNO/studyquiz-api2/benchmark-output
npm run dev
```

Use existing local credentials/configuration and confirm the test database target before a real UI test: normal feature operations can write application data. These commands enable diagnostics; they do not grant authorization for paid scenarios. Do not set these variables on Render/Vercel or use `NEXT_PUBLIC_` names.

Run one controlled scenario at a time, in one browser tab. Start with a fresh test ID; restart both processes when changing it. Include a repetition suffix. Avoid unrelated generation while recording. The ID must contain only letters, digits, hyphens or underscores (maximum 100 characters); do not put names, email addresses or filenames in it.

Finish the operation before reporting. For uploads, wait for taxonomy completion, not merely “upload complete.” Scheduled but unfinished background work is flagged. For Study Sessions, finish the intended sequence, then export with `--feature study_session`. The tool groups multiple requests by test ID and displays components; it does not add a persistent session engine or certify the student's full-session completion.

## Report without inspecting JSONL

```bash
cd /Users/MOKA/DOUNO/studyquiz-api2
venv/bin/python scripts/report_ai_cost_test.py \
  benchmark-output/quiz-medium-balanced-15-r1 \
  --out benchmark-output/reports/quiz-medium-balanced-15-r1
```

Outputs: `summary.md`, `summary.csv`, `requests.csv`, and `pricing-used.json`. The pricing snapshot makes exported calculations reproducible. Costs use Python Decimal and the centralized `benchmark/ai_prices.json`; no calculation happens inside the generation path. Reports include the version and distinguish known calculated subtotals from incomplete totals. They are not invoices.

For separate backend/frontend output locations, pass both directories as positional arguments. Repeated event IDs are deduplicated. Don't place unrelated real logs in the input directory. Regenerate the report after background work finishes. To label a complete controlled multi-request test, append `--feature study_session` (or another non-sensitive scenario label); this does not override recorded outcomes.

## Disable

Stop the local processes, unset `DOUNO_AI_COST_TEST` in both terminals (or set it to `0`), then restart normally. No source edit is needed. Disabled mode creates no output files and does not add `stream_options`. Keep diagnostic variables out of committed configuration.

## Safe mocked example (no AI or database access)

```bash
cd /Users/MOKA/DOUNO/studyquiz-api2
PYTHONDONTWRITEBYTECODE=1 venv/bin/python benchmark/mock_example.py --out benchmark-output/mock-example
venv/bin/python scripts/report_ai_cost_test.py benchmark-output/mock-example \
  --out benchmark-output/mock-example-report
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest discover -s tests -p test_ai_cost_test.py
```

From `/Users/MOKA/DOUNO/quiz-ui`:

```bash
node --test tests/aiCostTest.test.cjs
```

The example has three **synthetic** scenarios: quiz with additional calls, limited taxonomy input, and Oral Practice with unknown speech cost. Synthetic numbers are not real DOUNO cost measurements. Use a fresh output directory when regenerating examples to avoid appending a second run to the same test IDs.

## Measurements and limits

- Each row represents an observed SDK/fetch invocation, **not every hidden HTTP retry**. No custom transport observer was added. Application refills/repairs are tagged; SDK internal retries and their billing remain unobserved. Observed cost coverage never establishes an exact invoice total.
- Capture happens before content parsing/persistence. Provider success can coexist with application failure. `application_outcome=unknown` on a request is deliberate when no request-specific validator result is available; `operation_outcome` records the enclosing observed outcome without pretending every candidate was accepted.
- Streaming uses terminal usage only while enabled, without adding browser fields or buffering text for measurement. Interrupted streams and current MP3 speech can lack usage and remain unknown. No request is retried for telemetry.
- Missing cached-token detail, unknown model snapshots, tiers or categories leave costs unknown. Do not silently reuse a different model's rate. Update the small pricing configuration only after verifying official sources. Text reasoning tokens are already part of output and aren't billed twice by the calculator.
- Taxonomy scope rows use the existing eligible teaching count and actual selection results. `selected_rows` counts rows returned by the unchanged 400-row SQL selection; `teaching_selected` counts nonempty rows retained in the taxonomy input. The selection-limit exclusion is `available - selected_rows`, not `available - nonempty_rows`. Coverage is an **input count**, not proof of semantic topic coverage or final learning quality.
- Exactly 400 selected rows means the cap was reached; only an available count above selected rows demonstrates exclusion. Separate module/document/project processing passes remain separate; restoration passes are not merged into one misleading percentage.
- PDF source pages come from the extractor; extracted pages count its page blocks after successful extraction. Blank/scanned page blocks can exist; this is not a claim that OCR recovered usable content from every page. DOCX/PPTX page equivalents remain unavailable. No inferred pages are made up.
- Completed document chunk counts are emitted after persistence. Partial failed documents may have unknown counts. Per-100-page cost is deliberately withheld for incomplete or limited taxonomy, mixed non-PDF inputs, unrelated feature calls or unknown-cost calls. Multi-document module taxonomy is shared: the tool reports batch cost and per-document counters, **not fabricated individual document cost allocations**. To measure total cost per document, test one document in a fresh module.
- Reuploads into an existing module can process old material again; use a fresh isolated project/module for new-page processing cost. There is no automatic fixture reset or application-data cleanup.
- File writes are best effort. Each process has its own file. Write failures produce a fixed warning; subsequent successful events carry a failure count. If every write fails or a process dies before recording, a report cannot recover lost usage. Never interpret an absent report as zero spend. No durable telemetry guarantee is implied.
- Existing application console logs are unchanged and may contain source/output debugging information. **Do not export console logs** as benchmark artifacts; only this tool's metadata JSONL/CSV/Markdown outputs are intended for reporting.
- Local prototypes/benchmark interpreters and unused `rag.py` are not active application paths and are not instrumented. Deterministic map construction and SQL summaries make no AI call and receive no artificial AI charge.
- Prices were checked against official model pages on 15 October 2026; source URLs are in the config. Hosting, retained storage, OCR compute, fees, taxes and future study activity are outside measured API usage.

## Controlled real-test plan (requires separate authorization)

Pilot PDF ingestion once each at 30/100/200/500 pages; add two varied 100-page files before repeating expensive large tests. Try representative DOCX/PPTX once without page equivalents. For quizzes use three medium/balanced and three hard/balanced 15-question repetitions, then one initial sample per remaining difficulty/style combination. Use three repetitions for 10/20 cards, Ask and Oral Practice; two for a short Study Session and initial Planner. Include separate Professor follow-ups and relationship explanations. Expand only when variance warrants it. Small samples support ranges/medians, not reliable p95 claims.

Compare equivalent starting states. Existing quizzes and taxonomy may alter repeat behavior. Extra-page/subscription pricing remains a separate analysis: Free 30 lifetime pages, Pro 200/month and Pro Plus 500/month are assumptions, not rules implemented here.

## Implementation file inventory

Backend modified: `.gitignore`, `main.py`, `chat_stream.py`, `planner/professor_voice.py`, `relationship_explanation_service.py`.

Backend added: `ai_cost_test.py`, `scripts/report_ai_cost_test.py`, `benchmark/ai_prices.json`, `benchmark/README.md`, `benchmark/mock_example.py`, `tests/test_ai_cost_test.py`.

Frontend modified: `.gitignore`, `pages/api/educational-units.ts`, `pages/api/educational-unit-title.ts`.

Frontend added: `server/aiCostTest.ts`, `tests/aiCostTest.test.cjs`.

Other pre-existing frontend edits are outside this task and were preserved.

## Validation performed — 15 October 2026

- 22 new Python benchmark tests passed, including CLI exports, exact-limit versus excluded-input coverage, interrupted generators/streams, unknown pricing, cached/reasoning arithmetic, concurrency and fail-open logging.
- 26 selected Node tests passed: 5 new benchmark/route tests plus existing chat, audio and background-upload checks.
- 136 existing Python regression tests were exercised with diagnostics both disabled and enabled, dummy credentials, mocked provider construction and outbound networking blocked: 121 passed; the same 15 Planner API tests errored because their SQLite fixture lacks `study_modules`. Running the unmodified HEAD `main.py` reproduced those same 15 errors. No unrelated schema/test fixes were made.
- TypeScript checking reports 182 pre-existing diagnostics. A compiler-host comparison using the original HEAD versions of the two modified routes produced the same diagnostics, with no additional errors from this change.
- Structural AST comparison of the four modified backend behavior files, with diagnostic wrappers/annotations removed, matches their original HEAD AST exactly. This checks that original prompts, provider arguments and educational logic remain unchanged; it does not replace runtime tests.
- No paid benchmark scenario, deployment, production configuration change or production database change was performed. Generated examples use synthetic in-memory provider responses only.
