# MEMORY.md

## Architecture Decisions

- Windows builds are allowed to clean their own stale packaged desktop
  processes before PyInstaller runs. The cleanup scope is intentionally narrow:
  only exact packaged executable names under this repository's `build/` or
  `dist/` roots may be terminated, never Ollama or unrelated Python/Streamlit
  processes. Build artifact cleanup must fail with a precise locked-file
  diagnostic rather than asking the user to use Task Manager.
- Financial-health ratios use one canonical deterministic definition module,
  `finance_agent.calculations.financial_health_ratios`. Current ratio,
  cash ratio, total debt ratio, EBITDA margin, net margin, and ROA are
  calculated only from processed source fields and classified against
  configurable internal management thresholds. Those thresholds must be shown
  as analytical management references, not regulatory limits, and presentation
  layers consume the same calculation metadata rather than duplicating bands.
- School/department/organizational-unit budget analysis is derived from the
  existing deterministic `department_summary` output. Presentation adapters
  build grouped actual-vs-budget revenue and expense views plus variance
  rankings from that canonical summary; renderers and Streamlit must not
  recalculate or read raw workbooks for this view.

- Packaged historical chart freshness is runtime-DB-sensitive. Pipeline cache
  identity includes a compact fingerprint of completed prior-period runs, and
  Streamlit report-artifact freshness compares its canonical series against
  both the packaged application-data SQLite database and processed summaries.
  Synthetic history never ships or enters production implicitly; demo history
  is merged only through the explicit, period-scoped, idempotent
  `scripts/import_synthetic_history.py` utility.

- Normal desktop executive analysis is governed by a five-minute runtime SLA:
  deterministic structure preservation and the validated Python investigation
  queue must not trigger preliminary 30B model calls; Python builds one bounded
  `ExecutiveEvidencePackage`, normal AI mode permits one non-thinking structured
  reasoning call plus at most one targeted repair, and inference is terminated
  at its explicit budget rather than entering full regeneration cascades.
  Performance traces contain timings/token metadata but no raw financial
  contents. Model selection uses configurable QUALITY/BALANCED/FAST tiers and
  may select only exact installed Ollama model names; it never downloads or
  invents a model.

- The packaged macOS desktop lifecycle uses a persistent native controller as
  the application process and a distinctly named, session-owned PyInstaller
  helper for Streamlit. Each launch creates a fresh session ID and dynamically
  selects a port; closing the controller terminates only its verified Streamlit
  child and removes replaceable session state. Ollama is always treated as a
  shared external service and remains running on application exit, including
  when the application initially starts it.

- PyInstaller desktop builds must collect the complete `finance_agent` module
  graph because Streamlit executes its bundled UI script dynamically. Frozen
  Streamlit child startup must set `global.developmentMode=false` before using
  a dynamically selected server port. Historical/report refresh callers must
  read processed artifacts from `PipelineConfig.output_directory` in writable
  platform application data, never from the read-only bundle root.
- Windows PyInstaller Streamlit helpers must be invoked with the explicit
  `.exe` sibling path and must not treat `os.getppid()` mismatch as orphaning;
  the windowed bootloader can report a different parent even while the launcher
  PID is alive. Windows helper ownership uses the explicit launcher PID
  liveness check, while Streamlit stdout/stderr are written to a separate child
  log so lifecycle records remain parseable.

- Anomaly/finding records must preserve canonical provenance for every
  observed-vs-reference comparison. Executive-facing output distinguishes
  `institutional_violation`, `statistical_anomaly`, `system_review_rule`,
  `potential_duplicate`, `data_quality_finding`, and
  `informational_observation`; each finding carries observed value, reference
  value/type/origin/source, institutional-reference boolean, reason for
  flagging, supporting evidence, and recommended action. Configured/default
  detector thresholds are analytical system references unless explicit
  workbook/budget/policy provenance proves otherwise, and they must never be
  described as university goals, policies, limits, or violations. Potential
  duplicate findings require transaction-match evidence and must never rely on
  an arbitrary monetary threshold or imply fraud/confirmed wrongdoing.
- Department/category budget-variance findings must reconcile workbook-provided
  variance fields against the deterministic formula `actual_expense -
  budget_expense` and `(actual_expense - budget_expense) / budget_expense`
  before any finding is emitted. Disagreement beyond tolerance becomes a
  `data_quality_finding`; valid review findings carry structured
  `comparison_details` so UI/PDF/HTML lead with budget, actual spend, monetary
  difference, percentage difference, and then any system analytical reference.
- Goal/budget comparison charts are grouped actual-vs-reference visuals, never
  stacked bars. The presentation layer owns Spanish reference labels such as
  "Presupuesto", "Meta", "Límite máximo", and "Meta mínima"; currency and ratio
  metrics are charted separately, and renderers consume the same explicit
  grouped chart rows so actual and reference values are never summed visually.
- Spanish executive surfaces must not render deterministic English templates.
  Titles, finding descriptions, evidence sentences, recommendation/status
  labels, chart labels, tooltips, and hidden-but-visible diagnostic expanders
  pass through the presentation/localization layer before UI/HTML/PDF output.
  Raw source artifacts may retain English internals, but rendered Spanish
  output is guarded by localization-leak tests for phrases such as "spent",
  "against", "budget", "target", "review threshold", and "expense variance".
- The supported user input workflow is one integrated Excel workbook (`.xlsx`
  or `.xls`) per reporting period. The workbook must contain actuals, budgets,
  goals/targets, variances, departments, payroll, collections, cash-flow, and
  other finance tables needed by the pipeline. Separate goals PDFs/DOCX files,
  OCR, and two-uploader report+goals flows are retired from the active product
  path; PDF remains supported only as a generated executive report output.
- Persistent local services are user-owned during development. Codex must not
  start Streamlit, Ollama, dev servers, browser automation, watchers, or
  background workers as normal task completion; it validates with imports,
  Streamlit/AppTest-style tests, artifact rendering checks, and bounded helper
  scripts. Manual launchers live in `scripts/start_streamlit_windows.ps1` and
  `scripts/start_streamlit_macos.sh`; `scripts/check_local_services.py` only
  checks already-running services with a hard timeout.
- Streamlit KPI badges and cards must display deterministic report-model
  comparisons as separate facts: current value, prior-period value, computed
  current-minus-previous change, valid percent change, and percentage-point
  change for ratio metrics. A previous-period value must never be passed to a
  UI delta/arrow slot or rendered as if it were the change.
- Streamlit results use the renderer-agnostic report model as the canonical
  dashboard source. Deterministic sections such as anomalies, revenue/expense
  budget analysis, department results, historical trends, recurring risks,
  evidence, and prior-recommendation follow-up remain visible even when
  validated strategic recommendations are unavailable; PDF/HTML/report-model
  downloads are restored from period-slugged artifacts independently of
  strategic-analysis status.
- Streamlit executive presentation uses neutral, theme-safe surfaces with
  reusable design tokens and semantic accent borders/status chips. Full-card
  red/green/amber backgrounds are avoided for normal dashboard content; solid
  alert treatments are reserved for genuine failures.
- Historical memory retrieval separates internal identifiers from bound data
  values. Canonical metric/artifact identifiers can use strict identifier
  validation, but department/entity/category/subject names from reports or
  memory are normalized Unicode text values used only as SQLite bound
  parameters. Legitimate accents and business punctuation are preserved, while
  null/control characters, empty required values, and excessive length are
  rejected.
- Strategic LLM output is optional enrichment. Validated deterministic analysis
  must always remain reportable: if Ollama strategic synthesis is unavailable,
  rejected, or partially sanitized, the pipeline should preserve validated
  Python-derived KPIs, anomalies, evidence, history, and report artifacts while
  clearly warning that strategic recommendations were not fully validated.
- Successful deterministic financial runs must never produce an empty strategic
  executive section. Step 9 uses a bounded recovery pipeline: one primary
  Ollama generation, targeted/schema or evidence repair where applicable, one
  constrained reduced-context generation, one final tightly constrained
  regeneration when needed, and then a mandatory Python-authored degraded
  synthesis from verified evidence. Final accepted report artifacts must
  contain non-empty executive analysis, priorities, and recommendations, with
  source/recovery metadata shown as a badge rather than an
  unavailable-recommendations empty state.
- Production strategic reasoning is AI-first. Normal runs must pass an Ollama
  readiness contract before strategic analysis: service reachable, configured
  model installed, tiny health prompt succeeds, timeouts valid, and active model
  known. If readiness fails in normal AI mode, the pipeline stops at strategic
  analysis with the Spanish message "El motor de IA no está disponible" rather
  than silently generating a report. Deterministic strategy is now an explicit
  degraded mode, labeled "Modo degradado: análisis determinístico" with the
  note that AI reasoning capabilities were unavailable or exhausted; degraded
  artifacts must not be reused as normal AI cache hits.
- Pipeline progress reporting is an optional orchestration callback, not
  Streamlit-specific business logic. `PipelineProgressEvent` carries Spanish
  labels/details, status, elapsed seconds, and completed/total major-stage
  counters; Streamlit renders those events as a progress bar/checklist while
  CLI and non-UI callers can ignore the callback. Progress percentages reflect
  completed real pipeline stages only and must not invent sub-step/token
  progress during long Ollama calls.
- Streamlit reporting-period selection is temporarily monthly-only. The active
  UI options are "Detectar automáticamente" and "Mensual"; quarterly,
  semester, annual, and custom frequencies are not selectable until
  cross-frequency comparison, goals matching, retrieval, storage, and report
  semantics are explicitly implemented. Existing backend/monthly artifacts and
  period slugs such as `2026_04` through `2026_12` remain unchanged, and monthly
  previous-period comparison continues to map January to the prior December.
- Streamlit uploads use content-based persistent ingestion. Financial reports
  and goals are identified by raw-byte SHA-256, never filename, and accepted
  runs register source-document metadata in SQLite with exact-content
  uniqueness, period-aware version numbers, `supersedes_document_id`, and
  `is_current`. Same content under a renamed file reuses the existing document;
  same period with different content is a revision that requires explicit UI
  confirmation before it can become the current accepted version. Revision
  confirmation belongs to the per-submission `PipelineInputModel`, not
  `PipelineConfig`, because it is a user decision about a specific upload rather
  than a global runtime/model setting. Pipeline cache identity includes
  report/goals hashes, runtime configuration, and an explicit pipeline schema
  version.
- Current-period anomalies and historical recurring risks are separate report
  concepts. Report-model mapping sends the current anomaly artifact only to
  "AnomalÃ­as del perÃ­odo"; compact historical repeated-risk retrievals populate
  "Riesgos histÃ³ricos recurrentes". Renderers must not use a zero current-period
  anomaly count to suppress historical risks, and presentation labels/entities
  should be localized centrally while source artifacts keep canonical names.
- Historical recurring-risk and prior-recommendation sections are executive
  presentations, not raw memory-table dumps. Report mapping carries compact
  historical context into the report model; presentation code derives risk names
  from deterministic anomaly type/metric metadata, merges only identical
  `(risk_type, department)` records, and keeps distinct risks in the same
  department separate. Recommendation follow-up uses deterministic topic
  grouping, stored origin periods, related KPI trends, and fixed Spanish
  operational labels to show progress, evidence, objective, and next action
  without asking Ollama or inventing business facts.
- Executive report sections should answer explicit management questions rather
  than mirror report-model or database structures. Recommendation follow-up
  must explain what prior recommendations are tracked, when they were emitted,
  how status was determined, and what action remains; recurring-risk sections
  must explain what happened, why it is recurrent, its recurrence movement,
  and why management should care using deterministic historical metadata only.
- KPI card comparison values are deterministic report-model data, not renderer
  guesses and never Ollama-filled. Report generation builds a
  `kpi_comparisons` payload from current processed finance summaries, compact
  historical retrievals, previous processed finance summaries when available,
  and budget/cash-flow variance outputs. Renderers display only real comparison
  rows and omit genuinely absent comparisons instead of repeating unavailable
  placeholders.
- Historical trend charts use one canonical report-model series. SQLite memory
  retrieval returns completed prior periods only, de-duplicated to the latest
  accepted run per period and sorted chronologically; report presentation then
  appends the current period's deterministic processed KPI value for display.
  Renderers and Streamlit must consume this shared series rather than querying
  memory or recalculating trends independently.
- Historical trend chart display uses a rolling monthly window: current period
  plus up to the previous five comparable monthly periods, maximum six points.
  Every real accepted month inside the window must be preserved; renderers must
  never reduce the series to first/latest endpoints, interpolate missing
  months, or leak future periods.
- Missing-information claims must be verified against processed deterministic
  evidence before they reach report models or UI/PDF/HTML output. Department
  lookups use alias-aware data-value matching, such as `Arts & Humanities` and
  `Artes y Humanidades`, while preserving canonical source values. If Ollama
  claims a department field is missing but `department_summary` contains the
  field, Python removes the missing item, records source provenance, and falls
  back to deterministic department text rather than publishing contradicted
  prose.
- Modular reasoning uses compact Python-selected fact cards rather than
  placeholders, model-generated evidence IDs, or `supporting_facts`. Python owns
  deterministic facts, values, periods, entities, formatting, evidence
  selection, and validation; Ollama receives only 10-25 relevant current facts
  for Stage 1, 10-25 relevant historical facts for Stage 2, and accepted Stage
  1/2 reasoning plus a small synthesis fact set for Stage 3. Ollama writes
  professional Spanish reasoning only.
- Modular reasoning validation rejects unsupported quantitative claims, periods,
  entities, or causal certainty, but it no longer requires Ollama to reproduce
  internal identifiers or placeholder syntax. Python attaches internal evidence
  references deterministically after accepted Stage 3 synthesis so existing
  report/model/storage contracts remain compatible. Exact display variants of
  supplied facts, such as currency thousands separators or Spanish phrasing of a
  negative percentage as a decrease, are accepted without allowing new
  calculations.
- Phase 14 replaces the primary Step 9 path with a modular multi-stage
  reasoning pipeline. `ReasoningState` is ephemeral per report run, separate
  from SQLite memory, and carries the validated evidence ledger, accepted
  claims, risks, opportunities, open questions, stage outputs, evidence
  references, and cross-stage dependencies. The orchestrator and Step 9 CLI now
  call the modular path: Stage 1 reasons about current financial performance,
  Stage 2 reasons about historical/operational persistence using Stage 1 output,
  and Stage 3 produces strategic synthesis from validated reasoning state rather
  than the full evidence ledger. Existing report/storage compatibility is
  preserved by emitting the same strategic-analysis document shape, with
  `reasoning_state` embedded and separate stage artifacts saved for debugging.
- Phase 14 modular Stage 1/2 reasoning uses the provider-friendly internal
  schema `claims`, `risks`, `opportunities`, and `open_questions`. A deterministic
  schema adapter may unwrap one obvious wrapper object and rename safe aliases
  such as `identified_risks`, `identified_opportunities`, `financial_claims`,
  `questions`, and item-level `confidence_level`, but it must never write prose,
  add evidence IDs, infer conclusions, change numbers, or repair unsupported
  claims. Valid JSON with schema-only errors may receive one schema-only
  Ollama retry; evidence, language, period/entity, number, and causal-claim
  validation remain strict after adaptation.
- Ollama HTTP handling distinguishes connection timeout, service unavailable,
  inference/read timeout, malformed response, stage timeout, and validation
  rejection. The supported defaults are `connect_timeout_seconds=10`,
  `read_timeout_seconds=600`, `stage_timeout_seconds=900`, and
  `keep_alive=15m`; the legacy `timeout_seconds`/`--ollama-timeout` option is
  retained as a backward-compatible read-timeout alias. Benchmarks and explicit
  uncached runs may warm the model, but normal cached runs should not warm
  Ollama unnecessarily.
- Report generation is section-template driven: Python defines section IDs, Spanish titles, objectives, required/optional evidence, chart/table specs, narrative fields, visibility rules, and validation; Ollama supplies the section-specific Spanish analytical prose. Presentation code may format labels/values and build visuals, but must not hardcode report conclusions, generic management commentary, or deterministic narrative paragraphs.
- Executive report presentation uses reusable deterministic dashboard components
  in `finance_agent.reporting.presentation`: KPI card metadata, status badges,
  trend arrows/deltas, chart fact models, and "ConclusiÃ³n ejecutiva" summaries
  are generated from the same processed facts rendered in charts. These
  conclusions may describe visible start/current values for trends, category
  rankings, min/max values, deltas, current markers, and actual-vs-budget
  comparisons, but must not infer causes or change strategic narrative authored
  by Ollama.
- Executive report renderers use adaptive presentation by default: empty or
  low-value tables are replaced with compact Spanish status cards, short
  anomaly/evidence/recommendation lists render as executive cards, larger lists
  render as tables, and KPI cards use natural administrator-facing labels such
  as "VariaciÃ³n respecto al periodo anterior" instead of technical delta labels.
  This is presentation-only; detailed JSON/CSV artifacts remain the audit source.
- Step 9 strategic analysis now validates section-aligned narrative fields (`financial_health_analysis`, `kpi_analysis`, `historical_trend_analysis`, `department_analysis`, `anomaly_analysis`, recommendation follow-up, longitudinal risk, and strategic recommendations) against supplied evidence IDs. Schema placeholders, English prose, unsupported numbers/periods/entities, and copied context objects are rejected. Ollama may get schema-constrained JSON format hints plus bounded Spanish/evidence repair retries, but Python validation remains authoritative and report rendering is blocked when accepted strategy is unavailable.
- User-facing strategic-analysis prose must be generated directly in professional Spanish by Ollama and validated in Step 9. The reporting presentation layer may sanitize internal IDs/paths/tool names and format labels/values, but it must not translate narrative with exact English sentence mappings or word-by-word replacement dictionaries. If an Ollama response is schema-valid but English-dominant, Step 9 may retry once with a bounded Spanish rewrite prompt that preserves numbers, evidence, priorities, confidence, and meaning; if it still fails, the analysis is rejected and executive report rendering remains blocked.
- Executive report generation now has a dedicated presentation adapter (`finance_agent.reporting.presentation`) between the renderer-agnostic report model and HTML/PDF renderers. Internal model fields remain English for pipeline compatibility, but executive renderers consume Spanish labels, Step-9-authored Spanish narrative, compact source filenames, recommendation cards, cleaned historical trend/risk/follow-up records, and presentation validation that blocks raw dict/list strings, local paths, internal retrieval tool names, canonical KPI identifiers, missing recommendations, and ASCII chart bars. Detailed historical/retrieval artifacts stay in JSON/CSV sources, not in executive report bodies.
- Phase 13 historical reasoning integration uses `finance_agent.memory.context_builder` as the single deterministic layer between current pipeline outputs and SQLite memory. It selects relevant compact history, caches duplicate retrievals within one run, derives bounded KPI trend/goal-progress/recommendation-effectiveness summaries, and may read only processed artifact references such as normalized `Anomalies_Embedded` CSVs; it must never load full prior reports or send raw history to Ollama. Planner and strategic-analysis artifacts include historical-context summaries, and report models may include optional longitudinal sections when history exists.
- Generic pipeline input is represented as one financial report path, one goals document path, detected period metadata, optional period override, and report language defaulting to `es`. Period detection is deterministic and conservative; low-confidence or conflicting periods require an override before execution.
- Streamlit UI v1 is intentionally thin: it saves uploaded files, builds the generic pipeline input/config, calls `run_pipeline_for_report()`, and displays orchestrator/report artifacts. It must not duplicate calculations, retrieval, strategic analysis, reporting, or orchestration logic.
- Streamlit UI user-facing screens should be administrator-first Spanish, not
  architecture-first. The UI presents a guided upload-run-download workflow,
  validates required files before enabling execution, preserves completed
  results across reruns, keeps expert Ollama/cache/memory controls under
  collapsed advanced settings, and renders previews from the same report
  presentation adapter used by HTML/PDF outputs.
- Pipeline runtime optimization uses two safe shortcuts: structure fallback is skipped only when deterministic table/column confidence has no review items, and generic pipeline cache reuses artifacts only when the cache key matches input file hashes/settings and strategic-analysis/report-quality validation passes. Cache hits should be surfaced in summaries/UI as skipped reuse, not fresh execution.
- Ollama model routing defaults to one supported model, `qwen3:30b-a3b`, for all LLM stages. Stage-specific model overrides remain experimental/backward-compatible only; the benchmark on this machine showed the mixed `qwen3:latest` structure/planner setup was slower, so it is not the recommended default. Cache keys include the effective model routing.
- Final runtime optimization preserves `qwen3:30b-a3b` reasoning but reduces prompt noise: the Ollama planner receives only deterministically ranked Critical/High anomalies, capped by `max_planner_anomalies` default 5, while compact/deduplicated context and per-stage Ollama telemetry are recorded in pipeline summaries and profiling output.
- Phase 11A historical storage uses local SQLite via Python `sqlite3` at `data/memory/finance_memory.db`. Only successful, accepted-strategy, report-quality-valid runs are stored; large artifacts remain on disk with checksummed references, while compact KPIs/anomalies/recommendations/goals/memory facts are transactionally upserted by an idempotency key from report hash, goals hash, period, and configuration.
- Phase 11B historical retrieval tools are read-only Python functions over the SQLite memory repository. They validate periods, limits, metric/filter names, and detail levels; return structured summaries or explicit unavailable results; preserve full-history access through artifact references; and are registered alongside existing retrieval tools without replacing current-period aliases such as `get_department_history`.
- Final pre-UI generalization adds an object-based generic pipeline runner. `run_pipeline_for_report` now executes arbitrary report/goals inputs through orchestrator-owned state, writes period-slugged debug artifacts after each successful stage, and keeps the script-backed synthetic monthly/annual pipeline as backward-compatible convenience mode.
- Step 10B report rendering consumes only Step 10A report model JSON and produces Spanish-facing HTML/PDF presentation artifacts. Renderers stay separate from business logic, do not recalculate finance values, preserve processed-output source references, and use simple static tables/charts suitable for future renderer upgrades.
- Final report rendering must validate that Step 9 strategic analysis was accepted and recommendations are present. If strategy is unavailable, the renderer CLI warns and stops unless explicitly run with the draft-only `--allow-missing-strategy` flag.
- Report output cleanup keeps canonical active reports in `outputs/report/`: generic period-slugged reports such as `2026_06` plus current annual reports when strategy-backed. Legacy duplicate report triplets such as `june_2026` should be moved to `outputs/report/archive/`. Report quality validation blocks missing strategy, missing recommendations, placeholder text, and stale rendered artifacts.
- Generic period retrieval must preserve period-slugged artifact provenance, such as `finance_summary_2026_06.json`, instead of falling back to legacy aliases. Strategic analysis may conservatively remove only objective missing-information claims contradicted by processed anomaly, cash-flow, or payroll evidence; it must not rewrite recommendations or financial reasoning.
- Step 10A reporting creates renderer-agnostic Report Model JSON only. It assembles existing processed finance, KPI, anomaly, evidence, and strategic-analysis outputs into stable sections with source references; it must not duplicate calculations or generate PDF/HTML/UI/email/PowerPoint artifacts.
- The pipeline orchestrator is a thin coordination layer over existing CLI stage scripts. It preserves current output paths, records structured stage results and expected outputs, stops after critical-stage failures, and treats Ollama unavailability as a fail-safe warning when the underlying stage exits successfully.
- Step 8 retrieval should treat normalized tables as complementary evidence sources: department history aggregates Department_Summary, Payroll, Expenses, Vendor_Payments, and Budget_vs_Actual; payroll retrieval exposes department, period/month, headcount, salary, benefits, overtime, payroll amount, budget, and variance; placeholder filters such as `flagged_vendor` or `unknown` must resolve to processed evidence when possible rather than creating false unavailable evidence.
- The `finance_agent` package is organized by responsibility into subpackages: `ingestion`, `understanding`, `calculations`, `anomalies`, `agent`, `retrieval`, `analysis`, `llm`, and `common`. Root `finance_agent.__init__` keeps compatibility exports, while direct internal imports should use the responsibility-based subpackage paths.
- Step 9 is the first strategic reasoning stage: Ollama receives only compact summaries of processed finance outputs, anomaly/risk outputs, and Step 8 evidence packages. Python validates strict JSON analysis before saving; invalid or unavailable model output is rejected rather than repaired, and the LLM must not recalculate or modify financial data.
- Step 8 retrieval executes validated Step 7 queues sequentially through a registry of generic retrieval interfaces. The initial implementation reads only processed JSON/CSV outputs and enriched intermediate metadata, packages unavailable evidence explicitly, continues after per-call failures, and performs no financial reasoning, recommendations, database access, or LLM calls.
- Step 7 makes Ollama the primary investigation planner but treats every response as untrusted: Python enforces exact JSON shape, source identifiers, tool allowlists, typed/ranged arguments, and an 8-step cleaned-plan limit. Equivalent tool/argument calls are safely merged with the highest priority preserved; descriptive text may be trimmed/capped and is audited, while duplicate-ID conflicts or semantic/tool validation failures still select the full deterministic Step 6 baseline.
- Step 7 exposes retrieval interface schemas only, never implementations. Validated or fallback plans are converted to pending execution queues with `tools_executed: false`; this stage performs no retrieval or action.
- Step 6 investigation planning is deterministic and consumes processed calculation, anomaly, trend, risk-summary, and enriched-model outputs only; it never reads raw reports, calls an LLM, retrieves history, or executes evidence tools.
- Investigation tasks retain their source anomaly/data-quality identifier, a score with readable priority factors, and serialized future evidence requests. Repetition is measured from distinct annual anomaly/trend periods, while unresolved Step 5 tables become scope-specific human-review tasks.
- Step 5 Ollama structure fallback is optional and table-scoped: only low-confidence table/column metadata plus at most five sample rows is sent, all responses are validated against Python allowlists, and high-confidence deterministic mappings remain locked.
- Step 5 writes a separate `financial_document_model_enriched.json`; unavailable Ollama, invalid JSON, unresolved `Unknown` types, and conflicts with strong deterministic table types preserve Python results and require human review.
- Step 3 period handling separates source-workbook provenance from row-level `PeriodScope` filtering; supported scopes are monthly, annual, and inclusive custom date ranges.
- Annual finance outputs include a 12-row monthly trend derived only from normalized annual tables, and monthly trend totals must reconcile to annual headline totals.
- Step 3 finance calculations consume only `financial_document_model.json` and its referenced normalized CSVs; calculation code must never read raw Excel/PDF inputs.
- Calculation runs are explicitly scoped by source-workbook provenance so monthly and annual tables in one intermediate model are never summed together accidentally.
- Missing calculation inputs produce `None`/unavailable KPI values plus clear warnings instead of aborting the calculation run.
- Net operating result is actual revenue minus operating expenses from the normalized Expenses table; scholarships and capital cash outflows remain separate summaries.
- Step 2 is split into raw ingestion evidence, document understanding, normalization, classification, feature extraction, and intermediate-model serialization.
- Workbook understanding must use raw cell geometry and merged ranges; it must not rely on a fixed pandas header row.
- Step 3 will consume normalized table CSVs and the intermediate model manifest, without needing original workbook layout knowledge.
- Unknown tables are preserved rather than discarded. Low-confidence tables and columns are explicitly marked for possible future Ollama interpretation, but Step 2 never calls an LLM.
- Step 1 uses a standalone `finance_agent` Python package for deterministic document ingestion, workbook inspection, and starter schema helpers.
- Excel ingestion preserves original sheet names and returns DataFrames plus workbook metadata; JSON inspection is a separate serialization step.
- PDF ingestion returns raw text and basic metadata only. Advanced goal extraction is deferred.

- Target architecture: integrated Excel workbook â†’ ingestion â†’ schema normalization â†’ deterministic calculations/anomaly detection â†’ agent orchestration â†’ history/memory retrieval â†’ Ollama analysis â†’ PDF/Excel outputs â†’ Streamlit interface.
- Python is the source of truth for calculations.
- LLM must not perform financial math.
- Ollama is the default local LLM.
- Use `qwen3:30b-a3b` as the initial Ollama model.
- LLM may be used for structure interpretation when deterministic parsing fails.
- LLM may be used for investigation planning, explanations, strategy, and report writing.

## Agentic Design

- The agent should investigate issues, not only summarize.
- The agent should call tools based on detected anomalies.
- The agent should use memory/history to detect repeated problems.
- The agent should request deeper records only when justified.

## Historical Data Strategy

- Store full reports and processed data.
- Store compact memory of previous analysis outcomes.
- Send summaries to the LLM by default.
- Allow retrieval of detailed/full historical data through tools.

## Coding Preferences

- Executive-facing recommendation presentation is canonicalized in
  `finance_agent.reporting.presentation`: Streamlit, HTML, and PDF consume
  separate title, rationale, operational consideration, investigation,
  expected-impact, owner, priority, and status fields. Legacy inline Spanish
  labels may be split only at presentation time; direct structured values take
  precedence, empty/label-only values are omitted, exact repeated sentences
  are removed, and AI provenance plus underlying reasoning remain unchanged.

- Step 9 strategic analysis now uses an evidence ledger as the only approved
  fact surface for Ollama narrative generation. The ledger records stable
  `evidence_id` values, exact display/raw values, approved periods, approved
  entities, KPI trends, recurring anomalies, prior recommendations, goal
  progress, current findings, and source references. Python validates generated
  Spanish prose against that ledger and blocks report rendering when numbers,
  periods, entities, evidence IDs, or causal claims are unsupported. The
  Ollama-facing prompt uses a compact ledger view, while Python retains the full
  ledger for post-generation support validation.
- Phase 12B historical population uses `scripts/populate_synthetic_history.py` to run generated Phase 12A monthly report/goals pairs chronologically through the generic pipeline and store accepted runs in an isolated SQLite database, default `data/memory/recovery_2026_memory.db`. The completed `recovery_2026` population stores all 12 monthly periods with no duplicate period rows and validation reports under `outputs/history_population/`. Idempotency is defined as reusing/updating existing `pipeline_runs` for the same report/goals/period/config without creating duplicate periods; LLM-derived child facts may be rebuilt when live strategic wording varies. The script can resume missing periods without touching production `data/memory/finance_memory.db`.
- Phase 12A synthetic history generation lives under `finance_agent.synthetic_history` and produces deterministic multi-period university fixtures. The default dataset is `data/synthetic_history/recovery_2026`, with one row-5-header professional workbook and one Spanish goals PDF per month plus a scenario manifest. The generator must protect existing outputs unless `overwrite=True`/`--overwrite` is explicitly supplied, must not call Ollama, and must keep scenario expectations in `scenario_manifest.json` for future memory/retrieval tests.
- Budget variance percentages use aggregate `(actual - budget) / budget`; do not average row-level variance percentages.
- Use the project-local `.venv` for Python execution. Keep `requirements.txt` limited to pinned direct dependencies; do not list transitive packages.
- Keep full row-level normalized data in one CSV per detected table; keep model JSON focused on provenance, structure, confidence, feature metadata, and small samples.
- Prefer conservative deterministic confidence: recognized structures proceed automatically, while ambiguous classifications remain `Unknown`.
- Schema alias mapping remains deterministic and exact-match-first; LLM interpretation is deferred until deterministic mapping is insufficient.

- Clear docstrings for every function.
- Many inline comments.
- Explain how technical logic works.
- Keep code modular and readable.
- Avoid unnecessary frameworks early.

## Build Order

- Processing engine first.
- Then calculations and anomaly detection.
- Then memory/history tools.
- Then Ollama.
- Then agent orchestrator.
- Then outputs.
- Then interface.

## Future Considerations

- Streamlit interface.
- PDF executive reports.
- Excel processed summaries.
- Optional cloud LLMs later.
- Optional MCP only if tools need to be reused by independent agents.

-
