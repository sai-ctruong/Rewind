# Documentation Map

Every document in this repository, classified. Old documents are kept — deleting the
record of how a system got here destroys the ability to audit it — but a reader must be
able to tell in one line whether a file describes the system that exists today.

| Status | Meaning |
|---|---|
| **CURRENT** | Describes the system as it is now. Safe to quote. |
| **HISTORICAL** | Accurate record of a past state or a past decision. Do not quote as current capability. |
| **SUPERSEDED** | Describes behaviour that has since been replaced. Quote only with the replacement named. |

Last reviewed: R0 on branch `research/aic2026-metric-budget`.

## Entry points — CURRENT

| Document | Contents |
|---|---|
| [README.md](../README.md) | What the system is, how to run it, what it refuses to claim. |
| [TASKS.md](../TASKS.md) | Current work above the rule; historical notes below it. |
| [docs/KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | What the system cannot do or has not proven. |
| [docs/COMPETITION_RELEASE_CHECKLIST.md](COMPETITION_RELEASE_CHECKLIST.md) | Pre-session, per-query and pre-submission procedure. |
| [docs/CURRENT_IMPLEMENTATION_AUDIT.md](CURRENT_IMPLEMENTATION_AUDIT.md) | Item-by-item audit; the final section carries FIXED/PARTIAL/OPEN/BLOCKED_EXTERNAL. |
| [docs/RESEARCH_R0_R1_METRIC_AWARE_BUDGET.md](RESEARCH_R0_R1_METRIC_AWARE_BUDGET.md) | The research programme: problem, method, evaluation protocol, no-GT status. |
| [docs/RESEARCH_DATASET_AND_EVALUATION_PROTOCOL.md](RESEARCH_DATASET_AND_EVALUATION_PROTOCOL.md) | What is searchable, what may be measured, and the private-GT rules. |
| [evaluation/private_dev/README.md](../evaluation/private_dev/README.md) | Private development ground truth: schema, annotation protocol, split discipline. |
| [PHASE_REPORT.md](../PHASE_REPORT.md) | Chronological history. Each entry is a record of its own phase. |

## Phase reports — HISTORICAL (accurate for their phase)

`docs/PHASE_1_RUNTIME_CONFIG.md`, `PHASE_2_CACHE_MANIFEST.md`,
`PHASE_3_DATASET_VALIDATION.md`, `PHASE_3_1_DATASET_SCOPE_AND_MAPPING.md`,
`PHASE_3_2_VIDEO_BACKED_DEVELOPMENT.md`, `PHASE_4_DYNAMIC_DATA_ROOT.md`,
`PHASE_5_LOCAL_REFINEMENT.md`, `PHASE_6_GROUNDED_QA.md`,
`PHASE_7_TRAKE_STRUCTURAL_CORRECTNESS.md`, `PHASE_8_TRAKE_KBEST_AND_REFINEMENT.md`,
`PHASE_9_MULTI_CHANNEL_RETRIEVAL.md`, `PHASE_10_SUBMISSION_AND_UI_SAFETY.md`,
`PHASE_11_FINAL_INTEGRATION.md`.

Each describes the state at the end of its phase. Where R0 later changed something, the
change is recorded in the research document rather than by rewriting history. Known
R0 divergences from the phase reports:

| Phase text | Superseded by |
|---|---|
| Phase 9/11: "OCR/ASR/caption are left ENABLED so their absence is reported" | R0 disables them in the competition config; absence is still reported, as INFO |
| Phase 11: readiness has three statuses per check | R0 adds `INFO`, which never affects the verdict |
| Phase 3.2/11: `existing_videos` is the development scope | R0 adds `retrieval_ready` and names `existing_videos` a *visual* scope |
| Phase 5/8: `RankingConfig.diversity_lambda`, `recall_tail_size`, `AlignmentConfig.alignments_per_video`, `sequence_overlap_threshold` appear in config listings | R0 removed all four; they were never read |

## Reference — CURRENT

| Document | Contents |
|---|---|
| [docs/AIC2026_COMPETITION.md](AIC2026_COMPETITION.md) | Task definitions and official submission formats. |
| [docs/RELATED_WORK.md](RELATED_WORK.md) | Literature notes. |
| [evaluation/labels.README.md](../evaluation/labels.README.md) | How to annotate a development set. |

## Pre-AIC product documents — HISTORICAL

| Document | Why it is historical |
|---|---|
| [AUDIT.md](../AUDIT.md) | The Phase-0 audit that started the AIC work. Its findings are fixed; the audit table in `docs/CURRENT_IMPLEMENTATION_AUDIT.md` supersedes its status column. |
| [TEAM.md](../TEAM.md) | Describes the pre-AIC SigLIP product and its team split, including agent/dialogue/sketch features the competition runtime does not contain. |
| [CHATGPT_PROJECT_CONTEXT.md](../CHATGPT_PROJECT_CONTEXT.md) | A context brief for the earlier product. |
| [HUONG_DAN.md](../HUONG_DAN.md), [HUONG_DAN_GIAO_DIEN.md](../HUONG_DAN_GIAO_DIEN.md) | Vietnamese user guides for the earlier multi-tab UI. |
| `TASKS.md` below its horizontal rule | Pre-AIC engineering plan. |

**None of these describe the competition runtime**, which supports Textual KIS, Q&A and
TRAKE and nothing else. They are kept for provenance.

## Rule for new documents

State the status in the first ten lines. A document that does not say whether it is
current is treated as HISTORICAL by default, because that is the assumption that cannot
cause a false claim.
