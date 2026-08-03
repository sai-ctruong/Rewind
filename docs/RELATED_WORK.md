# Related Work Gate And Research Hypotheses

The systematic paper review is intentionally deferred. The repository has no AIC-format ground truth, so the prompt's prerequisite, a benchmarked pipeline on a fixed AIC development split, is not yet satisfied. No novelty, first-method, state-of-the-art or superiority claim is made.

After labels are supplied, review primary papers from CVPR, ICCV, ECCV, ACM ICMR, NeurIPS, ACL/EMNLP and original arXiv versions for text-video retrieval, temporal grounding, moment retrieval, temporal action localization, VideoQA grounding, frame sampling, dynamic-programming alignment and multimodal fusion. Record citation, venue/year, problem, method, dataset, metric, code, similarity, difference, limitation and novelty risk for every paper.

## Testable Hypotheses

### H1 - Event-Coverage Video Ranking

- Baseline: maximum single-event frame score per video.
- Proposed: coverage-aware video hypothesis score before alignment.
- Metrics: TRAKE video retrieval accuracy, mean event accuracy and Final Score.
- Ablation: `greedy_alignment` versus `dp_alignment` with and without coverage bonus.
- Expected failures: ambiguous repeated events and missing per-event candidates.

### H2 - Monotonic DP Alignment

- Baseline: independent retrieval plus hard order filtering.
- Proposed: monotonic DP with min/max gaps, transition cost and missing-event penalty.
- Metrics: partial R-score, all-events-correct accuracy and frame distance.
- Ablation: hard order, greedy, DP and DP plus refinement on one fixed split.
- Expected failures: simultaneous events, wrong event decomposition and sparse keyframes.

### H3 - Uncertainty-Triggered Refinement

- Baseline: mapped coarse keyframe and uniform dense local decoding.
- Proposed: bounded local decoding only for low margin/conflicting evidence.
- Metrics: frame accuracy, decoded frames/query, latency and peak memory.
- Ablation: keyframe-only, always refine and uncertainty-only refine.
- Expected failures: missing MP4, domain mismatch in the frame scorer and very fast actions between sampled frames.
