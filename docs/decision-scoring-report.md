# Decision scoring: verification report

Branch `decision-scoring` (worktree `~/splash-score`). Measured on the studio
(M3 Ultra, 256 GB) against the production instance
`unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL` on :8001 (`--max-memory 96G`,
`--max-context 48K`) on 4 Oct 2026, with production traffic running. Tool:
`dev/benchmarks/score_verify.py`. Raw records are in
`~/splash-bench/score/35b.jsonl`.

Qwen3.8-Flash-Next (:8003) is not yet verified. It needs its own window,
because a test instance does not fit beside production (~118 GB).

## Design in brief

`/v1/score` renders messages through Chat's own `_prepare_prompt` and
`_render_prompt`, then appends `assistant_prefix`. Each label must be one
token in that position. A native score-only request reads the final
position's fp32 logits. Native protocol version 8 adds the log-sum-exp over
the whole vocabulary and the top 20 tokens. A prefill command that contains a
score request runs its projections split-free. That makes the logits
independent of packing and of where the prefix cache resumes (see "Finding").
Details: `docs/decision-scoring.md`.

## Results (35B)

| # | Check | Result |
| --- | --- | --- |
| 1 | Agreement with generation, 300 prompts | **300/300** against `/v1/chat/completions` at T=0 (the label written after the model's own prefix) and **300/300** against `/v1/completions` over the same prompt. Synthetic product-matching prompts; the client's JSONL is still to come. |
| 2 | Determinism, 20× | max spread **0.0** |
| 3 | Concurrency 1 / 4 / 24 | argmax identical, max prob difference **0.0 / 0.0 / 0.0** |
| 4 | Cold vs warm prefix cache | max logprob difference **0.0** (8 pairs; also 8 long prompts of 1.1k–3.6k tokens resumed 18–40 tokens before the end) |
| 5 | Sanity | probabilities sum to 1, `label_mass` in [0, 1], a multi-token label returns 400 with its tokens (`" true false"` → 804 `" true"`, 867 `" false"`) |
| 6 | Performance | see the table below |
| 7 | Memory, 10 000 requests | no growth after the KV cache fills to its budget (~1 500 requests): server RSS 721 → 725 MB, engine RSS 76.97 GB, GPU memory 102.28 GB, 0 active KV pages, cached pages 1.7–2.4k |
| 8 | Image | 2 images (red circle / green triangle, "Is there a red circle?"): argmax " Yes" 0.9991 / " No" 0.9998, both equal to the generated answer, `label_mass` 1.0; the image prefix comes from the cache (256 of 272 tokens) |
| — | Chat regression | 15/15 Chat outputs bit-identical to the old build in every run with no other client's traffic; the runs that differed had 4–8 concurrent foreign requests, which change greedy output by batching on the old build too |

### Performance (96 prompts, generation `max_tokens` 80)

| Concurrency | score p50 / p95 | score req/min | generate p50 / p95 | generate req/min | speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 65 / 74 ms | 955 | 415 / 820 ms | 138 | 6.9× |
| 4 | 133 / 182 ms | 1688 | 1009 / 2072 ms | 222 | 7.6× |
| 12 | 207 / 602 ms | 2248 | 2525 / 4172 ms | 271 | 8.3× |
| 24 | 618 / 877 ms | 2300 | 5207 / 7485 ms | 259 | 8.9× |

GPU time over the whole run: prefill 32.4 s and decode 85.7 s. Decode time
is generation only; scoring has none.

Packing: a prefill command carries at most 4 sequences
(`ExecutionLimits::maximumBatchWidth`). Throughput levels off at about
2 300 requests/min from concurrency 12 on. Packing more score sequences would
mean larger state and arena allocations; that is not done here.

## Finding: batch-dependent prefill numerics

Before the fix, the same score request alone and packed with others differed
by up to **0.63** in a label logit (**0.09** in probability, with the same
argmax). The difference did not depend on the partner's content. It came
from the GGUF prefill plan: chunks of up to 32 rows run the decode tile with
a K split, and the split reorders fp32 sums. In a 512-expert MoE model a
reordered sum can flip a top-k expert choice, which propagates through the
layers. Packed batches and cache-resumed suffixes make small chunks.

Turning the split off everywhere made scoring exact. It also changed 7 of 15
Chat outputs on warm-cache prompts, so the change is limited to prefill
commands that contain a score request (`splitFree`). Chat keeps its plans.

## Known limits

- Bit-identical scores across cache states assume the shared prefix was
  computed by score prefills, or by Chat chunks of more than 32 rows.
- Answers must be single tokens in their position. `label_mass` shows whether
  the labels match what the model writes. "Yes" without a leading space had
  a mass of about 0.0 where the model writes " Yes".
- For the synthetic prompts the model writes `{\n"same":` (sometimes
  `{\n  "same":`), not `{"same":`. The prefix must match the client's actual
  output; check it on the real prompts.
- The SSD tier does not keep Flash-Next's QSA index keys.
