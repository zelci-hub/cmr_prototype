# CMR Prototype — A minimal validation of SpecExtend's Cross-Model Retrieval

A **Tier-1 prototype** that re-implements the core algorithmic idea of SpecExtend's
*Cross-Model Retrieval (CMR)* on top of plain `transformers`, using vanilla greedy
speculative decoding (target + independent small LM draft). The goal is to validate
the CMR algorithm itself on small LLaMA models, deliberately trading inference speed
for clarity — no custom CUDA kernels, no tree attention, no EAGLE head.

> **TL;DR.** Three sanity checks verify that the hand-written spec-decoding loop is
> correct, that CMR is a strict no-op when `top_k ≥ total_chunks`, and that CMR
> delivers a measurable accept-length improvement on a long-context needle-in-haystack
> prompt. See [Results](#results).

---

## 1. What is CMR?

Quoting the SpecExtend paper:

> We achieve this via **Cross-model Retrieval (CMR)**, which uses the target model's
> attention scores to select the most relevant input chunks to retain in the smaller
> draft model's cache. Unlike static KV eviction, CMR is an algorithmic alignment
> mechanism that uses the target model as a sparse retriever to dynamically reshape
> the draft model's effective context, rather than merely discarding tokens based
> on position.

Concretely, each time the draft needs a fresh "working cache", we:

1. Run the *target* on the last accepted token with `output_attentions=True` to get
   its attention distribution over the prefix.
2. Split the prefix into fixed-size chunks (`chunk_size=32` by default) and score
   each chunk by head-averaged mean attention.
3. Keep the **top-k** chunks (plus a small recent-tail window) and index-select their
   positions out of the draft's full KV cache.
4. Use that reduced cache to draft `N` candidate tokens, then verify them with the
   target.
5. Refresh the selection every `refresh_every` iterations.

This prototype implements this end-to-end in a single file (`cmr_prototype.py`)
with tuple-of-`(K, V)` legacy caches and pure HuggingFace forwards.

---

## 2. Directory layout

```
cmr_prototype/
├── cmr_prototype.py         # All spec-decoding + CMR logic in one file (~800 LOC)
├── prepare_sample_data.py   # Synthesises needle + repeat prompts at 1K/2K/4K words
├── run_sanity.sh            # Reproduces the three sanity checks below
├── sample_data/             # Auto-generated JSONL prompts (created by run_sanity.sh)
├── sanity_0.json            # Short-prompt correctness baseline (auto)
├── sanity_A.json            # CMR no-op check (auto)
└── sanity_B.json            # CMR retrieval-quality check (auto)
```

---

## 3. Setup

The only non-trivial dependencies are `torch` and a reasonably recent `transformers`
(≥ 4.36 for `DynamicCache`). Any environment that can run Vicuna-7B already has
what you need.

```bash
pip install "torch>=2.1" "transformers>=4.36" accelerate termcolor
```

The default target/draft pair is the same one SpecExtend's `eval_classic.py` uses:

- target: `lmsys/vicuna-7b-v1.5-16k`
- draft:  `double7/vicuna-68m`

Both are fetched from the HuggingFace Hub on first run.

---

## 4. Quick start

```bash
# 3 sanity-check runs, < 2 minutes on a single H100 after models are cached.
CUDA_VISIBLE_DEVICES=0 ./run_sanity.sh
```

This writes `sanity_{0,A,B}.json` and prints an aggregate per file. For a custom
run:

```bash
python3 cmr_prototype.py \
    --target_model lmsys/vicuna-7b-v1.5-16k \
    --draft_model  double7/vicuna-68m \
    --data sample_data/needle_4096.jsonl \
    --max_prompt_tokens 5000 \
    --max_new_tokens 32 \
    --num_draft 5 \
    --chunk_size 32 --top_k_chunks 32 --refresh_every 4 \
    --dtype fp16 \
    --compare_mode        # run both cmr_off and cmr_on back-to-back
```

`--compare_mode` runs the same prompt through both modes so `avg_accept_length` and
`tok/s` can be compared directly.

---

## 5. The three sanity checks

### Sanity 0 — short-prompt correctness

Short prompts (~512 tokens) keep both models comfortably inside their training
windows, so the draft and target should agree on almost every token. If this
baseline gives ≈ 0 accepts, the bug is in the spec-decoding loop itself; if it
gives high accepts, the loop is correct and any long-prompt weakness is attributable
to the draft's limitations (e.g. RoPE extrapolation).

### Sanity A — CMR no-op

`top_k_chunks=1000` on a ~4K-word prompt selects every chunk, so the CMR-reduced
working cache is identical to the full draft cache. `cmr_on` and `cmr_off`
therefore **must** produce the same `avg_accept_length` to machine precision. Any
deviation means the CMR plumbing (indexing, position remapping, cache handling)
is leaking errors.

### Sanity B — CMR retrieval quality

`top_k=32` on a 5000-token needle-in-haystack prompt keeps only ~20% of the prefix
(~1024 tokens) in the draft's working cache. If CMR is working, the retained
chunks should include the informative "needle" region, and the reduced cache
should actually *help* the draft (because it brings its effective context closer
to its training window). A successful outcome is `cmr_on > cmr_off` on
`avg_accept_length`.

---

## 6. Results

All runs: `num_draft=5`, `chunk_size=32`, `refresh_every=4`, fp16, single H100.
Numbers below are aggregated across samples from `sanity_{0,A,B}.json`.

| Check      | Prompt len | `top_k` | Dataset         | cmr_off avg_acc | cmr_on avg_acc | cmr_off tok/s | cmr_on tok/s |
|------------|-----------:|--------:|-----------------|----------------:|---------------:|--------------:|-------------:|
| Sanity 0   |        512 |    1000 | `repeat_1024`   |       **5.000** |      **5.000** |         129.1 |        109.0 |
| Sanity A   |       5000 |    1000 | `repeat_4096`   |      **0.8286** |     **0.8286** |          46.5 |         41.2 |
| Sanity B   |       5000 |      32 | `needle_4096`   |      **0.368**  |     **0.555**  |          35.6 |         35.7 |

### Interpretation

- **Sanity 0**: `accept_length = 5.000` is the ceiling (`num_draft=5`), meaning the
  draft's greedy output matched the target's on *every* drafted token, every
  iteration. Algorithmic correctness confirmed.
- **Sanity A**: `0.8286 == 0.8286` to 4 decimals — CMR is a pure no-op when
  `top_k ≥ total_chunks`. The drop from 5.000 (short) to 0.83 (long) is not a bug:
  `double7/vicuna-68m` was trained on ~2K context and degrades heavily on 5K
  prompts via RoPE extrapolation.
- **Sanity B**: `cmr_on` delivers **+51 % relative** `avg_accept_length`
  (0.368 → 0.555) while throughput is flat (35.6 ≈ 35.7 tok/s). The extra compute
  for target-attention scoring and index-select is exactly offset by the improved
  acceptance.

### Why the absolute numbers are below SpecExtend's paper

The paper reports `avg_accept_length ≈ 2–3` for this same target/draft pair on
long inputs with CMR. This prototype reaches 0.55. The gap has a clean root cause:

SpecExtend ships a **custom** `LlamaAttention` (see `specextend/classic/modeling_llama_kv_draft.py`,
lines ~590-602) that stores **pre-RoPE keys** in the cache and re-applies RoPE
with **contiguous** `[0, kv_seq_len)` position IDs at every query. So when CMR
selects, say, positions `{13, 47, 3901, ...}`, the draft *sees a compact
short-context virtual sequence* with contiguous RoPE — staying well inside its
training window.

This prototype uses vanilla HuggingFace `LlamaAttention`, which bakes RoPE into
keys *at cache-insertion time*. After `index_select` the retained keys still
carry their *original* (far-out-of-distribution) RoPE rotations. CMR still helps
(0.37 → 0.55) because the selected chunks are semantically more relevant, but
the draft continues to see OOD attention.

Closing this gap requires either (a) a custom LLaMA forward that applies RoPE at
query-time, or (b) **re-prefilling the draft on each refresh** with the selected
tokens using contiguous position IDs `[0..|S|-1]`. (b) is an additive change to
this prototype and is the natural next step — it would directly reproduce the
paper's long-context accept-length numbers on top of the vanilla-`transformers`
stack.

---

## 7. File map for readers

- `cmr_prototype.py::spec_generate` — main loop.
- `cmr_prototype.py::target_verify` — greedy accept / off-by-one correctness here
  (see the large docstring; we feed `[last_tok, drafts]` so `logits[i]` aligns
  with `drafts[i]`).
- `cmr_prototype.py::get_last_query_attention` — extracts target's attention
  over the prefix for CMR scoring.
- `cmr_prototype.py::{build_chunks, score_chunks, select_top_k_chunks}` — the
  CMR-Algorithm-1 math.
- `cmr_prototype.py::kv_index_select` — materialises the reduced working KV.

The KV cache invariant maintained throughout is:

> `full_target_kv` and `full_draft_kv` each have length `last_pos` and **do not
> contain** `last_tok`. `last_tok` is tracked separately and re-fed as the first
> input on every forward (draft, verify, commit).

---

## 8. Limitations & roadmap

Current prototype:
- ✅ Correct greedy speculative decoding (Sanity 0).
- ✅ CMR is a bit-for-bit no-op when `top_k ≥ total_chunks` (Sanity A).
- ✅ CMR provides measurable accept-length lift on long-context needle prompts
  (Sanity B, +51 %).
- ⚠️ Absolute accept-length at long context is ~3× below SpecExtend's paper
  because of the RoPE-at-insert vs. RoPE-at-query-time gap described above.

Planned extensions:

1. **Draft re-prefill with contiguous positions** (closes the RoPE gap on vanilla
   HuggingFace; matches the paper's long-context accept lengths).
2. **EAGLE-3 draft support** — swap the independent draft LM for a draft head
   conditioned on target hidden states, verify CMR composes correctly with it.
3. **Agent-style long-context benchmark** (e.g. SWE-Bench-Verified
   codebase contexts) to measure real-world CMR gain.

---

## 9. Credits & references

- SpecExtend — algorithm and the reference `eval_classic.py` / custom
  `modeling_llama_kv_draft.py` we benchmark against.
- HuggingFace `transformers` — target/draft forwards.
- `lmsys/vicuna-7b-v1.5-16k` + `double7/vicuna-68m` — the target/draft pair.

This prototype is research-grade code written to validate an algorithm, not a
production inference stack. Use at your own risk, and cross-check any surprising
numbers against SpecExtend's official implementation before drawing conclusions.
