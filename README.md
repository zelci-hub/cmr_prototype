# CMR Prototype — A minimal validation of SpecExtend's Cross-Model Retrieval

A **Tier-1 prototype** that re-implements the core algorithmic idea of SpecExtend's
*Cross-Model Retrieval (CMR)* on top of plain `transformers`, using vanilla greedy
speculative decoding (target + independent small LM draft). The goal is to validate
the CMR algorithm itself on small LLaMA models, deliberately trading inference speed
for clarity — no custom CUDA kernels, no tree attention, no EAGLE head.

> **TL;DR.** Five sanity checks verify that the hand-written spec-decoding loop is
> correct, that CMR is a bit-for-bit no-op under both implementations when
> `top_k ≥ total_chunks`, and that CMR with `--cache_pre_rotate` (re-prefill
> the draft on selected tokens with contiguous RoPE) takes a Vicuna-7B-v1.5-16k
> / vicuna-68m pair from **avg_accept_length 0.37 → 4.17** and **35.9 → 114.6
> tok/s** on 5000-token needle-in-haystack prompts. See [Results](#results).

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
├── cmr_prototype.py          # All spec-decoding + CMR logic in one file (~900 LOC)
├── prepare_sample_data.py    # Synthesises needle + repeat prompts at 1K/2K/4K words
├── run_sanity.sh             # Reproduces the five sanity checks below
├── sample_data/              # Auto-generated JSONL prompts (created by run_sanity.sh)
├── sanity_0.json             # Short-prompt correctness baseline (auto)
├── sanity_A.json             # CMR (index-select) no-op check (auto)
├── sanity_A_prerotate.json   # CMR (cache-pre-rotate) no-op check (auto)
├── sanity_B.json             # CMR (index-select) retrieval-quality check (auto)
└── sanity_C.json             # CMR (cache-pre-rotate) retrieval-quality check (auto)
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
    --compare_mode \
    --cache_pre_rotate    # enable the re-prefill variant of CMR
```

`--compare_mode` runs the same prompt through both modes (`cmr_off` / `cmr_on`) so
`avg_accept_length` and `tok/s` can be compared directly. `--cache_pre_rotate` is
a **sub-option** of CMR: on every refresh it re-runs the draft on the selected
tokens with contiguous position IDs `[0..S-1]`, so RoPE stays in-distribution. It
is a no-op when CMR is off.

---

## 5. The five sanity checks

### Sanity 0 — short-prompt correctness

Short prompts (~512 tokens) keep both models comfortably inside their training
windows, so the draft and target should agree on almost every token. If this
baseline gives ≈ 0 accepts, the bug is in the spec-decoding loop itself; if it
gives high accepts, the loop is correct and any long-prompt weakness is
attributable to the draft's limitations (e.g. RoPE extrapolation).

### Sanity A — CMR no-op (index-select path)

`top_k_chunks=1000` on a ~4K-word prompt selects every chunk, so the CMR-reduced
working cache is identical to the full draft cache. `cmr_on` and `cmr_off`
therefore **must** produce the same `avg_accept_length` to machine precision. Any
deviation means the CMR plumbing (indexing, position remapping, cache handling)
is leaking errors.

### Sanity A′ — CMR no-op (cache-pre-rotate path)

Same configuration as A but with `--cache_pre_rotate`. Re-prefilling on *all*
positions in order with contiguous IDs is bit-for-bit equivalent to a plain full
prefill, so the re-prefill plumbing (virtual positions, commit-start accounting,
`prefix_tokens` bookkeeping) must also produce `cmr_on == cmr_off`. This is the
correctness guard for the re-prefill code path.

### Sanity B — CMR retrieval quality (index-select path)

`top_k=32` on a 5000-token needle-in-haystack prompt keeps only ~20 % of the
prefix (~1024 tokens) in the draft's working cache. In the index-select path the
retained keys still carry their original (far-OOD) RoPE phases, so CMR helps only
modestly. Success criterion: `cmr_on > cmr_off` on `avg_accept_length`.

### Sanity C — CMR retrieval quality (cache-pre-rotate path)

Same prompt and `top_k=32` as B, but with `--cache_pre_rotate`. The draft now
sees a compact 1024-token virtual sequence with contiguous RoPE, which is inside
its 2K training window. Success criterion: `cmr_on ≫ cmr_off` on
`avg_accept_length` *and* `cmr_on` throughput noticeably higher than `cmr_off`
throughput. This is the experiment that, in practice, closes the gap vs. the
SpecExtend paper.

---

## 6. Results

All runs: `num_draft=5`, `chunk_size=32`, `refresh_every=4`, fp16, single H100.
Numbers below are aggregated across the samples in the corresponding
`sanity_*.json` file.

| Check    | Variant       | Prompt len | `top_k` | Dataset         | cmr_off acc | cmr_on acc | cmr_off tok/s | cmr_on tok/s |
|----------|---------------|-----------:|--------:|-----------------|------------:|-----------:|--------------:|-------------:|
| Sanity 0 | —             |        512 |    1000 | `repeat_1024`   |   **5.000** |  **5.000** |         150.2 |        135.9 |
| Sanity A | index-select  |       5000 |    1000 | `repeat_4096`   |  **0.8286** | **0.8286** |          47.6 |         41.5 |
| Sanity A′| pre-rotate    |       5000 |    1000 | `repeat_4096`   |  **0.8286** | **0.8286** |          44.7 |         38.0 |
| Sanity B | index-select  |       5000 |      32 | `needle_4096`   |   **0.368** |  **0.555** |          35.4 |         34.4 |
| Sanity C | pre-rotate    |       5000 |      32 | `needle_4096`   |   **0.368** |  **4.167** |          35.9 |    **114.6** |

### Interpretation

- **Sanity 0**: `accept_length = 5.000` is the ceiling (`num_draft=5`). Greedy
  spec-decoding loop is correct.
- **Sanity A / A′**: `0.8286 == 0.8286` to 4 decimals under both CMR code paths.
  Both implementations (index-select and re-prefill) are bit-for-bit no-ops when
  `top_k ≥ total_chunks`. The drop from 5.000 (short) to 0.83 (long) is not a
  bug: `double7/vicuna-68m` was trained on ~2K context and degrades on 5K via
  RoPE extrapolation.
- **Sanity B**: `cmr_on` delivers **+51 % relative** `avg_accept_length`
  (0.368 → 0.555) while throughput is flat (35.4 ≈ 34.4 tok/s). The extra
  compute for target-attention scoring and `index_select` is roughly offset by
  the improved acceptance, but the draft is still attending to keys whose RoPE
  rotations correspond to their *original* far-OOD positions.
- **Sanity C — the payoff**: switching to `--cache_pre_rotate` pushes
  `avg_accept_length` from 0.368 → **4.167**, i.e. **+1031 % over no-CMR** and
  **+650 % over Sanity B's index-select CMR**. At `num_draft=5`, 4.167
  corresponds to **~83 % per-slot acceptance**, matching the SpecExtend paper's
  reported range for this target/draft pair. Throughput jumps from 35.9 →
  **114.6 tok/s (≈ 3.2× speedup)** — the ~1000-token draft re-prefill every
  4 iters is cheap compared to the ~7× longer accept runs it enables.

### Why this works (and why the index-select path can't)

SpecExtend's own `modeling_llama_kv_draft.py` stores **pre-RoPE keys** and
re-applies RoPE with **contiguous** position IDs at every query — effectively
the same geometric trick as re-prefill, just implemented inside a custom
attention kernel. This prototype achieves the same outcome without a custom
kernel by re-running the draft on the selected tokens every refresh, which is
what `--cache_pre_rotate` does.

Vanilla HuggingFace `LlamaAttention`, by contrast, bakes RoPE into cached K at
insertion time, so `index_select` retains keys whose rotational phase matches
their *original* (far-OOD) position. That's exactly why Sanity B's `cmr_on` is
only 0.555 while Sanity C's is 4.167 on the same prompts and the same selected
chunks.

---

## 7. File map for readers

- `cmr_prototype.py::spec_generate` — main loop. Three draft-side code paths:
  (i) no-CMR, (ii) CMR index-select (legacy), (iii) CMR cache-pre-rotate.
- `cmr_prototype.py::target_verify` — greedy accept / off-by-one correctness
  here (see the large docstring; we feed `[last_tok, drafts]` so `logits[i]`
  aligns with `drafts[i]`).
- `cmr_prototype.py::get_last_query_attention` — extracts target's attention
  over the prefix for CMR scoring.
- `cmr_prototype.py::{build_chunks, score_chunks, select_top_k_chunks}` — the
  CMR-Algorithm-1 math.
- `cmr_prototype.py::kv_index_select` — materialises the reduced working KV in
  the index-select path.
- `cmr_prototype.py::prefill_draft_on_positions` — the re-prefill primitive used
  by the cache-pre-rotate path.

### KV cache invariants

Target-side, always:

> `full_target_kv` has length `last_pos` and does **not** contain `last_tok`.
> `last_tok` is tracked separately and re-fed as the first input on every
> verify/commit forward.

Draft-side, depending on mode:

| mode                      | draft state                                                   | draft positions           |
|---------------------------|---------------------------------------------------------------|---------------------------|
| no CMR                    | `full_draft_kv` at real positions `[0, last_pos)`             | real: `last_pos, …`       |
| CMR + index-select        | `full_draft_kv` untouched; `working = index_select(full, S)`  | real: `last_pos, …`       |
| CMR + **cache-pre-rotate**| `working_draft_kv` rebuilt on refresh via re-prefill, then extended in-place on commits | **virtual**: `virtual_base + (last_pos − real_refresh_pos), …` |

In pre-rotate mode, `virtual_base = S` right after a refresh and
`kv_len(working_draft_kv) == virtual_base + (last_pos − real_refresh_pos)` at
all times, which is exactly `draft_start_position` passed into the next
drafting call.

---

## 8. Limitations & roadmap

Current prototype:
- ✅ Correct greedy speculative decoding (Sanity 0).
- ✅ CMR bit-for-bit no-op under both code paths when `top_k ≥ total_chunks`
  (Sanity A and A′).
- ✅ CMR (index-select) provides a modest accept-length lift on long-context
  needle prompts (Sanity B, +51 %).
- ✅ CMR + `--cache_pre_rotate` closes the gap vs. SpecExtend's paper:
  accept_length 0.37 → 4.17, tok/s 35.9 → 114.6 on 5K-token needle prompts
  (Sanity C).

Planned extensions:

1. **EAGLE-3 draft support** — swap the independent draft LM for a draft head
   conditioned on target hidden states, and verify CMR + cache-pre-rotate
   compose correctly with it.
2. **Agent-style long-context benchmark** (e.g. SWE-Bench-Verified codebase
   contexts) to measure real-world CMR gain on realistic inputs rather than
   synthetic needle prompts.
3. **Amortise re-prefill cost** — currently each refresh re-runs the draft on
   ~1000 tokens; this is already net-positive (3× tok/s lift) but could be
   reduced further, e.g. by caching pre-RoPE keys once per selection and only
   re-rotating on subsequent queries.

---

## 9. Credits & references

- SpecExtend — algorithm and the reference `eval_classic.py` / custom
  `modeling_llama_kv_draft.py` we benchmark against.
- HuggingFace `transformers` — target/draft forwards.
- `lmsys/vicuna-7b-v1.5-16k` + `double7/vicuna-68m` — the target/draft pair.

This prototype is research-grade code written to validate an algorithm, not a
production inference stack. Use at your own risk, and cross-check any surprising
numbers against SpecExtend's official implementation before drawing conclusions.
