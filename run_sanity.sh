#!/usr/bin/env bash
# Sanity-check runs for the Tier-1 CMR prototype.
#
# Models: matched to SpecExtend's `eval_classic.py` defaults so numbers are
# directly comparable with the SpecExtend paper's classic (non-EAGLE) setting.
#   target = lmsys/vicuna-7b-v1.5-16k
#   draft  = double7/vicuna-68m          (68M-param Vicuna; shares tokenizer)
#
# Note: `eval_eagle.py`'s default draft (`jycha-98/EAGLE-vicuna-7b-v1.5-16k`)
# is an EAGLE-1 head and depends on target hidden states, so it cannot be used
# as a standalone LM here. We use the classic draft instead. All CMR core logic
# (chunking, attention-based selection, subset KV) is identical to the paper.
#
# CMR hyperparameters are set to SpecExtend's classic defaults:
#   retrieval_chunk_size   = 32
#   retrieve_top_k         = 32   (=> effective draft context ~1024 tokens)
#   retrieve_every_n_steps = 4

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TARGET_MODEL="lmsys/vicuna-7b-v1.5-16k"
DRAFT_MODEL="double7/vicuna-68m"
CHUNK_SIZE=32
TOP_K=32
REFRESH_EVERY=4
NUM_DRAFT=5

# 1. Build synthetic prompts. Vicuna-7B-v1.5-16k handles 16K context, so test
#    at several lengths representative of long-context agent inputs.
python3 prepare_sample_data.py \
    --output_dir sample_data \
    --num_samples 5 \
    --context_tokens 1024 2048 4096

# 2. Sanity A: top_k >= total chunks => CMR should be a no-op.
#    A 4096-word prompt / chunk_size 32 ~= 128 chunks, so top_k=1000 keeps all.
#    Expectation: cmr_on avg_accept_length ~= cmr_off avg_accept_length (delta < 0.1).
echo "==== SANITY 0: short prompt (~512 tok) baseline — validates draft OOR hypothesis ===="
# At short prompts both models operate within their training ranges, so
# accept_length should be healthy (paper reports ~2-3 for this pair at 1K).
# If this run also gives ~0 accept, the bug is in spec_generate; if this gives
# a reasonable number, the long-prompt failure is RoPE OOR in the 68M draft.
python3 cmr_prototype.py \
    --target_model "$TARGET_MODEL" \
    --draft_model  "$DRAFT_MODEL" \
    --data sample_data/repeat_1024.jsonl \
    --max_samples 3 \
    --max_prompt_tokens 512 \
    --max_new_tokens 64 \
    --num_draft "$NUM_DRAFT" \
    --chunk_size "$CHUNK_SIZE" \
    --top_k_chunks 1000 \
    --refresh_every "$REFRESH_EVERY" \
    --dtype fp16 \
    --compare_mode \
    --output_file sanity_0.json

echo "==== SANITY A: top_k=all should match cmr_off (no-op check) ===="
python3 cmr_prototype.py \
    --target_model "$TARGET_MODEL" \
    --draft_model  "$DRAFT_MODEL" \
    --data sample_data/repeat_4096.jsonl \
    --max_samples 3 \
    --max_prompt_tokens 5000 \
    --max_new_tokens 64 \
    --num_draft "$NUM_DRAFT" \
    --chunk_size "$CHUNK_SIZE" \
    --top_k_chunks 1000 \
    --refresh_every "$REFRESH_EVERY" \
    --dtype fp16 \
    --compare_mode \
    --output_file sanity_A.json

# 3. Sanity B: aggressive retrieval on the needle dataset (index-select path).
#    top_k=32 chunks * chunk_size=32 = 1024-token working draft cache from 4K+ prefix.
#    Vanilla HF bakes RoPE into cached K at insertion time, so retained keys keep
#    their original (potentially far-OOD) rotations. We expect cmr_on > cmr_off but
#    the absolute numbers remain below the SpecExtend paper because of this.
echo "==== SANITY B: top_k=$TOP_K on needle dataset (index-select; RoPE-at-insert) ===="
python3 cmr_prototype.py \
    --target_model "$TARGET_MODEL" \
    --draft_model  "$DRAFT_MODEL" \
    --data sample_data/needle_4096.jsonl \
    --max_samples 5 \
    --max_prompt_tokens 5000 \
    --max_new_tokens 32 \
    --num_draft "$NUM_DRAFT" \
    --chunk_size "$CHUNK_SIZE" \
    --top_k_chunks "$TOP_K" \
    --refresh_every "$REFRESH_EVERY" \
    --dtype fp16 \
    --compare_mode \
    --verbose \
    --output_file sanity_B.json

# 4. Sanity C: same needle dataset + top_k as B, but with --cache_pre_rotate.
#    Each CMR refresh re-prefills the draft on the selected tokens with contiguous
#    position IDs, so RoPE stays in-distribution. Expectation: cmr_on (pre-rotate)
#    >= cmr_on (index-select) from Sanity B, closing the gap vs. the SpecExtend paper.
echo "==== SANITY C: top_k=$TOP_K on needle dataset (cache_pre_rotate; RoPE-at-query) ===="
python3 cmr_prototype.py \
    --target_model "$TARGET_MODEL" \
    --draft_model  "$DRAFT_MODEL" \
    --data sample_data/needle_4096.jsonl \
    --max_samples 5 \
    --max_prompt_tokens 5000 \
    --max_new_tokens 32 \
    --num_draft "$NUM_DRAFT" \
    --chunk_size "$CHUNK_SIZE" \
    --top_k_chunks "$TOP_K" \
    --refresh_every "$REFRESH_EVERY" \
    --dtype fp16 \
    --compare_mode \
    --cache_pre_rotate \
    --verbose \
    --output_file sanity_C.json

# 5. Sanity A': cache_pre_rotate must ALSO be ~no-op when top_k covers all chunks.
#    This checks that the re-prefill plumbing (virtual positions, commit accounting)
#    is itself correct: re-prefilling on all positions in order with contiguous IDs
#    should agree with a normal prefill, so cmr_on ~= cmr_off.
echo "==== SANITY A': cache_pre_rotate + top_k=all (no-op check for re-prefill path) ===="
python3 cmr_prototype.py \
    --target_model "$TARGET_MODEL" \
    --draft_model  "$DRAFT_MODEL" \
    --data sample_data/repeat_4096.jsonl \
    --max_samples 3 \
    --max_prompt_tokens 5000 \
    --max_new_tokens 64 \
    --num_draft "$NUM_DRAFT" \
    --chunk_size "$CHUNK_SIZE" \
    --top_k_chunks 1000 \
    --refresh_every "$REFRESH_EVERY" \
    --dtype fp16 \
    --compare_mode \
    --cache_pre_rotate \
    --output_file sanity_A_prerotate.json

echo
echo "Done."
echo "  sanity_0.json            : short-prompt correctness baseline"
echo "  sanity_A.json            : CMR (index-select) no-op sanity check"
echo "  sanity_A_prerotate.json  : CMR (cache_pre_rotate) no-op sanity check"
echo "  sanity_B.json            : CMR (index-select) retrieval quality at long context"
echo "  sanity_C.json            : CMR (cache_pre_rotate) retrieval quality at long context"
