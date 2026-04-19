#!/usr/bin/env python3
"""Tier-1 prototype of Cross-Model Retrieval (CMR) for speculative decoding.

This script runs vanilla two-model speculative decoding (a small draft LM drafting
for a larger target LM) on long-context prompts, with an optional CMR path that
selects only the top-k most relevant prefix chunks for the draft to attend over.

Goals (intentional scope limits):
    * Validate that the CMR algorithm is implemented correctly and that its effect
      on the average acceptance length matches the expectations from the
      SpecExtend paper (higher or on-par accept length vs. no-CMR at long contexts
      with a much smaller draft context budget).
    * Keep the speculative method simple (greedy, linear drafting of N tokens,
      batch size 1) so CMR can be isolated from tree-attention / EAGLE effects.
    * Structure the draft interface (`draft_generate_N`) so an EAGLE-3 draft can
      be dropped in later (Tier-2).

CMR follows Algorithm 1 of SpecExtend:
    1. Split the prefix into fixed-size chunks (default 32 tokens).
    2. Every `refresh_every` accepted steps, re-run the last-accepted token
       through the target with `output_attentions=True` to obtain the last
       query's attention over the prefix.
    3. Average the attention per chunk and take the top-k chunks.
    4. Build a "working" draft KV cache that contains ONLY those top-k chunks'
       positions (plus a small `recent_window` of always-kept tail positions).
    5. The draft attends over the working cache when producing candidate tokens.
       The full draft KV cache is preserved untouched so refreshes are cheap.

Greedy verification (temperature=0) matches SpecExtend's default eval setting.

Usage (sanity check on tiny models)::

    python cmr_prototype.py \
        --target_model meta-llama/Llama-3.2-3B-Instruct \
        --draft_model  meta-llama/Llama-3.2-1B-Instruct \
        --data sample_data/longctx.jsonl \
        --max_samples 5 --compare_mode
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import DynamicCache  # >= 4.36
    _HAS_DYNAMIC_CACHE = True
except ImportError:  # pragma: no cover
    DynamicCache = None  # type: ignore
    _HAS_DYNAMIC_CACHE = False


def _as_cache_for_forward(past_kv):
    """Convert a legacy tuple KV into a ``DynamicCache`` if needed.

    Newer ``transformers`` versions only auto-convert legacy tuple caches when
    ``use_cache=True``. For read-only forwards with ``use_cache=False`` the
    tuple is passed through unchanged and the inner model calls
    ``.get_seq_length()`` on it, which raises ``AttributeError``. This helper
    normalizes the input so the forward works regardless of ``use_cache``.
    """
    if past_kv is None:
        return None
    if _HAS_DYNAMIC_CACHE and isinstance(past_kv, tuple):
        return DynamicCache.from_legacy_cache(past_kv)
    return past_kv


# ---------------------------------------------------------------------------
# KV cache helpers (tuple-of-(k,v) legacy format; works across transformers >= 4.40)
# ---------------------------------------------------------------------------

KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def _to_legacy(past_kv) -> KVCache:
    """Normalize past_key_values to a plain tuple-of-tuples format."""
    if past_kv is None:
        return None
    if hasattr(past_kv, "to_legacy_cache"):
        return past_kv.to_legacy_cache()
    return tuple(tuple(layer) for layer in past_kv)


def _in_dev(model) -> torch.device:
    """Device where the model expects input_ids (= input embedding device).

    When ``device_map="auto"`` pipeline-shards the model, input_ids must go to
    the device of the embedding layer, NOT ``next(model.parameters()).device``
    (which may pick a later layer's device).
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def kv_index_select(kv: KVCache, positions: torch.Tensor) -> KVCache:
    """Select positions along the seq dim of a KV cache.

    Args:
        kv: tuple of (K, V) per layer. Each tensor is [B, num_kv_heads, L, head_dim].
        positions: 1D long tensor of positions to keep (values in [0, L)).

    Positions are moved to each layer's device so this works with pipeline-
    sharded KV caches where different layers live on different GPUs.
    """
    out = []
    for (k, v) in kv:
        pos = positions.to(k.device) if positions.device != k.device else positions
        out.append((k.index_select(2, pos), v.index_select(2, pos)))
    return tuple(out)


def kv_cat_seq(kv_a: KVCache, kv_b: KVCache) -> KVCache:
    """Concatenate two KV caches along the seq dim."""
    out = []
    for (ka, va), (kb, vb) in zip(kv_a, kv_b):
        out.append((torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2)))
    return tuple(out)


def kv_truncate(kv: KVCache, length: int) -> KVCache:
    """Keep only the first `length` positions."""
    out = []
    for (k, v) in kv:
        out.append((k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous()))
    return tuple(out)


def kv_len(kv: Optional[KVCache]) -> int:
    if kv is None:
        return 0
    return kv[0][0].shape[2]


# ---------------------------------------------------------------------------
# CMR core: chunking and top-k selection
# ---------------------------------------------------------------------------

def build_chunks(prefix_len: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Split [0, prefix_len) into consecutive chunks of size `chunk_size`.

    Returns: list of (start, end) tuples; last chunk may be shorter.
    """
    chunks = []
    start = 0
    while start < prefix_len:
        end = min(start + chunk_size, prefix_len)
        chunks.append((start, end))
        start = end
    return chunks


def score_chunks(attn_weights: torch.Tensor, chunks: List[Tuple[int, int]]) -> torch.Tensor:
    """Average attention weight inside each chunk (vectorized via cumsum).

    Args:
        attn_weights: [prefix_len] float tensor of per-position attention scores.
        chunks: list of (start, end).
    Returns:
        [num_chunks] mean-per-chunk tensor.
    """
    attn = attn_weights.to(torch.float32)
    cum = torch.cat([attn.new_zeros(1), attn.cumsum(dim=0)])  # [L+1]
    starts = torch.tensor([c[0] for c in chunks], device=attn.device, dtype=torch.long)
    ends = torch.tensor([c[1] for c in chunks], device=attn.device, dtype=torch.long)
    sums = cum[ends] - cum[starts]
    lengths = (ends - starts).clamp(min=1).to(torch.float32)
    return sums / lengths


def select_top_k_chunks(
    chunk_scores: torch.Tensor,
    chunks: List[Tuple[int, int]],
    top_k: int,
    always_keep_tail: int = 1,
) -> List[Tuple[int, int]]:
    """Top-k chunks by score; optionally force-keep the last `always_keep_tail`
    chunks so the draft always sees recent context.
    Returns chunks sorted by original order (start ascending).
    """
    n = len(chunks)
    if top_k >= n:
        return list(chunks)
    forced = set(range(max(0, n - always_keep_tail), n))
    remaining_budget = max(0, top_k - len(forced))
    # Score-based selection excluding forced indices
    if remaining_budget > 0:
        mask = torch.ones_like(chunk_scores, dtype=torch.bool)
        for i in forced:
            mask[i] = False
        scored_idx = torch.arange(n, device=chunk_scores.device)[mask]
        scores_filtered = chunk_scores[mask]
        k = min(remaining_budget, scores_filtered.numel())
        topk = torch.topk(scores_filtered, k=k)
        selected = set(scored_idx[topk.indices].tolist()) | forced
    else:
        selected = forced
    selected_sorted = sorted(selected)
    return [chunks[i] for i in selected_sorted]


def chunks_to_positions(selected: List[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    """Flatten [(s1,e1),(s2,e2),...] into a 1D long tensor of positions."""
    pieces = [torch.arange(s, e, device=device, dtype=torch.long) for (s, e) in selected]
    if not pieces:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(pieces, dim=0)


# ---------------------------------------------------------------------------
# Draft re-prefill (used by the cache_pre_rotate option)
# ---------------------------------------------------------------------------

@torch.no_grad()
def prefill_draft_on_positions(
    draft_model,
    prefix_tokens: torch.Tensor,    # [1, last_pos], CPU long tensor: all committed tokens
    positions: torch.Tensor,        # [S] CPU long tensor, strictly ascending, values in [0, last_pos)
) -> KVCache:
    """Re-prefill the draft model on the tokens at ``positions`` with CONTIGUOUS
    position IDs ``[0, 1, ..., S-1]``.

    This is the core of the ``cache_pre_rotate`` option: vanilla HuggingFace
    LLaMA bakes RoPE into cached K at insertion time, so an ``index_select``
    over positions like ``{13, 47, 3901, ...}`` leaves retained keys rotated
    with far-out-of-distribution phases. By re-running the draft on just the
    selected tokens with positions ``[0..S-1]``, RoPE is applied in-distribution
    and the draft effectively sees a dense short-context virtual sequence.

    The caller is responsible for choosing ``positions`` (e.g. from CMR top-k
    selection) and for subsequently extending the returned cache with virtual
    position IDs ``S, S+1, ...``.
    """
    dev = _in_dev(draft_model)
    tokens = prefix_tokens.index_select(1, positions).to(dev)  # [1, S]
    S = tokens.shape[1]
    pos_ids = torch.arange(S, device=dev, dtype=torch.long).unsqueeze(0)
    out = draft_model(
        input_ids=tokens,
        position_ids=pos_ids,
        use_cache=True,
        output_attentions=False,
    )
    return _to_legacy(out.past_key_values)


# ---------------------------------------------------------------------------
# Target attention extraction (for CMR refresh)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_last_query_attention(
    target_model,
    full_target_kv: KVCache,
    last_token_id: torch.Tensor,   # [1, 1]
    prefix_len: int,               # total prefix length BEFORE the last token, i.e. last-token position is prefix_len
    layer_idx: int = -1,
) -> torch.Tensor:
    """Re-run the last accepted token through the target to capture its attention
    to the earlier prefix. Returns [prefix_len] attention weights (head-averaged).

    INVARIANT: ``full_target_kv`` has length ``prefix_len`` and does NOT contain
    ``last_token_id`` (the token sits at position ``prefix_len``).
    """
    assert kv_len(full_target_kv) == prefix_len, (
        f"full_target_kv has {kv_len(full_target_kv)} tokens but prefix_len={prefix_len}"
    )
    dev = _in_dev(target_model)
    ids = last_token_id.to(dev)
    pos = torch.tensor([[prefix_len]], device=dev, dtype=torch.long)

    # With use_cache=False newer transformers won't auto-convert legacy tuple
    # caches, so we do it ourselves. The cache is read-only since use_cache=False.
    kv_cache = _as_cache_for_forward(full_target_kv)

    out = target_model(
        input_ids=ids,
        past_key_values=kv_cache,
        position_ids=pos,
        output_attentions=True,
        use_cache=False,
    )
    # out.attentions: tuple of [B, num_heads, Q=1, K=prefix_len+1] per layer
    attn = out.attentions[layer_idx]  # [1, H, 1, prefix_len+1]
    # Drop the self-attention column (position prefix_len); keep attention over prefix only.
    attn = attn[:, :, 0, :prefix_len]  # [1, H, prefix_len]
    attn_avg = attn.mean(dim=1).squeeze(0)  # [prefix_len]
    return attn_avg.detach().to(torch.float32).cpu()


# ---------------------------------------------------------------------------
# Draft forward: greedy N-token drafting
# ---------------------------------------------------------------------------

@torch.no_grad()
def draft_generate_N(
    draft_model,
    last_token_id: torch.Tensor,     # [1, 1]  (the last accepted token)
    working_past_kv: KVCache,        # draft KV containing the working set (possibly subset of full)
    num_draft: int,
    start_position: int,             # position (in ORIGINAL target coordinates) of `last_token_id`
) -> Tuple[torch.Tensor, KVCache]:
    """Greedy autoregressive drafting of `num_draft` tokens.

    Returns
    -------
    draft_tokens : [1, num_draft] long tensor (tokens sampled greedily by draft)
    updated_kv   : draft KV cache after drafting (contains `working_past_kv` plus
                   `num_draft + 1` newly appended KV entries for the drafted tokens
                   and the last-accepted token itself, in that order).
    """
    dev = _in_dev(draft_model)
    draft_tokens = []
    cur_input = last_token_id.to(dev)
    cur_pos = start_position
    kv = working_past_kv

    for _ in range(num_draft):
        pos_ids = torch.tensor([[cur_pos]], device=dev, dtype=torch.long)
        out = draft_model(
            input_ids=cur_input,
            past_key_values=kv,
            position_ids=pos_ids,
            use_cache=True,
            output_attentions=False,
        )
        logits = out.logits[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
        draft_tokens.append(next_tok)
        kv = _to_legacy(out.past_key_values)
        cur_input = next_tok
        cur_pos += 1

    draft_tokens = torch.cat(draft_tokens, dim=1)  # [1, N]
    return draft_tokens, kv


# ---------------------------------------------------------------------------
# Target verification: greedy match against drafted tokens
# ---------------------------------------------------------------------------

@torch.no_grad()
def target_verify(
    target_model,
    full_target_kv: KVCache,
    last_accepted_tok: torch.Tensor,  # [1, 1] the last committed token; NOT in full_target_kv
    draft_tokens: torch.Tensor,       # [1, N]
    prefix_len: int,                  # length of full_target_kv (= position where last_accepted_tok lives)
):
    """Greedy speculative verification.

    INVARIANT on input: ``full_target_kv`` contains KV for positions ``[0, prefix_len)``
    and does NOT include ``last_accepted_tok`` (which sits at position ``prefix_len``).

    We feed ``[last_accepted_tok, draft_tokens[0..N-1]]`` at positions
    ``[prefix_len, prefix_len+1, ..., prefix_len+N]`` against ``full_target_kv``. This
    yields N+1 logits; ``logits[i]`` is target's prediction for position
    ``prefix_len+i+1`` given everything up to and including the i-th input token:

        logits[0] = P(next | prompt + last_tok)                -> compare to draft_tokens[0]
        logits[i] = P(next | prompt + last_tok + drafts[:i])   -> compare to draft_tokens[i]
        logits[N] = P(next | prompt + last_tok + drafts[:N])   -> free bonus if all accepted.

    Returns
    -------
    accept_count      : int in [0, N].
    corrected_token   : [1, 1] long tensor, target's argmax at the first mismatch
                        position (or the bonus prediction if all drafts accepted).
    new_full_kv       : truncated cache of length ``prefix_len + accept_count + 1``,
                        containing [prefix, last_accepted_tok, drafts[:accept_count]].
                        The corrected token itself is NOT in this cache (it becomes the
                        caller's new ``last_tok`` and is committed in its own commit step).
    """
    N = draft_tokens.shape[1]
    dev_t = _in_dev(target_model)
    last = last_accepted_tok.to(dev_t)
    drafts = draft_tokens.to(dev_t)
    # inputs: [last_tok, drafts[0], ..., drafts[N-1]], length N+1
    inputs = torch.cat([last, drafts], dim=1)
    positions = torch.arange(
        prefix_len, prefix_len + N + 1, device=dev_t, dtype=torch.long
    ).unsqueeze(0)
    out = target_model(
        input_ids=inputs,
        past_key_values=full_target_kv,
        position_ids=positions,
        use_cache=True,
        output_attentions=False,
    )
    logits = out.logits  # [1, N+1, V]
    target_argmax = torch.argmax(logits, dim=-1)  # [1, N+1]

    # Compare target_argmax[0..N-1] against draft_tokens[0..N-1]
    match = (target_argmax[:, :N] == drafts).squeeze(0)  # [N]
    if match.numel() == 0 or match.all():
        accept_count = N
    else:
        accept_count = int(torch.nonzero(~match, as_tuple=False)[0].item())

    # Cache now has length prefix_len + N + 1 (prefix + last_tok + all drafts).
    extended_kv = _to_legacy(out.past_key_values)
    # Truncate to [prefix + last_tok + accepted_drafts]. The bonus/corrected token is
    # not in the cache: the caller commits it as the new last_tok in a subsequent step.
    kept_len = prefix_len + 1 + accept_count
    new_full_kv = kv_truncate(extended_kv, kept_len)

    # corrected = target's prediction at the first unaccepted position.
    # target_argmax has N+1 entries; index ``accept_count`` is well-defined for both
    # accept_count < N (first mismatch) and accept_count == N (bonus prediction).
    corrected = target_argmax[:, accept_count:accept_count + 1]  # [1, 1]
    return accept_count, corrected, new_full_kv


# ---------------------------------------------------------------------------
# Main speculative generation loop with optional CMR
# ---------------------------------------------------------------------------

@dataclass
class SpecResult:
    generated_ids: torch.Tensor
    accept_lens: List[int]
    num_iters: int
    num_new_tokens: int
    wall_time: float


@torch.no_grad()
def spec_generate(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    num_draft: int,
    eos_token_id: Optional[int],
    *,
    use_cmr: bool,
    cache_pre_rotate: bool = False,
    chunk_size: int = 32,
    top_k_chunks: int = 64,
    refresh_every: int = 4,
    recent_tail_chunks: int = 2,
    verbose: bool = False,
) -> SpecResult:
    """Run greedy speculative decoding with (optional) Cross-Model Retrieval.

    Parameters of interest
    ----------------------
    use_cmr : bool
        If True, every ``refresh_every`` iterations use the target's attention
        over the prefix to select the top-k chunks for the draft to attend to.
    cache_pre_rotate : bool
        Only meaningful when ``use_cmr=True``. If True, on each CMR refresh
        the draft is **re-prefilled** on the selected tokens with contiguous
        position IDs ``[0..S-1]``, so RoPE is applied in-distribution rather
        than carrying the original (potentially far-OOD) rotations of the
        retained keys. Between refreshes, new tokens are appended with virtual
        contiguous positions ``S, S+1, ...``. This corresponds to SpecExtend's
        "store pre-RoPE K + reapply at query" behaviour, implemented purely
        on top of vanilla HF by re-running the draft on each refresh.

    See module docstring for algorithm description.
    """
    dev_t = _in_dev(target_model)
    dev_d = _in_dev(draft_model)
    prompt_len = input_ids.shape[1]

    pre_rotate = bool(use_cmr and cache_pre_rotate)  # sub-option of CMR

    # ==========================================================================
    # Shared invariant:
    #   full_target_kv always contains KV for real positions [0, last_pos).
    #   It does NOT include ``last_tok`` (which sits at position ``last_pos``
    #   and is re-fed as input on each drafting/verify/commit forward).
    #
    # Draft-side state depends on the mode:
    #   * pre_rotate=False : full_draft_kv has real positions [0, last_pos);
    #     CMR subselects via kv_index_select.
    #   * pre_rotate=True  : full_draft_kv is unused. working_draft_kv holds
    #     virtual-position KV:
    #         [re-prefilled selection] + [tokens committed since refresh]
    #     at contiguous virtual positions [0, virtual_base + commits_since_refresh).
    #     virtual_last_pos = virtual_base + (last_pos - real_refresh_pos) is the
    #     virtual position where ``last_tok`` sits as *input* (not in cache).
    # ==========================================================================

    # ---- Prefill target on the whole prompt ----
    ids_t = input_ids.to(dev_t)
    pos_prefill = torch.arange(prompt_len, device=dev_t, dtype=torch.long).unsqueeze(0)
    out_t = target_model(
        input_ids=ids_t,
        position_ids=pos_prefill,
        use_cache=True,
        output_attentions=False,
    )
    full_target_kv = _to_legacy(out_t.past_key_values)  # length prompt_len
    first_tok = torch.argmax(out_t.logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1] on dev_t

    # ---- Prefill draft on the prompt (skipped in pre-rotate mode: first CMR
    #      refresh on iter 1 will build the working cache from scratch) ----
    if not pre_rotate:
        ids_d = input_ids.to(dev_d)
        pos_prefill_d = torch.arange(prompt_len, device=dev_d, dtype=torch.long).unsqueeze(0)
        out_d = draft_model(
            input_ids=ids_d,
            position_ids=pos_prefill_d,
            use_cache=True,
            output_attentions=False,
        )
        full_draft_kv = _to_legacy(out_d.past_key_values)  # length prompt_len
    else:
        full_draft_kv = None  # never touched in pre-rotate mode

    # Running CPU tensor of committed tokens at real positions [0, last_pos).
    # Needed by the pre_rotate path for re-prefill and harmless otherwise.
    prefix_tokens = input_ids.detach().to("cpu")  # [1, prompt_len]

    # Keep the running generated buffer on CPU to avoid device mismatch on final cat.
    generated = [first_tok.detach().cpu()]
    accept_lens: List[int] = []
    last_pos = prompt_len  # real position where last_tok logically sits
    last_tok = first_tok
    new_tok_count = 1  # first_tok is our first generated token
    iters = 0

    # ---- CMR state ----
    # non-pre_rotate path:
    #   cached_selected_positions  : 1D long tensor of prefix positions (CPU)
    # pre_rotate path:
    #   working_draft_kv           : KV at virtual positions [0, kv_len(working_draft_kv))
    #   virtual_base               : kv_len(working_draft_kv) right after a refresh
    #                                (= number of tokens re-prefilled)
    #   real_refresh_pos           : value of last_pos at the most recent refresh
    cached_selected_positions: Optional[torch.Tensor] = None
    working_draft_kv: Optional[KVCache] = None
    virtual_base: int = 0
    refresh_last_pos: int = 0  # value of last_pos at the most recent refresh
    real_refresh_pos: int = 0  # same, but used by the pre_rotate code path
    iters_since_refresh = math.inf  # force refresh on first iter if CMR is on
    start_time = time.time()

    while new_tok_count < max_new_tokens:
        iters += 1

        # --- CMR: decide draft's working KV and where ``last_tok`` sits ---
        if use_cmr and not pre_rotate:
            # Legacy index-select path. full_draft_kv covers [0, last_pos).
            need_refresh = iters_since_refresh >= refresh_every or cached_selected_positions is None
            if need_refresh and last_pos > chunk_size:
                attn_over_prefix = get_last_query_attention(
                    target_model=target_model,
                    full_target_kv=full_target_kv,
                    last_token_id=last_tok,
                    prefix_len=last_pos,
                )
                chunks = build_chunks(last_pos, chunk_size)
                chunk_scores = score_chunks(attn_over_prefix, chunks)
                selected = select_top_k_chunks(
                    chunk_scores, chunks, top_k_chunks, always_keep_tail=recent_tail_chunks
                )
                sel_prefix = chunks_to_positions(selected, device=torch.device("cpu"))
                cached_selected_positions = sel_prefix
                refresh_last_pos = last_pos
                iters_since_refresh = 0
                if verbose:
                    print(
                        f"  [CMR refresh | index-select] prefix={last_pos}, "
                        f"kept {len(sel_prefix)}/{last_pos} positions "
                        f"({len(selected)}/{len(chunks)} chunks)"
                    )

            if cached_selected_positions is not None:
                if last_pos > refresh_last_pos:
                    tail = torch.arange(refresh_last_pos, last_pos, dtype=torch.long)
                    selection = torch.cat([cached_selected_positions, tail])
                else:
                    selection = cached_selected_positions
                if selection.numel() < kv_len(full_draft_kv):
                    working_draft_kv = kv_index_select(full_draft_kv, selection)
                else:
                    working_draft_kv = full_draft_kv
            else:
                working_draft_kv = full_draft_kv
            draft_start_position = last_pos

        elif pre_rotate:
            # Re-prefill path. working_draft_kv lives in virtual position space.
            need_refresh = (
                iters_since_refresh >= refresh_every
                or working_draft_kv is None
            )
            if need_refresh:
                if last_pos > chunk_size:
                    attn_over_prefix = get_last_query_attention(
                        target_model=target_model,
                        full_target_kv=full_target_kv,
                        last_token_id=last_tok,
                        prefix_len=last_pos,
                    )
                    chunks = build_chunks(last_pos, chunk_size)
                    chunk_scores = score_chunks(attn_over_prefix, chunks)
                    selected = select_top_k_chunks(
                        chunk_scores, chunks, top_k_chunks, always_keep_tail=recent_tail_chunks
                    )
                    sel_positions = chunks_to_positions(selected, device=torch.device("cpu"))
                else:
                    # Prefix too short to chunk meaningfully: just keep everything.
                    sel_positions = torch.arange(last_pos, dtype=torch.long)
                working_draft_kv = prefill_draft_on_positions(
                    draft_model=draft_model,
                    prefix_tokens=prefix_tokens,
                    positions=sel_positions,
                )
                virtual_base = kv_len(working_draft_kv)  # == sel_positions.numel()
                real_refresh_pos = last_pos
                iters_since_refresh = 0
                if verbose:
                    print(
                        f"  [CMR refresh | pre-rotate] prefix={last_pos}, "
                        f"re-prefilled {virtual_base} positions with contiguous RoPE"
                    )
            draft_start_position = virtual_base + (last_pos - real_refresh_pos)

        else:
            working_draft_kv = full_draft_kv
            draft_start_position = last_pos

        # --- Draft generates N candidate tokens ---
        draft_tokens, _new_draft_kv_unused = draft_generate_N(
            draft_model=draft_model,
            last_token_id=last_tok,
            working_past_kv=working_draft_kv,
            num_draft=num_draft,
            start_position=draft_start_position,
        )

        # --- Target verification (feeds [last_tok, *drafts], N+1 logits) ---
        accept_count, corrected_tok, full_target_kv = target_verify(
            target_model=target_model,
            full_target_kv=full_target_kv,
            last_accepted_tok=last_tok,
            draft_tokens=draft_tokens,
            prefix_len=last_pos,
        )
        accept_lens.append(accept_count)

        # --- Commit the new tokens to the draft cache ---
        # commit_tokens = [last_tok, *accepted_drafts], i.e. 1 + accept_count tokens.
        # After commit the draft cache grows by exactly this many slots, and the
        # new ``last_tok`` becomes corrected_tok (NOT in the cache).
        commit_parts = [last_tok.to(dev_d)]
        if accept_count > 0:
            commit_parts.append(draft_tokens[:, :accept_count].to(dev_d))
        commit_tokens = torch.cat(commit_parts, dim=1)  # [1, 1 + accept_count]

        if pre_rotate:
            commit_start_virtual = virtual_base + (last_pos - real_refresh_pos)
            commit_positions = torch.arange(
                commit_start_virtual,
                commit_start_virtual + commit_tokens.shape[1],
                device=dev_d, dtype=torch.long,
            ).unsqueeze(0)
            out_draft_commit = draft_model(
                input_ids=commit_tokens,
                past_key_values=working_draft_kv,
                position_ids=commit_positions,
                use_cache=True,
                output_attentions=False,
            )
            working_draft_kv = _to_legacy(out_draft_commit.past_key_values)
        else:
            commit_positions = torch.arange(
                last_pos, last_pos + commit_tokens.shape[1],
                device=dev_d, dtype=torch.long,
            ).unsqueeze(0)
            out_draft_commit = draft_model(
                input_ids=commit_tokens,
                past_key_values=full_draft_kv,
                position_ids=commit_positions,
                use_cache=True,
                output_attentions=False,
            )
            full_draft_kv = _to_legacy(out_draft_commit.past_key_values)

        # --- Record newly generated tokens and advance state ---
        newly_generated_parts = []
        if accept_count > 0:
            newly_generated_parts.append(draft_tokens[:, :accept_count].detach().cpu())
        newly_generated_parts.append(corrected_tok.detach().cpu())
        newly_generated = torch.cat(newly_generated_parts, dim=1)
        generated.append(newly_generated)
        new_tok_count += newly_generated.shape[1]

        # Extend prefix_tokens with the just-committed tokens (NOT corrected_tok).
        prefix_tokens = torch.cat([prefix_tokens, commit_tokens.detach().cpu()], dim=1)

        last_pos = last_pos + accept_count + 1
        last_tok = corrected_tok
        iters_since_refresh += 1

        if eos_token_id is not None and (newly_generated == eos_token_id).any().item():
            break

    wall = time.time() - start_time
    all_gen = torch.cat(generated, dim=1)  # [1, total_new]
    return SpecResult(
        generated_ids=all_gen,
        accept_lens=accept_lens,
        num_iters=iters,
        num_new_tokens=all_gen.shape[1],
        wall_time=wall,
    )


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def load_texts(path: str, max_samples: Optional[int]) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("text")
            if t is None:
                continue
            texts.append(t)
    return texts


def summarize(res: SpecResult) -> dict:
    acc = res.accept_lens
    return {
        "num_iters": res.num_iters,
        "num_new_tokens": res.num_new_tokens,
        "avg_accept_length": (sum(acc) / len(acc)) if acc else 0.0,
        "total_accept": sum(acc),
        "wall_time_s": round(res.wall_time, 3),
        "tokens_per_sec": round(res.num_new_tokens / res.wall_time, 3) if res.wall_time > 0 else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description="Tier-1 CMR prototype (HF transformers, vanilla spec decoding).")
    p.add_argument("--target_model", required=True, help="HF id or local path for the target model")
    p.add_argument("--draft_model", required=True, help="HF id or local path for the draft model")
    p.add_argument("--data", required=True, help="JSONL file with {'text': ...} per line")
    p.add_argument("--max_samples", type=int, default=5)
    p.add_argument("--max_prompt_tokens", type=int, default=4096,
                   help="Truncate each prompt to at most this many tokens (by head)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_draft", type=int, default=5, help="Draft tokens per speculative iter")
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--top_k_chunks", type=int, default=64)
    p.add_argument("--refresh_every", type=int, default=4)
    p.add_argument("--recent_tail_chunks", type=int, default=2,
                   help="Number of trailing chunks to always force-include in the working cache.")
    p.add_argument("--compare_mode", action="store_true",
                   help="Run both with and without CMR and report both.")
    p.add_argument("--cache_pre_rotate", action="store_true",
                   help=("On each CMR refresh, re-prefill the draft on the selected "
                         "tokens with contiguous position IDs so RoPE is applied "
                         "in-distribution instead of carrying the original "
                         "(potentially far-OOD) rotations of retained keys. "
                         "Only meaningful together with CMR; ignored when CMR is off."))
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--output_file", default="cmr_prototype_results.json")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[load] target: {args.target_model}")
    target_tok = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",  # REQUIRED for output_attentions
        low_cpu_mem_usage=True,
    ).eval()

    print(f"[load] draft:  {args.draft_model}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    # Assume shared tokenizer (same model family). Fail loudly otherwise.
    draft_tok = AutoTokenizer.from_pretrained(args.draft_model)
    if draft_tok.vocab_size != target_tok.vocab_size:
        raise ValueError(
            f"target vocab={target_tok.vocab_size} != draft vocab={draft_tok.vocab_size}. "
            "This prototype assumes the draft and target share a tokenizer."
        )

    eos_id = target_tok.eos_token_id
    # input_ids will be moved to each model's input-embedding device inside spec_generate;
    # we just need to move it off CPU here so tokenization doesn't blow the main thread.
    device = _in_dev(target_model)

    texts = load_texts(args.data, args.max_samples)
    print(f"[data] loaded {len(texts)} prompts")

    modes = [("cmr_off", False)]
    if args.compare_mode:
        modes.append(("cmr_on", True))

    per_sample: List[dict] = []
    agg = {name: {"avg_accept_length": [], "tokens_per_sec": [], "prompt_len": [], "num_new_tokens": []}
           for (name, _) in modes}

    for si, text in enumerate(texts):
        enc = target_tok(
            text, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=args.max_prompt_tokens,
        )
        input_ids = enc.input_ids.to(device)
        prompt_len = input_ids.shape[1]
        print(f"\n[sample {si+1}/{len(texts)}] prompt_len={prompt_len}")

        sample_record = {"idx": si, "prompt_len": prompt_len, "modes": {}}
        for (name, use_cmr) in modes:
            print(f"  [run] {name}")
            res = spec_generate(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tok,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                num_draft=args.num_draft,
                eos_token_id=eos_id,
                use_cmr=use_cmr,
                cache_pre_rotate=args.cache_pre_rotate,
                chunk_size=args.chunk_size,
                top_k_chunks=args.top_k_chunks,
                refresh_every=args.refresh_every,
                recent_tail_chunks=args.recent_tail_chunks,
                verbose=args.verbose,
            )
            summary = summarize(res)
            sample_record["modes"][name] = summary
            print(f"    avg_accept_length={summary['avg_accept_length']:.3f}  "
                  f"tok/s={summary['tokens_per_sec']}  gen={summary['num_new_tokens']}")
            agg[name]["avg_accept_length"].append(summary["avg_accept_length"])
            agg[name]["tokens_per_sec"].append(summary["tokens_per_sec"])
            agg[name]["prompt_len"].append(prompt_len)
            agg[name]["num_new_tokens"].append(summary["num_new_tokens"])
        per_sample.append(sample_record)

    def _mean(xs): return (sum(xs) / len(xs)) if xs else 0.0

    aggregate = {
        name: {
            "mean_avg_accept_length": round(_mean(v["avg_accept_length"]), 4),
            "mean_tokens_per_sec":    round(_mean(v["tokens_per_sec"]), 3),
            "num_samples":            len(v["avg_accept_length"]),
        }
        for name, v in agg.items()
    }

    out = {
        "config": vars(args),
        "per_sample": per_sample,
        "aggregate": aggregate,
    }
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== AGGREGATE ===")
    print(json.dumps(aggregate, indent=2))
    print(f"\n[write] results -> {args.output_file}")


if __name__ == "__main__":
    main()
