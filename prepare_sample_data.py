#!/usr/bin/env python3
"""Generate small synthetic long-context prompts for sanity-testing cmr_prototype.py.

Produces:
  - sample_data/needle.jsonl : needle-in-a-haystack style (good for diagnostic CMR)
  - sample_data/repeat.jsonl : natural continuation prompts from repeated passages

The "text" field is what cmr_prototype.py will feed to the tokenizer. These are
intentionally synthetic and small so the prototype can be exercised in seconds
without downloading benchmark datasets.
"""
import argparse
import json
import os
import random


LOREM = (
    "Speculative decoding accelerates large language model inference by "
    "running a lightweight draft model to propose multiple tokens per step, "
    "which the target model then verifies in parallel. Recent work including "
    "EAGLE and its successors show that a single lightweight transformer "
    "layer conditioned on the target model's hidden states can achieve "
    "substantial acceptance lengths. At long context, however, the draft "
    "model tends to slow down because it must process the entire prefix. "
    "SpecExtend proposes Cross-Model Retrieval (CMR) to address this: it "
    "treats the target model's attention scores as a sparse retriever and "
    "selects only the most relevant input chunks for the draft to attend "
    "over. This prototype validates the CMR algorithm on small models. "
)


def make_needle_prompt(context_len_tokens_approx: int, seed: int) -> str:
    rng = random.Random(seed)
    num_copies = max(1, context_len_tokens_approx // 50)
    filler = LOREM * num_copies
    words = filler.split()
    # Insert a "needle": a unique fact the model can be asked to continue about.
    needle_idx = rng.randint(num_copies // 4, 3 * num_copies // 4)
    key = f"KEY-{rng.randint(10_000, 99_999)}"
    needle_sentence = f"Remember this important fact: the secret password is {key}. "
    # Insert needle
    words = filler.split()
    insert_position = min(needle_idx * 10, len(words) - 10)
    words.insert(insert_position, needle_sentence.strip())
    filled = " ".join(words)
    # Question prompts the continuation to reference the needle
    prompt = (
        filled
        + "\n\nGiven all of the above, the secret password that the text asks you to "
        "remember is "
    )
    return prompt


def make_repeat_prompt(context_len_tokens_approx: int, seed: int) -> str:
    num_copies = max(1, context_len_tokens_approx // 50)
    text = LOREM * num_copies
    prompt = (
        text
        + "\n\nIn summary, the main topic of the passage above is "
    )
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="sample_data")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--context_tokens", nargs="+", type=int,
                   default=[512, 1024, 2048],
                   help="Approximate context lengths to generate (in words, roughly = tokens).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for style, maker in [("needle", make_needle_prompt), ("repeat", make_repeat_prompt)]:
        for L in args.context_tokens:
            out_path = os.path.join(args.output_dir, f"{style}_{L}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for i in range(args.num_samples):
                    prompt = maker(L, seed=i * 991 + L)
                    f.write(json.dumps({"text": prompt}) + "\n")
            print(f"[write] {out_path}  ({args.num_samples} samples, ~{L} words)")


if __name__ == "__main__":
    main()
