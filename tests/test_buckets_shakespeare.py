"""
Test frequency bucketing on Shakespeare character-level data.

Run from the repo root:
    python tests/test_buckets_shakespeare.py

Assumes you've already run:
    cd data/shakespeare_char && python prepare.py
"""

import numpy as np
import os
import sys
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# Add repo root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.frequency import build_target_frequency_map, assign_buckets_percentile


def main():
    data_path = os.path.join("data", "shakespeare", "train.bin")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        print("Run this first:  cd data/shakespeare && python prepare.py")
        sys.exit(1)

    # Shakespeare char-level uses a small vocabulary.
    # Load the data to figure out the vocab size.
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    vocab_size = int(data.max()) + 1
    total_tokens = len(data)

    print("=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    print(f"  File:         {data_path}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Vocab size:   {vocab_size}")
    print()

    # Build target frequency map
    # "Targets" = every token except the first (the next-token prediction targets)
    counts = build_target_frequency_map(data_path, vocab_size)
    total_targets = counts.sum()
    active_tokens = (counts > 0).sum()

    print("TARGET FREQUENCY SUMMARY")
    print("-" * 60)
    print(f"  Total target tokens:  {total_targets:,}")
    print(f"  Active token types:   {active_tokens} / {vocab_size}")
    print(f"  Inactive (count=0):   {vocab_size - active_tokens}")
    print(f"  Most common token:    ID {counts.argmax()} (count: {counts.max():,})")
    print(f"  Least common (>0):    count = {counts[counts > 0].min():,}")
    print()

    # Assign buckets
    num_buckets = 5
    assignments, stats = assign_buckets_percentile(counts, num_buckets=num_buckets)

    print(f"PERCENTILE BUCKETS ({num_buckets} buckets)")
    print("-" * 60)
    print(f"  {'Bucket':<8} {'Types':<8} {'Occurrences':<15} {'% of Targets':<14} {'Freq Range':<20} {'Label'}")
    print(f"  {'------':<8} {'-----':<8} {'-----------':<15} {'----------':<14} {'----------':<20} {'-----'}")

    labels = ["Rarest", "Rare", "Medium", "Common", "Most Common"]

    for b in range(num_buckets):
        s = stats[b]
        pct = (s['total_occurrences'] / total_targets * 100) if total_targets > 0 else 0
        label = labels[b] if b < len(labels) else f"Bucket {b}"
        print(f"  {b:<8} {s['num_types']:<8} {s['total_occurrences']:<15,} {pct:<14.4f} {s['freq_range']:<20} {label}")

    print()

    # Show which actual characters are in each bucket
    # shakespeare_char encodes characters directly, so we can decode them
    meta_path = os.path.join("data", "shakespeare_char", "meta.pkl")
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        itos = meta.get("itos", {})

        print("TOKENS PER BUCKET")
        print("-" * 60)
        for b in range(num_buckets):
            token_ids = np.where(assignments == b)[0]
            # Sort by frequency within bucket
            bucket_freqs = counts[token_ids]
            sorted_order = np.argsort(bucket_freqs)
            token_ids_sorted = token_ids[sorted_order]

            chars = []
            for tid in token_ids_sorted:
                char = itos.get(tid, f"[{tid}]")
                token = enc.decode([tid])
                freq = counts[tid]
                # Make whitespace visible
                display = repr(token) if token in (' ', '\n', '\t', '\r') else token
                chars.append(f"{display}({freq:,})")

            label = labels[b] if b < len(labels) else f"Bucket {b}"
            print(f"\n  Bucket {b} ({label}):")
            # Print in rows of ~8 for readability
            for i in range(0, len(chars), 8):
                row = "    ".join(chars[i:i+8])
                print(f"    {row}")
    else:
        print("  (meta.pkl not found — can't decode token IDs to characters)")

    print()
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check: every active token should be assigned to exactly one bucket
    assigned = assignments[assignments >= 0]
    unassigned_active = ((counts > 0) & (assignments < 0)).sum()
    print(f"  Active tokens with no bucket:  {unassigned_active}  {'✓' if unassigned_active == 0 else '✗ BUG'}")

    # Check: bucket assignments cover all active tokens
    total_types_in_buckets = sum(s['num_types'] for s in stats.values())
    print(f"  Types in buckets sum to active: {total_types_in_buckets} == {active_tokens}  {'✓' if total_types_in_buckets == active_tokens else '✗ BUG'}")

    # Check: occurrences sum to total targets
    total_occ_in_buckets = sum(s['total_occurrences'] for s in stats.values())
    print(f"  Occurrences sum to total:      {total_occ_in_buckets:,} == {total_targets:,}  {'✓' if total_occ_in_buckets == total_targets else '✗ BUG'}")

    # Check: buckets are ordered by frequency (min of bucket N <= min of bucket N+1)
    ordered = all(stats[b]['min_freq'] <= stats[b+1]['min_freq'] for b in range(num_buckets - 1))
    print(f"  Buckets ordered by frequency:  {'✓' if ordered else '✗ BUG'}")

    print()


if __name__ == "__main__":
    main()
