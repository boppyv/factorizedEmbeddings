"""
Token frequency analysis and bucket assignment for frequency-stratified evaluation.
"""

import numpy as np


def build_target_frequency_map(data_path, vocab_size):
    """
    Count how often each token ID appears as a next-token prediction target.

    In language modeling, position t's target is the token at position t+1.
    So for sequence [A, B, C, D], the targets are [B, C, D].

    Args:
        data_path: path to a .bin file (memory-mapped array of uint16 token IDs)
        vocab_size: size of the tokenizer vocabulary

    Returns:
        numpy array of shape (vocab_size,) with target counts per token ID
    """
    data = np.memmap(data_path, dtype=np.uint16, mode='r')

    # Targets are every token except the first
    targets = data[1:]

    counts = np.bincount(targets, minlength=vocab_size)

    return counts


def assign_buckets_percentile(counts, num_buckets=5):
    """
    Assign tokens to buckets based on frequency percentile rank.

    Bucket 0 = rarest token types, bucket num_buckets-1 = most common.
    Tokens with zero count are excluded (assigned bucket = -1).
    Among active tokens, split into equal-sized groups by type count.

    This approach adapts to any dataset size: "rare" is always the bottom
    percentile of token types, regardless of absolute counts.

    Args:
        counts: array of shape (vocab_size,) — output of build_target_frequency_map
        num_buckets: number of buckets to create

    Returns:
        bucket_assignment: array of shape (vocab_size,) with bucket index per token
                          (-1 for tokens that never appear as targets)
        bucket_stats: dict with per-bucket statistics
    """
    vocab_size = len(counts)
    bucket_assignment = np.full(vocab_size, -1, dtype=np.int32)

    # Only bucket tokens that actually appear as targets
    active_mask = counts > 0
    active_ids = np.where(active_mask)[0]
    active_counts = counts[active_ids]

    if len(active_ids) == 0:
        return bucket_assignment, {}

    # Rank by frequency (lowest first)
    sorted_indices = np.argsort(active_counts)
    # Rank by frequency (lowest first)
    sorted_order = np.argsort(active_counts)
    sorted_counts = active_counts[sorted_order]

    # Calculate ideal split points
    split_size = len(sorted_order) / num_buckets

    bucket_idx = 0
    for i, idx in enumerate(sorted_order):
        token_id = active_ids[idx]
        
        # Move to next bucket if we've passed the split point
        # BUT only if the current token has a different frequency than the previous one
        while (bucket_idx < num_buckets - 1 
               and i >= split_size * (bucket_idx + 1)
               and (i == 0 or sorted_counts[i] != sorted_counts[i - 1])):
            bucket_idx += 1
        
        bucket_assignment[token_id] = bucket_idx

    # Compute stats per bucket
    bucket_stats = {}
    for b in range(num_buckets):
        mask = bucket_assignment == b
        if mask.any():
            bucket_counts = counts[mask]
            bucket_stats[b] = {
                "num_types": int(mask.sum()),
                "total_occurrences": int(bucket_counts.sum()),
                "min_freq": int(bucket_counts.min()),
                "max_freq": int(bucket_counts.max()),
                "mean_freq": float(bucket_counts.mean()),
                "freq_range": f"{bucket_counts.min()}-{bucket_counts.max()}",
            }
        else:
            bucket_stats[b] = {
                "num_types": 0,
                "total_occurrences": 0,
                "min_freq": 0,
                "max_freq": 0,
                "mean_freq": 0.0,
                "freq_range": "N/A",
            }

    return bucket_assignment, bucket_stats
