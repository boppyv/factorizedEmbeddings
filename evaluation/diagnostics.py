# evaluation/diagnostics.py
import torch
import numpy as np

def effective_rank(embedding_matrix):
    """
    Compute the effective rank of an embedding matrix via SVD.
    Effective rank = exp(entropy of normalized singular values).

    A matrix with all variance in K directions has effective rank ≈ K.
    A matrix with variance spread across all directions has effective rank ≈ min(V, D).

    Args:
        embedding_matrix: tensor of shape (vocab_size, embed_dim)

    Returns:
        float: the effective rank
    """
    # SVD can be expensive for large matrices — use a random subset of rows if needed
    if embedding_matrix.shape[0] > 10000:
        indices = torch.randperm(embedding_matrix.shape[0])[:10000]
        matrix = embedding_matrix[indices]
    else:
        matrix = embedding_matrix

    # Compute singular values
    S = torch.linalg.svdvals(matrix.float())

    # Normalize to form a probability distribution
    S_normalized = S / S.sum()

    # Remove zeros to avoid log(0)
    S_normalized = S_normalized[S_normalized > 1e-10]

    # Entropy
    entropy = -(S_normalized * torch.log(S_normalized)).sum()

    return torch.exp(entropy).item()


def per_bucket_gradient_norms(embedding_weight, bucket_assignment_tensor, num_buckets):
    """
    Compute average gradient L2 norm per frequency bucket.
    Call this AFTER loss.backward() but BEFORE optimizer.step().

    Args:
        embedding_weight: the nn.Embedding weight parameter (has .grad after backward)
        bucket_assignment_tensor: torch.LongTensor of shape (vocab_size,)
        num_buckets: number of frequency buckets

    Returns:
        list of average gradient norms per bucket
    """
    if embedding_weight.grad is None:
        return [0.0] * num_buckets

    grad_norms = embedding_weight.grad.norm(dim=1)  # shape: (vocab_size,)

    bucket_grad_norms = []
    for b in range(num_buckets):
        mask = (bucket_assignment_tensor == b)
        if mask.sum() > 0:
            bucket_grad_norms.append(grad_norms[mask].mean().item())
        else:
            bucket_grad_norms.append(0.0)

    return bucket_grad_norms


def pairwise_cosine_stats(embedding_matrix, num_samples=10000):
    """
    Compute mean and std of pairwise cosine similarities between random embedding pairs.

    Args:
        embedding_matrix: tensor of shape (vocab_size, embed_dim)
        num_samples: number of random pairs to sample

    Returns:
        dict with 'mean' and 'std'
    """
    V = embedding_matrix.shape[0]
    idx_a = torch.randint(0, V, (num_samples,))
    idx_b = torch.randint(0, V, (num_samples,))

    vecs_a = embedding_matrix[idx_a]
    vecs_b = embedding_matrix[idx_b]

    cosines = F.cosine_similarity(vecs_a, vecs_b, dim=1)

    return {
        "mean": cosines.mean().item(),
        "std": cosines.std().item(),
    }
