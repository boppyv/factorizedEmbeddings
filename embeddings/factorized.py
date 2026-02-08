# embeddings/factorized.py
import torch
import torch.nn as nn

class FactorizedEmbedding(nn.Module):
    """
    E(i) = A[i] @ B

    A is (vocab_size × K): per-token mixing weights in the low-dimensional space.
    B is (K × embed_dim): shared base vectors projected to full dimension.

    Every token's embedding is a linear combination of K base vectors.
    """

    def __init__(self, vocab_size, embed_dim, K):
        super().__init__()
        self.A = nn.Embedding(vocab_size, K)
        self.B = nn.Linear(K, embed_dim, bias=False)
        self.K = K
        self.embed_dim = embed_dim

        # Initialize A with small random values (standard embedding init)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        # Initialize B with Xavier uniform (good for linear projections)
        nn.init.xavier_uniform_(self.B.weight)

    @property
    def weight(self):
        """
        Compute and return the full V×D matrix.
        Needed for weight tying with lm_head.
        NOTE: This recomputes the product each time. For weight tying,
        you may need a different approach — see the note in Phase 4.
        """
        return self.A.weight @ self.B.weight.T

    def forward(self, token_ids, step=None):
        low_dim = self.A(token_ids)      # (batch, seq_len, K)
        full_dim = self.B(low_dim)       # (batch, seq_len, embed_dim)
        return full_dim

    def get_embedding_matrix(self):
        with torch.no_grad():
            return (self.A.weight @ self.B.weight.T).data
