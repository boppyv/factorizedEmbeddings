# embeddings/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorEmbedding(nn.Module):
    """
    E(i) = C(GELU(W @ A[i]))

    A[i] is a per-token seed vector in R^K.
    W is a shared K×K linear transform.
    GELU is a nonlinear activation.
    C is a shared K→D projection.

    Unlike factorized, this can learn nonlinear relationships between tokens.
    """

    def __init__(self, vocab_size, embed_dim, K):
        super().__init__()
        self.A = nn.Embedding(vocab_size, K)
        self.W = nn.Linear(K, K, bias=False)
        self.C = nn.Linear(K, embed_dim, bias=False)
        self.K = K
        self.embed_dim = embed_dim

        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.C.weight)

    @property
    def weight(self):
        """Full V×D matrix for weight tying (approximate — includes nonlinearity)."""
        with torch.no_grad():
            seeds = self.A.weight                       # (V, K)
            hidden = F.gelu(self.W(seeds))              # (V, K)
            return self.C(hidden)                       # (V, D)

    def forward(self, token_ids, step=None):
        seeds = self.A(token_ids)                       # (batch, seq, K)
        hidden = F.gelu(self.W(seeds))                  # (batch, seq, K)
        return self.C(hidden)                           # (batch, seq, D)

    def get_embedding_matrix(self):
        return self.weight.data
