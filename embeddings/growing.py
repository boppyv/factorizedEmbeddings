# embeddings/growing.py
import torch
import torch.nn as nn

class GrowingFactorizedEmbedding(nn.Module):
    """
    Like FactorizedEmbedding, but K can be increased during training.
    New base vectors are added with zero mixing weights (so output is unchanged at growth time).
    """

    def __init__(self, vocab_size, embed_dim, initial_K, max_K):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.current_K = initial_K
        self.max_K = max_K

        self.A = nn.Embedding(vocab_size, initial_K)
        self.B = nn.Linear(initial_K, embed_dim, bias=False)

        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.B.weight)

    def grow(self, new_K):
        """
        Increase bottleneck dimension from current_K to new_K.
        Preserves existing parameters. New parameters initialized for zero contribution.
        Returns True if growth happened, False if already at max or new_K <= current_K.
        """
        if new_K <= self.current_K or new_K > self.max_K:
            return False

        added_K = new_K - self.current_K

        # Expand A: add new columns initialized to zero
        old_A_weight = self.A.weight.data  # (V, current_K)
        new_A = nn.Embedding(self.vocab_size, new_K).to(old_A_weight.device)
        new_A.weight.data[:, :self.current_K] = old_A_weight
        new_A.weight.data[:, self.current_K:] = 0.0  # zero → no initial contribution
        self.A = new_A

        # Expand B: add new rows initialized with small random values
        old_B_weight = self.B.weight.data  # (embed_dim, current_K) — note Linear stores (out, in)
        new_B = nn.Linear(new_K, self.embed_dim, bias=False).to(old_B_weight.device)
        new_B.weight.data[:, :self.current_K] = old_B_weight
        nn.init.xavier_uniform_(new_B.weight.data[:, self.current_K:].unsqueeze(0))
        self.B = new_B

        self.current_K = new_K
        return True

    @property
    def weight(self):
        return self.A.weight @ self.B.weight.T

    def forward(self, token_ids, step=None):
        low_dim = self.A(token_ids)
        full_dim = self.B(low_dim)
        return full_dim

    def get_embedding_matrix(self):
        with torch.no_grad():
            return (self.A.weight @ self.B.weight.T).data
