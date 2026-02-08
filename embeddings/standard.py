# embeddings/standard.py
import torch.nn as nn

class StandardEmbedding(nn.Module):
    """Thin wrapper around nn.Embedding for interface consistency."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    @property
    def weight(self):
        """Expose weight for weight tying with lm_head."""
        return self.embedding.weight

    def forward(self, token_ids, step=None):
        return self.embedding(token_ids)

    def get_embedding_matrix(self):
        """Return the full V×D matrix for diagnostics."""
        return self.embedding.weight.data
