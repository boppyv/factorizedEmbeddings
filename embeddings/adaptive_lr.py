# embeddings/adaptive_lr.py
import torch
import torch.nn as nn
import numpy as np

class AdaptiveLREmbedding(nn.Module):
    """
    Standard embedding with a gradient scaling hook that gives rare tokens
    larger effective learning rates.

    This is the "boring baseline" that tests whether the improvement from
    architectural sharing is just an optimization effect.
    """

    def __init__(self, vocab_size, embed_dim, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.alpha = config["embedding"].get("adaptive_lr_alpha", 0.5)
        self.max_scale = config["embedding"].get("adaptive_lr_max_scale", 10.0)

        # Gradient scale will be set later once we have frequency data
        self.register_buffer('grad_scale', torch.ones(vocab_size, 1))

    def set_frequency_data(self, token_frequencies):
        """
        Call this once after loading frequency data.
        Computes per-token gradient scaling factors.
        """
        freqs = torch.tensor(token_frequencies, dtype=torch.float32)
        freqs = freqs.clamp(min=1)  # avoid division by zero
        freq_median = freqs[freqs > 0].median()

        scale = (freq_median / freqs) ** self.alpha
        scale = scale.clamp(max=self.max_scale)

        self.grad_scale = scale.unsqueeze(1)  # shape: (V, 1) for broadcasting

        # Register the hook
        self.embedding.weight.register_hook(self._scale_grad)

    def _scale_grad(self, grad):
        return grad * self.grad_scale.to(grad.device)

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, token_ids, step=None):
        return self.embedding(token_ids)

    def get_embedding_matrix(self):
        return self.embedding.weight.data
