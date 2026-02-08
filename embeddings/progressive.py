# embeddings/progressive.py
import torch
import torch.nn as nn
from embeddings.generator import GeneratorEmbedding

class ProgressiveEmbedding(nn.Module):
    """
    E(i, t) = E_gen(i) + lambda(t) * E_res(i)

    Early in training: lambda=0, embedding is purely from the shared generator.
    Late in training: lambda=1, token-specific residual fully active.
    """

    def __init__(self, vocab_size, embed_dim, K, ramp_start, ramp_end):
        super().__init__()
        self.generator = GeneratorEmbedding(vocab_size, embed_dim, K)
        self.residual = nn.Embedding(vocab_size, embed_dim)
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end

        # IMPORTANT: initialize residual to zero so it has no effect at start
        nn.init.zeros_(self.residual.weight)

    def get_lambda(self, step):
        if step is None or step < self.ramp_start:
            return 0.0
        elif step >= self.ramp_end:
            return 1.0
        else:
            return (step - self.ramp_start) / (self.ramp_end - self.ramp_start)

    @property
    def weight(self):
        # Return generator weight + full residual (lambda=1)
        # This is used for weight tying; at inference time lambda=1
        return self.generator.weight + self.residual.weight

    def forward(self, token_ids, step=None):
        gen = self.generator(token_ids, step)
        res = self.residual(token_ids)
        lam = self.get_lambda(step)
        return gen + lam * res

    def get_embedding_matrix(self):
        return self.weight.data

    def get_residual_fraction(self, step):
        """
        Diagnostic: what fraction of embedding norm comes from residual vs generator?
        Returns float in [0, 1].
        """
        with torch.no_grad():
            gen_norm = self.generator.weight.norm().item()
            res_norm = self.residual.weight.norm().item()
            lam = self.get_lambda(step)
            scaled_res_norm = lam * res_norm
            total = gen_norm + scaled_res_norm
            if total == 0:
                return 0.0
            return scaled_res_norm / total
