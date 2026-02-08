# embeddings/__init__.py
from embeddings.standard import StandardEmbedding
from embeddings.factorized import FactorizedEmbedding
from embeddings.generator import GeneratorEmbedding
from embeddings.progressive import ProgressiveEmbedding
from embeddings.growing import GrowingFactorizedEmbedding
from embeddings.adaptive_lr import AdaptiveLREmbedding

def create_embedding(config, vocab_size, embed_dim):
    emb_type = config.embedding_type if hasattr(config, 'embedding_type') else 'standard'
    K = config.embedding_K if hasattr(config, 'embedding_K') else 64

    if emb_type == 'standard':
        return StandardEmbedding(vocab_size, embed_dim)
    elif emb_type == 'factorized':
        return FactorizedEmbedding(vocab_size, embed_dim, K)
    elif emb_type == 'generator':
        return GeneratorEmbedding(vocab_size, embed_dim, K)
    elif emb_type == 'progressive':
        # these will need ramp_start and ramp_end from config later
        ramp_start = getattr(config, 'ramp_start', 0)
        ramp_end = getattr(config, 'ramp_end', 1)
        return ProgressiveEmbedding(vocab_size, embed_dim, K, ramp_start, ramp_end)
    elif emb_type == 'growing':
        initial_K = getattr(config, 'initial_K', 16)
        max_K = getattr(config, 'max_K', 64)
        return GrowingFactorizedEmbedding(vocab_size, embed_dim, initial_K, max_K)
    else:
        raise ValueError(f"Unknown embedding type: {emb_type}")
