# tests/test_embeddings.py
import torch
from embeddings.standard import StandardEmbedding
from embeddings.factorized import FactorizedEmbedding
from embeddings.generator import GeneratorEmbedding
from embeddings.progressive import ProgressiveEmbedding
from embeddings.growing import GrowingFactorizedEmbedding

V, D, K = 100, 32, 8  # tiny sizes for testing
BATCH, SEQ = 4, 16

def test_shapes():
    """All variants should produce (batch, seq_len, embed_dim) outputs."""
    token_ids = torch.randint(0, V, (BATCH, SEQ))

    for name, emb in [
        ("standard", StandardEmbedding(V, D)),
        ("factorized", FactorizedEmbedding(V, D, K)),
        ("generator", GeneratorEmbedding(V, D, K)),
        ("progressive", ProgressiveEmbedding(V, D, K, ramp_start=100, ramp_end=200)),
    ]:
        out = emb(token_ids, step=150)
        assert out.shape == (BATCH, SEQ, D), f"{name}: expected {(BATCH, SEQ, D)}, got {out.shape}"
        print(f"  {name}: shape OK")

def test_progressive_ramp():
    """Progressive embedding should transition from generator-only to generator+residual."""
    emb = ProgressiveEmbedding(V, D, K, ramp_start=100, ramp_end=200)
    token_ids = torch.randint(0, V, (BATCH, SEQ))

    # At step 0: lambda=0, so output should equal generator output
    out_early = emb(token_ids, step=0)
    gen_out = emb.generator(token_ids, step=0)
    assert torch.allclose(out_early, gen_out, atol=1e-6), "At step 0, output should equal generator"

    # At step 200: lambda=1, so output should equal generator + residual
    out_late = emb(token_ids, step=200)
    expected = emb.generator(token_ids) + emb.residual(token_ids)
    assert torch.allclose(out_late, expected, atol=1e-6), "At step 200, output should equal gen + res"

    print("  progressive ramp: OK")

def test_growing_preserves_output():
    """Growing K should not change output when new weights are zero."""
    emb = GrowingFactorizedEmbedding(V, D, initial_K=8, max_K=64)
    token_ids = torch.randint(0, V, (BATCH, SEQ))

    out_before = emb(token_ids).clone()
    grew = emb.grow(16)
    out_after = emb(token_ids)

    assert grew == True, "Growth should have happened"
    assert emb.current_K == 16, f"Expected K=16, got {emb.current_K}"
    assert torch.allclose(out_before, out_after, atol=1e-6), "Output changed after growth!"

    print("  growing K preservation: OK")

def test_gradients_flow():
    """Verify gradients reach all parameters in each variant."""
    for name, emb in [
        ("factorized", FactorizedEmbedding(V, D, K)),
        ("generator", GeneratorEmbedding(V, D, K)),
        ("progressive", ProgressiveEmbedding(V, D, K, ramp_start=0, ramp_end=100)),
    ]:
        token_ids = torch.randint(0, V, (BATCH, SEQ))
        out = emb(token_ids, step=50)
        loss = out.sum()
        loss.backward()

        for pname, param in emb.named_parameters():
            assert param.grad is not None, f"{name}.{pname}: no gradient!"
            assert param.grad.abs().sum() > 0, f"{name}.{pname}: gradient is all zeros!"

        print(f"  {name}: gradients OK")

if __name__ == "__main__":
    print("Testing shapes...")
    test_shapes()
    print("Testing progressive ramp...")
    test_progressive_ramp()
    print("Testing growing K...")
    test_growing_preserves_output()
    print("Testing gradients...")
    test_gradients_flow()
    print("\nAll tests passed!")
