# evaluation/bucketed_eval.py
import torch
import torch.nn.functional as F

def bucketed_validation(model, val_dataloader, bucket_assignment_tensor, num_buckets, eval_iters, device):
    """
    Run validation and compute loss per frequency bucket.

    Args:
        model: the GPT model
        val_dataloader: iterator that yields (input_ids, target_ids) batches
        bucket_assignment_tensor: torch.LongTensor of shape (vocab_size,) on device
        num_buckets: number of frequency buckets
        eval_iters: number of batches to evaluate
        device: 'cuda' or 'cpu'

    Returns:
        dict with 'global_loss', 'macro_loss', 'bucket_losses' (list), 'bucket_counts' (list)
    """
    model.eval()

    bucket_loss_sums = torch.zeros(num_buckets, device=device)
    bucket_counts = torch.zeros(num_buckets, device=device)
    global_loss_sum = 0.0
    global_count = 0

    with torch.no_grad():
        for i in range(eval_iters):
            inputs, targets = next(val_dataloader)
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)  # shape: (batch, seq_len, vocab_size)

            # Compute per-token loss (no reduction)
            # Reshape: (batch * seq_len, vocab_size) and (batch * seq_len,)
            B, T, V = logits.shape
            losses = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                reduction='none'
            )  # shape: (B * T,)

            # Global loss
            global_loss_sum += losses.sum().item()
            global_count += losses.numel()

            # Per-bucket loss
            target_buckets = bucket_assignment_tensor[targets.view(-1)]  # shape: (B * T,)

            for b in range(num_buckets):
                mask = (target_buckets == b)
                count = mask.sum()
                if count > 0:
                    bucket_loss_sums[b] += losses[mask].sum()
                    bucket_counts[b] += count

    # Compute averages
    global_loss = global_loss_sum / global_count

    bucket_losses = []
    for b in range(num_buckets):
        if bucket_counts[b] > 0:
            bucket_losses.append((bucket_loss_sums[b] / bucket_counts[b]).item())
        else:
            bucket_losses.append(float('nan'))

    # Macro loss: average of bucket losses (equal weight per bucket)
    valid_bucket_losses = [l for l in bucket_losses if l == l]  # filter NaN
    macro_loss = sum(valid_bucket_losses) / len(valid_bucket_losses) if valid_bucket_losses else float('nan')

    model.train()

    return {
        "global_loss": global_loss,
        "macro_loss": macro_loss,
        "bucket_losses": bucket_losses,
        "bucket_counts": bucket_counts.cpu().tolist(),
    }
