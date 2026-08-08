"""
Token sampling — converts logits to next token ID.

Supports:
    - Greedy (argmax)
    - Temperature scaling
    - Top-k filtering
    - Top-p (nucleus) filtering
"""

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample the next token from logits.

    Args:
        logits: [vocab_size] logits for the last position
        temperature: scaling factor (0 = greedy, >0 = sampling)
        top_k: if > 0, keep only top-k logits
        top_p: if < 1.0, use nucleus sampling

    Returns:
        Token ID (int)
    """
    # Greedy
    if temperature <= 0 or temperature < 1e-8:
        return logits.argmax(dim=-1).item()

    # Temperature scaling
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[..., -1]
        logits = torch.where(logits < kth_val, torch.full_like(logits, float("-inf")), logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (always keep top-1)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_mask[..., 0] = False
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original ordering
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
