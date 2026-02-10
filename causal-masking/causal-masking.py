import numpy as np
import torch 
def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = torch.from_numpy(scores)
    mask = torch.tril(torch.ones((scores.shape[-1], scores.shape[-1]))).bool()
    masked_scores = scores.masked_fill(~mask, mask_value)
    return masked_scores.numpy()
    # Write code here