import numpy as np
import torch

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    pos = torch.arange(seq_length).unsqueeze(1) # T, 1
    i = torch.arange(0, d_model, 2) # D/2,
    
    
    theta = pos/(10000**(i/d_model)) # (T, 1) / (D/2, ) -> (T,1) x (1, D/2) = (T, D/2)
    
    
    pe = torch.zeros((seq_length, d_model))
    for i in range(theta.shape[-1]):
        pe[:, 2*i] = torch.sin(theta[:, i])
        pe[:, 2*i+1] = torch.cos(theta[:, i])
    return pe.numpy()