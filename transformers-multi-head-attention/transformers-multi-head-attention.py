import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    B, S, D = Q.shape
    dh = D // num_heads
    Q = Q.reshape(B, S, num_heads, dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, dh).transpose(0, 2, 1, 3)

    qk = Q @ K.transpose(0, 1, 3, 2)
    qkv = softmax(qk/(D**0.5)) @ V
    qkv_reshape = qkv.transpose(0, 2, 1, 3).reshape(B, S, num_heads*dh)
    out = qkv_reshape @ W_o
    return out