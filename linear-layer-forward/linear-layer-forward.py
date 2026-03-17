import numpy as np
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    a = ((np.array(X) @ np.array(W))+np.array(b)).tolist()
    return a