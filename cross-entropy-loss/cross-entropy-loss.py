import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    l = [-np.log(y_pred[i][yt]) for i, yt in enumerate(y_true) ]
    return np.mean(l)