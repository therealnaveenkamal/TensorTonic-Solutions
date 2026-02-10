import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    theta = np.zeros((seq_len, d_model))
    for s in range(seq_len):
        for i in range(0, d_model, 2):
            power = (i)/d_model
            theta[s, i] = np.sin(s/(base**power))
            if(i+1 == d_model):
                break
            theta[s, i+1] = np.cos(s/(base**power))
    return theta