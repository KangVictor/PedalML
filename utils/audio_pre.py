import torch.nn.functional as F

# Pre-emphasis filter function
# Default pre_emphasis_coef = 0.95 As used in https://arxiv.org/pdf/1811.00334 and other previous works
def pre_emphasis(signal, pre_emphasis_coef = 0.95):
    # signal shape: (batch, samples) or (batch, 1, samples)
    # We will ensure it's 2D (batch, samples) for simplicity
    if signal.dim() == 3:
        # assume shape (batch, 1, L)
        signal = signal[:, 0, :]
    # Apply y[n] = x[n] - alpha * x[n-1]
    # Pad the beginning with zero for alignment (y[0] = x[0])
    y = signal.clone()
    y[:, 1:] = signal[:, 1:] - pre_emphasis_coef * signal[:, :-1]
    # y[:, 0] is left as signal[:,0] (we could also do y[:,0] = signal[:,0] for clarity)
    return y
    