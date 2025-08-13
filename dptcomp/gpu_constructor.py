import cupy as cp

def constructF(amplitude: cp.ndarray, J: int, g: cp.ndarray) -> cp.ndarray:
    """
    Constructs the short-time matrix for transformation using GPU.

    Inputs:
        amplitude (cp.ndarray) - The 1D input signal array (on GPU)
        J (int)                - The window size
        g (cp.ndarray)         - The nGaussian window (on GPU)

    Output:
        F (cp.ndarray)         - A (J x N) matrix for short-time transformation
    """
    N = len(amplitude)
    indices = cp.arange(N)
    shifts = cp.arange(J) - J // 2  # Centered window

    # Generate a (J, N) index matrix: each row corresponds to shifted indices
    shifted_indices = indices[None, :] + shifts[:, None]

    # Mask out-of-bounds indices
    mask = (shifted_indices >= 0) & (shifted_indices < N)
    shifted_indices = cp.where(mask, shifted_indices, 0)

    # Gather the shifted values from amplitude
    F = amplitude[shifted_indices] * g[:, None]

    # Zero out the masked regions
    F *= mask

    return F
