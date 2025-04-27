import torch
from torch import Tensor


def find_independent_tensors(tensors: list[Tensor], tolerance=1e-4):
    """Find linearly independent tensors using QR decomposition.

    Args:
        tensors: list of tensors
        tolerance: tolerance for checking diagonal elements is non-zero
    """
    vectors = [t.flatten() for t in tensors]
    matrix = torch.vstack(vectors)
    Q, R = torch.linalg.qr(matrix.T, mode="complete")

    # Check all diagonal elements
    independent_indices = []
    for i in range(len(vectors)):
        # TODO, double check that it is OK to only check diagonal
        if torch.abs(R[i, i]) > tolerance:
            independent_indices.append(i)

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices
