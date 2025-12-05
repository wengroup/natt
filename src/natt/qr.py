"""
Find linearly independent tensors using QR decomposition.
"""

from fractions import Fraction

import torch
from torch import Tensor


def find_independent_tensors(
    tensors: list[Tensor], tolerance=1e-4
) -> tuple[list[Tensor], list[int]]:
    """Find linearly independent tensors using QR decomposition.

    Args:
        tensors: list of tensors
        tolerance: tolerance for checking diagonal elements is non-zero

    Returns:
        independent_tensors: list of linearly independent tensors
        independent_indices: indices of the independent tensors in the original list
    """
    vectors = [t.flatten() for t in tensors]
    matrix = torch.vstack(vectors)
    Q, R = torch.linalg.qr(matrix.T, mode="complete")

    # Check all diagonal elements
    independent_indices = []
    for i in range(min(matrix.shape)):
        # TODO, double check that it is OK to only check diagonal
        if torch.abs(R[i, i]) > tolerance:
            independent_indices.append(i)

    independent_tensors = [tensors[i] for i in independent_indices]

    return independent_tensors, independent_indices


# TODO, this is not used, delete
def is_linear_independent(a: list[Fraction], m: list[list[Fraction]]) -> bool:
    """Check whether a vector can be written as a linear combination of other vectors.

    Args:
        a: vector to check
        m: list of vectors to check against, each row is a vector

    Returns:
        True if a is linearly dependent on m, False otherwise
    """
    matrix = m.copy()
    matrix.append(a)
    matrix = [torch.tensor([float(x) for x in row]) for row in matrix]

    _, independent_indices = find_independent_tensors(matrix)

    if set(independent_indices) == set(range(len(matrix))):
        return True
    else:
        return False
