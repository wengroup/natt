"""Matrix operations for Fraction objects."""

from fractions import Fraction


def matrix_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    """
    Calculate the inverse of a matrix containing Fraction objects.

    Returns the inverse matrix with Fraction elements.

    Args:
        matrix: List of lists containing Fraction objects

    Returns:
        Inverse matrix as list of lists with Fraction objects
    """
    n = len(matrix)

    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(matrix[i][j])
        for j in range(n):
            row.append(Fraction(1) if i == j else Fraction(0))
        augmented.append(row)

    # Gaussian elimination
    for i in range(n):
        # Find pivot
        pivot = augmented[i][i]
        if pivot == 0:
            raise ValueError("Matrix is not invertible")

        # Scale row to make pivot 1
        for j in range(2 * n):
            augmented[i][j] = augmented[i][j] / pivot

        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]

    # Extract inverse matrix
    inverse = []
    for i in range(n):
        inverse.append([])
        for j in range(n):
            inverse[i].append(augmented[i][j + n])

    return inverse


def matrix_multiply(
    m1: list[list[Fraction]], m2: list[list[Fraction]]
) -> list[list[Fraction]]:
    """Perform matrix multiplication on two matrices containing Fraction objects.

    Args:
        m1: First matrix
        m2: Second matrix

    Returns:
        Resulting matrix as list of lists with Fraction objects
    """
    n = len(m1)
    m = len(m2[0])
    p = len(m2)

    if len(m1[0]) != p:
        raise ValueError("Incompatible matrix dimensions for multiplication")

    result = []
    for i in range(n):
        row = []
        for j in range(m):
            value = Fraction(0)
            for k in range(p):
                value += m1[i][k] * m2[k][j]
            row.append(value)
        result.append(row)

    return result


def matrix_transpose(m: list[list[Fraction]]) -> list[list[Fraction]]:
    """Transpose a matrix containing Fraction objects.

    Args:
        m: Matrix to be transposed

    Returns:
        Transposed matrix as list of lists with Fraction objects
    """
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def float_matrix(m: list[list[Fraction]]) -> list[list[float]]:
    """
    Convert a matrix of Fraction objects to a matrix of floats.

    Args:
        m: List of lists containing Fraction objects

    Returns:
        List of lists containing floats
    """
    return [[float(fraction) for fraction in row] for row in m]


def fraction_matrix(m: list[list[Fraction]]) -> list[list[str]]:
    """
    Convert a matrix of Fraction objects to a matrix of strings.

    Each Fraction is represented as a string in the form "numerator/denominator".

    Args:
        m: List of lists containing Fraction objects

    Returns:
        List of lists containing strings representing the fractions
    """
    return [[str(fraction) for fraction in row] for row in m]
