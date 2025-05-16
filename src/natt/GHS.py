"""
Symbolic and numerical G, H, S tensors.

G, H, and S are made of only the Kronecker delta and Levi-Civita symbols.

G, H, and S can be used to map a general tensor T and a natural tensor X.
S = G H
X = H T
T' = G X = (G H) T = S T
where T' is the embedding of X in the T space.
"""

from fractions import Fraction
from pprint import pprint

import torch
from torch import Tensor

from natt.EGH import get_G_even, get_g_matrix, get_G_odd, get_H, get_S
from natt.evaluate import embed, evaluate_tensors, extract
from natt.matrix import (
    float_matrix,
    fraction_matrix,
    matrix_inverse,
    matrix_multiply,
    matrix_transpose,
)
from natt.ops import simplify_linear_combination
from natt.qr import find_independent_tensors, is_linear_independent
from natt.sym import get_random_tensor_of_symmetry
from natt.symbolic import LinearCombination
from natt.symmetrize import get_random_natural_tensor
from natt.utils import letter_index


def get_G_H_S(n: int, symmetry: str = None, numerical: bool = True) -> dict:
    """
    Get all the G, H, S tensors of dimension n.

    Args:
        n: dim of the space T is in
        symmetry: symmetry of the tensor in space n, if any. For example,
            - "ij=ji" means that the target is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ijk=ikj" means that the target is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the target is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the target is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry;
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).
        numerical: whether to return numerical values of G, H, S.

    Returns:
        G, H, S, and g_pq, h_pq information.
    """
    out = {}
    for j in range(n + 1):
        G, H, S, g, h = get_G_H_S_of_j(j, n, symmetry)

        # No natural tensor of this rank
        if len(G) == 0:
            continue

        # Get rules and numerical values
        out_j = get_G_H_S_rules_and_values(
            j, n, G, H, S, g, h, numerical, include_g=True, include_h=True
        )

        out[j] = out_j

    return out


def get_G_H_S_natural(
    j1: int, j2: int, max_j3: int = None, numerical: bool = True
) -> dict:
    """
    Get all the G, H, S tensors of a tensor product of two natural tensors.

    Z = X \otimes Y, where X and Y are natural tensors.

    Args:
        j1: rank of X
        j2: rank of Y
        max_j3: rank of Z. The output will have ranks of abs(j1-j2) <= j3 <= max_j3.
            If max_j3 is None, it will be set to j1 + j2.
        numerical: whether to return numerical values of G, H, S.

    Returns:
        G, H, S, and g_pq, h_pq information corresponding to Z.
    """

    if max_j3 is None:
        max_j3 = j1 + j2
    else:
        if max_j3 > j1 + j2:
            raise ValueError("`max_j3` must be smaller than or equal to `j1 + j2`.")

    out = {}
    for j in range(abs(j1 - j2), max_j3 + 1):
        G, H, S, g, h = get_G_H_S_of_j_natural(j1, j2, j)

        # No natural tensor of this rank
        if len(G) == 0:
            continue

        # Get rules and numerical values
        out_j = get_G_H_S_rules_and_values(
            j, j1 + j2, [G], [H], [S], g, h, numerical, include_g=True, include_h=True
        )

        out[j] = out_j

    return out


def get_G_H_S_of_j(j: int, n: int, symmetry: str = None) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the G, H, S tensors for a given weight j and rank n.

    This can deal with / without symmetry.

    Args:
        j: weight
        n: dim of the space T is in
        symmetry: symmetry of the tensor in space n, if any. For example,
            - "ij=ji" means that the target is a fully symmetric rank-2 tensor (e.g.
                stress tensor);
            - "ijk=ikj" means that the target is a rank-3 tensor with the last two
                indices symmetric (e.g. piezoelectric tensor);
            - "ijk=ikj=jik" means that the target is a fully symmetric rank-3 tensor;
            - "ijkl=jikl=klij" means that the target is a rank-4 tensor with both minor
                symmetry (between i and j, and between k and l) and major symmetry (
                between ij and kl). For example, the elastic tensor has this symmetry;
            The number of unique letters gives the rank of the tensor (what letters to
            use does not matter).

    Returns:
        G: independent G tensors of different seniority p
        H: H corresponding to G
        S: S corresponding to G and H
        g: g_pq matrix
        h: h_pq matrix
    """
    # Get independent G and H tensors for a general tensor
    ind_G, ind_H, g, h = get_G_H_of_j(j, n)

    # Further down select G and H for tensors with symmetry
    if symmetry is not None:
        # Get independency of G by using a random tensor of the given symmetry
        T = get_random_tensor_of_symmetry(n, symmetry)
        _, indices_group = group_G(T, ind_G)

        # Combine G (and H) to create new independent G (and H) tensors
        ind_G, ind_H = combine_G_H_of_j(ind_G, ind_H, h, indices_group)

    # Get S tensors
    G = [simplify_linear_combination(G) for G in ind_G]
    H = [simplify_linear_combination(H) for H in ind_H]
    S = get_S(G, H, n)

    return G, H, S, g, h


def get_G_H_S_of_j_natural(j1: int, j2: int, j3: int) -> tuple[
    LinearCombination,
    LinearCombination,
    LinearCombination,
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the G, H, S tensors Z = X \otimes Y, where X and Y are natural tensors.

    There will be a single G, H, S tensors for a given j1, j2, and j3.

    Args:
        j1: rank of X
        j2: rank of Y
        j3: rank of Z

    Returns:
        G: independent G tensor of different seniority p
        H: H corresponding to G
        S: S corresponding to G and H
        g: g_pq matrix
        h: h_pq matrix
    """
    n = j1 + j2

    # Get independent G and H tensors for a general tensor
    ind_G, ind_H, g, h = get_G_H_of_j(j3, n)

    # Further down select G and H for tensors with symmetry

    # Create a random tensor Z = X \otimes Y
    X = get_random_natural_tensor(j1, seed=35)
    Y = get_random_natural_tensor(j2, seed=36)
    X_indices = letter_index(j1)
    Y_indices = letter_index(j2, start=j1)
    Z = torch.einsum(f"{X_indices},{Y_indices}->{X_indices}{Y_indices}", X, Y)

    # Get independency of G by using Z
    _, indices_group = group_G(Z, ind_G)

    if len(indices_group) != 1:
        raise RuntimeError(
            "There should only be one group of G tensors, but got "
            f"{len(indices_group)} groups"
        )

    # Combine G (and H) to create new independent G (and H) tensors
    ind_G, ind_H = combine_G_H_of_j(ind_G, ind_H, h, indices_group)

    # Get S tensors
    G = [simplify_linear_combination(G) for G in ind_G]
    H = [simplify_linear_combination(H) for H in ind_H]
    S = get_S(G, H, n)

    # There should only be one G, H, S
    return G[0], H[0], S[0], g, h


def get_G_H_of_j(j: int, n: int) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the independent G and H tensors for a given weight j and rank n.

    Note, here, the independence of G and H are for a general tensor T. For tensors with
    certain symmetry (e.g. tensor that is the product of two natural tensors),
    further processing is needed to get the independent G and H tensors.

    Args:
        j: weight of the natural tensor X
        n: rank of the ordinary tensor T

    Returns:
        G: independent G tensors for ordinary tensor
        H: independent H tensors for ordinary tensor, corresponding to G
        g: g_pq matrix
        h: h_pq matrix
    """
    # create G mapping operator
    if (n - j) % 2 == 0:
        all_G = get_G_even(j, n)
    else:
        all_G = get_G_odd(j, n)

    # WARNING, should not simplify G using the below function, as get_g_matrix() below
    # is set up to work with the original G tensors.
    # all_G = [simplify_linear_combination(g) for g in all_G]

    # Get numerical S tensors, embedding a random natura tensor X in space j to space n
    X = get_random_natural_tensor(j)
    all_num_S = [embed(G, X) for G in all_G]

    # Get linearly independent S tensors
    _, independent_indices = find_independent_tensors(all_num_S)

    # Get linearly independent G tensors
    ind_G = [all_G[i] for i in independent_indices]

    # Get g_pq matrix for independent G
    g = get_g_matrix(j, n, ind_G)

    # Get h_pq matrix
    h = matrix_inverse(g)

    # Get H tensors, corresponding to independent G
    ind_H = get_H(h, ind_G)

    return ind_G, ind_H, g, h


def combine_G_H_of_j(
    G: list[LinearCombination],
    H: list[LinearCombination],
    h: list[list[Fraction]],
    indices_group: list[list[int]],
) -> tuple[list[LinearCombination], list[LinearCombination]]:
    """
    Combine the G and H tensors based on the symmetry of the tensor.

    For tensors with certain symmetry, the independent G and H tensors obtained via
    `get_G_H_of_j` are not independent anymore. This functions linearly combines the
    G and H tensors to obtain new independent G and H tensors.

    Args:
        G: independent G tensors for ordinary tensor
        H: independent H tensors for ordinary tensor
        h: h_pq matrix
        indices_group: each inner list contains indices of G tensors that are equivalent
        to each other.

    Returns:
        ind_G: independent G tensors for tensor with symmetry
        ind_H: independent H tensors for tensor with symmetry, corresponding to G
    """

    # All G result in zero
    if len(indices_group) == 0:
        K = []
        ind_idx = []
    # Each G form its own group, i.e. all G are independent
    elif len(indices_group) == len(G):
        K = G
        ind_idx = range(len(G))
    # Some G are not unique
    else:
        # TODO, These two can be combined as a single function
        coeff, ind_idx, dep_idx = get_independent_H_coeff(h, indices_group)
        K = get_K(G, coeff, ind_idx, dep_idx, indices_group)

    # We use K as G now
    ind_G = K
    ind_H = [H[i] for i in ind_idx]

    return ind_G, ind_H


def get_G_H_S_rules_and_values(
    j: int,
    n: int,
    G: list[LinearCombination],
    H: list[LinearCombination],
    S: list[LinearCombination],
    g: list[list[Fraction]],
    h: list[list[Fraction]],
    numerical: bool = True,
    include_g: bool = True,
    include_h: bool = True,
):
    """
    Get the numerical values of G, H, S tensors and the rules for performing tensor
    products.

    Args:
        j:
        n:
        G:
        H:
        S:
        g:
        h:
        numerical:
        include_g:
        include_h:

    Returns:

    """

    out_j = {"G": [], "H": [], "S": []}

    if include_g:
        out_j["g_pq"] = {"symbolic": fraction_matrix(g), "numerical": float_matrix(g)}

    if include_h:
        out_j["h_pq"] = {"symbolic": fraction_matrix(h), "numerical": float_matrix(h)}

    # loop over seniority p
    for G_p, H_p, S_p in zip(G, H, S):

        lower = letter_index(j)
        upper = letter_index(n, upper_case=True)
        upper2 = letter_index(n, start=n, upper_case=True)

        # G
        out_j["G"].append(
            {
                "symbolic": str(G_p),
                "rule": (f"{upper}{lower},...{lower}->...{upper}"),
            },
        )
        if numerical:
            out_j["G"][-1]["numerical"] = evaluate_tensors(G_p, mode="G")

        # H
        out_j["H"].append(
            {"symbolic": str(H_p), "rule": f"{lower}{upper},...{upper}->...{lower}"}
        )
        if numerical:
            out_j["H"][-1]["numerical"] = evaluate_tensors(H_p, mode="H")

        # S
        out_j["S"].append(
            {
                "symbolic": str(S_p),
                "rule": f"{upper}{upper2},...{upper2}->...{upper}",
            }
        )
        if numerical:
            out_j["S"][-1]["numerical"] = evaluate_tensors(S_p, mode="S")

    return out_j


# TODO, this can be done symbolically. Probably do it.
#  We need:
#  1. symbolic symmetrize() to get T. It is implemented in ops.py, but commented out
#  2. multiply_2() to get X = G \odot^n T
#  3. Simplify_linear_combination() to get the simplified X.
#  4. Compare X to see if they are the same.


#
# TODO, this can be refactored to remove `symmetry` and provide a tensor `T` as input.
#
def group_G(
    T: Tensor,
    all_G: list[LinearCombination],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[list[int], list[list[int]]]:
    r"""
    Group the G tensors by their uniqueness when operating on a tensor T.

    This is achieved by numerical experiments:
    1. For each G, obtain X = G \odot^n T.
    2. Check each X to verify whether:
        a. it is zero;
        b) it is unique (i.e. the same as another X),
    and then label the corresponding G accordingly.

    Args:
        T: the tensor for G to operate on. It can be a general tensor, or tensors of
            certain symmetry, and it can be traceless too.
        all_G: linear independent G tensors.
        rtol: relative tolerance for checking if two tensors are equal.
        atol: absolute tolerance for checking if two tensors are equal.

     Returns:
        indices_zero: Indices of zero G.
        indices_group: Each inner list contains the indices of G tensors that are
            equivalent to each other, meaning their corresponding X tensors are the
            same.
    """
    all_X = [extract(G, T) for G in all_G]

    indices_zero = []
    indices_group = []
    for i, X in enumerate(all_X):

        # Check zeros
        if torch.allclose(X, torch.tensor(0.0), rtol=rtol, atol=atol):
            indices_zero.append(i)
            continue

        # Create groups of equivalent G tensors
        is_unique = True
        for group in indices_group:
            j = group[0]

            if torch.allclose(X, all_X[j], rtol=rtol, atol=atol):
                # Equivalent to values in an existing group, then add to the group
                group.append(i)
                is_unique = False
                break

        # Not in existing groups, create a new group
        if is_unique:
            indices_group.append([i])

    return indices_zero, indices_group


def get_independent_H_coeff(
    h: list[list[Fraction]], indices_group: list[list[int]]
) -> tuple[list[list[Fraction]], list[int], list[int]]:
    """
    Construct coefficient matrix to combine independent H tensors to obtain other H.

    This is based on the values of the G:
    1. For GT=0, we ignore the corresponding h_pq.
    2. For G1, G2...Gq that gives the same GT values, we sum the corresponding h_pq
    over q.

    Args:
        h:
        indices_group:

    Returns:
        coeff: Each column gives the coefficients of combining independent H to obtain
            other H.
        ind_indices: indices of independent H
        dep_indices: indices of dependent H
    """
    num_ind = len(indices_group)

    # Gather coefficients of equivalent G tensors
    # We have:
    # H_p = h_p1 G_1 _+ h_p2 G_2 + ... + h_pq G_q
    # and the G are equivalent in each group.
    # We obtain:
    # H_p = u_p1 G_1 + u_p2 G_2 + u_pr G_r
    # where r is the number of unique groups.
    u = []
    for h_p in h:
        u_p = []
        for group in indices_group:
            # sum over the group
            u_pq = sum(h_p[i] for i in group)
            u_p.append(u_pq)
        u.append(u_p)

    # Split the H tensors (u here) into independent ones M and dependent ones N
    M = []  # independent H
    M_indices = []  # indices of independent H
    N = []  # dependent H
    N_indices = []  # indices of dependent H

    num = 0
    for i, row in enumerate(u):

        # Independent H tensor
        if (
            num < num_ind  # not find enough
            # and any(val != 0 for val in row)  # all zeros are not independent
            and is_linear_independent(row, M)  # independent to currently selected
        ):
            M.append(row)
            M_indices.append(i)
            num += 1
        # dependent H tensor
        else:
            N.append(row)
            N_indices.append(i)
    if not num == num_ind:
        raise RuntimeError("Not enough independent H tensors found.")

    M = matrix_transpose(M)
    N = matrix_transpose(N)
    M_inv = matrix_inverse(M)
    coeff = matrix_multiply(M_inv, N)

    return coeff, M_indices, N_indices


def get_K(
    all_G: list[LinearCombination],
    coeff: list[list[Fraction]],
    ind_indices: list[int],
    dep_indices: list[int],
    indices_group: list[list[int]],
) -> list[LinearCombination]:
    """
    Get K tensors.

    These will be used as the new G tensors.

    Args:
        all_G:
        coeff: Each column gives the coefficients of combining independent H to obtain
            other H.
        indices_group: each inner list contains indices of G tensors that are
            equivalent to each other.

    Returns:
    """

    all_K = []
    for ii, i in enumerate(ind_indices):
        K = all_G[i]
        for jj, j in enumerate(dep_indices):
            K += coeff[ii][jj] * all_G[j]
        all_K.append(K)

    return all_K


if __name__ == "__main__":

    # # elastic tensor
    # j = 4
    # rank = 4
    # symmetry = "ijkl=jikl=klij"
    # get_G_H_S_of_j(j, rank, symmetry)

    ######
    # rank = 2
    # symmetry = "ij=ji"
    rank = 4
    symmetry = "ijkl=jikl=klij"
    out = get_G_H_S(rank, symmetry, numerical=False)
    pprint(out)
    # dumpfn(out, "out.yaml")
