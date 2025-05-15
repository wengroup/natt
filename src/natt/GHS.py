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

from natt.EGH import get_G_even, get_g_matrix, get_G_odd, get_H, shift_index_2
from natt.evaluate import embed, evaluate_tensors, extract
from natt.matrix import (
    float_matrix,
    fraction_matrix,
    matrix_inverse,
    matrix_multiply,
    matrix_transpose,
)
from natt.ops import multiply_2, simplify_linear_combination
from natt.qr import find_independent_tensors, is_linear_independent
from natt.sym import symmetrize
from natt.symbolic import LinearCombination
from natt.symmetrize import symmetrize_and_remove_trace
from natt.utils import letter_index


def get_G_H_S_of_j(j: int, n: int, symmetry: str = None, seed: int = 35) -> tuple[
    list[LinearCombination],
    list[LinearCombination],
    list[LinearCombination],
    list[list[Fraction]],
    list[list[Fraction]],
]:
    """
    Get the G, H, S tensors for a given weight j and dimension n.

    Args:
        j: weight
        n: dim of the space T is in
        symmetry:

    Returns:
        G: independent G tensors of different seniority p
        H: H corresponding to G
        S: S corresponding to G and H
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
    torch.manual_seed(seed)
    X = torch.randn(3**j).reshape([3] * j)
    X = symmetrize_and_remove_trace(X)
    all_num_S = [embed(G, X) for G in all_G]

    # Get linearly independent S tensors
    _, independent_indices = find_independent_tensors(all_num_S)

    # Get linearly independent G tensors
    independent_G = [all_G[i] for i in independent_indices]

    # Get g_pq matrix for independent G
    g_pq = get_g_matrix(j, n, independent_G)

    # Get h_pq matrix
    h_pq = matrix_inverse(g_pq)

    # Get H tensors, corresponding to independent G
    independent_H = get_H(h_pq, independent_G)

    # Further down select unique G tensors by symmetry
    if symmetry is not None:
        indices_zero, indices_group = group_G_by_symmetry(independent_G, n, symmetry)

        # All G result in zero
        if len(indices_group) == 0:
            all_K = []
            ind_idx = []
        # Each G form its own group, i.e. all G are independent
        elif len(indices_group) == len(independent_G):
            all_K = independent_G
            ind_idx = range(len(independent_G))
        # Some G are not unique
        else:
            # TODO, These two can be combined as a single function
            coeff, ind_idx, dep_idx = get_independent_H_coeff(h_pq, indices_group)
            all_K = get_K(independent_G, coeff, ind_idx, dep_idx, indices_group)

        # We use K as G now
        independent_G = all_K
        independent_H = [independent_H[i] for i in ind_idx]

    # Get G, H, and S tensors
    all_G = []
    all_H = []
    all_S = []
    for i, (G, H) in enumerate(zip(independent_G, independent_H)):
        G = simplify_linear_combination(G)

        # Shift upper letters of H to distinguish those from G
        H = shift_index_2(H, n, letter_index(24, upper_case=True))
        H = simplify_linear_combination(H)

        S = multiply_2(G, H)
        S = simplify_linear_combination(S)

        all_G.append(G)
        all_H.append(H)
        all_S.append(S)

    return all_G, all_H, all_S, g_pq, h_pq


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

        # There is no natural tensor of this rank
        if len(G) == 0:
            continue

        out_j = {
            "g_pq": {"symbolic": fraction_matrix(g), "numerical": float_matrix(g)},
            "h_pq": {"symbolic": fraction_matrix(h), "numerical": float_matrix(h)},
            "G": [],
            "H": [],
            "S": [],
        }

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

        out[j] = out_j

    return out


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


# TODO, this can be done symbolically. Probably do it.
#  We need:
#  1. symbolic symmetrize() to get T. It is implemented in ops.py, but commented out
#  2. multiply_2() to get X = G \odot^n T
#  3. Simplify_linear_combination() to get the simplified X.
#  4. Compare X to see if they are the same.


#
# TODO, this can be refactored to remove `symmetry` and provide a tensor `T` as input.
#
def group_G_by_symmetry(
    all_G: list[LinearCombination],
    rank: int,
    symmetry: str,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> tuple[list[int], list[list[int]]]:
    r"""
    Group the G tensors by their uniqueness for a given symmetry.

    This is achieved by numerical experiments (although it can be done symbolically):
    1. Creating a tensor T with the given symmetry.
    2. For each G, obtain X = G \odot^n T.
    3. Check each X to verify whether a) it is zero, or b) it is unique (i.e. a
       duplicate of another X), and then label the corresponding G accordingly.

    Args:
        all_G: linear independent G tensors.
        rank: rank of the tensor
        symmetry: symmetry specifying the tensor. e.g.
            - "ij=ji" denotes a rank-2 tensor that is symmetric in the last two indices
                (e.g. dielectric tensor, stress tensor);
            - "ijk=ikj" denotes a rank-3 tensor that is symmetric in the last two indices
                (e.g. piezoelectric tensor).
            - "ijkl=ijlk=klij" denotes a rank-4 tensor that is symmetric in first two
            indices, symmetric in last two indices, and symmetric in first-two and
            last-two indices ( e.g. elastic tensor).
        rtol: relative tolerance for checking if two tensors are equal.
        atol: absolute tolerance for checking if two tensors are equal.

     Returns:
        indices_zero: Indices of zero G.
        indices_group: Each inner list contains the indices of G tensors that are
            equivalent to each other, meaning their corresponding X tensors are the
            same.
    """
    # Create a tensor T with the specified symmetry
    torch.manual_seed(35)
    T = torch.randn(*([3] * rank))
    T = symmetrize(T, symmetry)

    # TODO, delete,  Hard code it
    # X = torch.randn((3,) * 2)
    # X = symmetrize_and_remove_trace(X)
    # Y = torch.randn((3,) * 2)
    # Y = symmetrize_and_remove_trace(Y)
    # T = torch.einsum(f"ij,kl->ijkl", X, Y)
    #####

    all_X = [extract(G, T) for G in all_G]

    indices_zero = []
    indices_group = []
    for i, X in enumerate(all_X):

        # 1. Remove zeros
        if torch.allclose(X, torch.tensor(0.0), rtol=rtol, atol=atol):
            indices_zero.append(i)
            continue

        # 2. Create groups of equivalent G tensors
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
