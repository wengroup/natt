"""
Symmetrization of a tensor (w or w/o partial symmetry) to get a fully symmetric tensor.
"""

import itertools

import torch
from torch import Tensor

from natt.utils import dij, double_index, letter_index, repeat_double_index


def symmetrize_and_remove_trace(
    t: Tensor, start_dim: int = 0, symmetry: str = None
) -> Tensor:
    """
    Symmetrize and remove the trace of a (generic) tensor.

    This only extracts the symmetric traceless part of the tensor at the same rank.
    The antisymmetric part is totally ignored, which can further be decomposed into
    symmetric and traceless tensors of lower ranks.

    Args:
        t: input tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.
        symmetry: a string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between

    Returns:
        A symmetric traceless tensor of the same rank as the input tensor.
    """
    return remove_trace(symmetrize(t, start_dim, symmetry), start_dim)


def symmetrize(
    t: Tensor, start_dim: int = 0, symmetry: str = None, mode: str = "mean"
) -> Tensor:
    """
    Symmetrize a tensor.

    The symmetrization is done by averaging/sum over unique permutations of the indices,
    considering the symmetry of the indices.

    Args:
        t: The tensor to symmetrize
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.
        symmetry: A string that describes the symmetry of the indices. For example,
            `abba` means the first and the fourth indices are symmetric, and the
            second and the third indices are symmetric. Default is None, which means
            there is no symmetry between the indices.
        mode: `mean` or `sum`. If `mean`, the tensor is averaged over the permutations.
            If `sum`, the tensor is summed over the permutations.

    Returns:
        The symmetrized tensor.
    """

    # fully symmetrize the tensor
    if symmetry is None:
        permutations = itertools.permutations(range(start_dim, t.ndim))
        if start_dim > 0:
            prefix = list(range(start_dim))
            permutations = [prefix + list(p) for p in permutations]

    # symmetrize with the given symmetry
    else:
        assert (
            start_dim + len(symmetry) == t.ndim
        ), "The length of the symmetry string must match the tensor shape."
        permutations = get_permutations(symmetry, start_dim)

    if mode == "mean":
        u = torch.mean(torch.stack([t.permute(p) for p in permutations]), dim=0)
    elif mode == "sum":
        u = torch.sum(torch.stack([t.permute(p) for p in permutations]), dim=0)
    else:
        raise ValueError("The mode must be either 'mean' or 'sum'.")

    return u


def symmetrize_2(t: Tensor, num_delta: int, start_dim: int = 0) -> Tensor:
    """
    Symmetrize a tensor that is obtained by contracting a symmetric tensor with deltas.

    Symmetrization is done by summation over unique permutations of the indices,
    considering the three set of symmetries. See `get_permutations_2` for more details.

    Args:
        t: the tensor
        num_delta: number of deltas used to obtain the tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        A symmetrized tensor.
    """
    # rank of the symmetric tensor, after considering the start_dim
    m = t.ndim - start_dim

    # Get unique permutations
    permutations = get_permutations_2(m, num_delta, start_dim)

    # Sum over the permutations
    u = torch.sum(torch.stack([t.permute(p) for p in permutations]), dim=0)

    return u


# TODO, this can be refactored to be similar as unit_vector.py
def remove_trace(u: Tensor, start_dim: int = 0) -> Tensor:
    """
    Remove the trace of a symmetric tensors to get a natural tensor of the same rank.

    Args:
        u: a fully symmetric tensor
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    This implements:
    X_{ij\dots m} = U_{ij\dots m} + \sum_{d=1}^D (-1)^d \frac{(2m-2d-1)!!}{(2m-1)!!}
    \{ \delta_{ij}\delta_{kl} \dots U_{rrss\dots m} \}
    where:
    U: fully symmetric tensor of rank m
    X: natural tensor of rank m
    D: D=m/2 if m is even, D=(m-1)/2 if m is odd
    {} denotes fully symmetrization.

    References:
        1. Eq 10 of http://dx.doi.org/10.1080/00018737800101454
        2. Cartesian tensors writeup by Mingjian Wen, which is an explicit form of the
           above reference.

    Returns:
        A natural tensor of the same shape as the input tensor.
    """
    device = u.device

    m = u.ndim - start_dim
    D = m // 2

    delta = dij(device)
    coeff = 1
    out = u
    for d in range(1, D + 1):
        rule = remove_trace_rule(m, d)

        # Contract with multiple deltas to get a tensor of the same rank as u
        prod = torch.einsum(rule, u, *([delta] * d))

        prod = symmetrize_2(prod, num_delta=d, start_dim=start_dim)

        # coeff = (-1) ** d / double_factorial(2 * m - 1, 2 * m - 2 * d - 1 + 2, device)
        coeff = -coeff / (2 * m - 2 * d + 1)

        out = out + coeff * prod

    return out


# TODO, this is a generalization of get_sym_rule_2 and get_sym_rule_3 in tensor_product1.py
#  Can we merge them?
def get_permutations(symmetry: str, start_dim: int = 0) -> list[list[int]]:
    """
    Get the unique permutations of the indices to fully symmetrize a tensor.

    This works for the case where part or all of the indices are symmetric.

    Args:
        symmetry: A string that describes the symmetry already in the tensor. For
            example, `abba` means the first and the fourth indices are symmetric, and
            the second and the third indices are symmetric. The symmetry only applies to
            the indices after the `start_dim`.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Example:
        >>> get_permutations('abba')
        [[0, 1, 2, 3],  # abba
         [0, 1, 3, 2],  # abab
         [0, 3, 1, 2],  # aabb
         [1, 0, 2, 3],  # baba
         [1, 0, 3, 2],  # baab
         [1, 2, 0, 3]]  # bbaa
        >>> get_permutations('abba', 2)
        [[0, 1, 2, 3, 4, 5],  # abba
         [0, 1, 2, 3, 5, 4],  # abab
         [0, 1, 2, 5, 3, 4],  # aabb
         [0, 1, 3, 2, 4, 5],  # baba
         [0, 1, 3, 2, 5, 4],  # baab
         [0, 1, 3, 4, 2, 5]]  # bbaa

    Returns:
        Each tuple contains the permutation indices for symmetrization.
    """

    all_perms = itertools.permutations(range(start_dim, start_dim + len(symmetry)))

    prefix = list(range(start_dim))
    unique_perms = []
    unique_perm_string = set()

    # Filter permutations based on the symmetry
    for perm in all_perms:
        perm_string = "".join(symmetry[i - start_dim] for i in perm)

        if perm_string not in unique_perm_string:
            unique_perms.append(prefix + list(perm))
            unique_perm_string.add(perm_string)

    return unique_perms


# TODO, we need to rename this to make it general.
#  We can call this minor and major symmetries. just like elastic tensor.
#  T delta_ij delta_kl = T delta_kl delta_ij we have minor symmetry between i and j, \
#  and between k and l. We have major symmetry between (ij) and (kl).
#  This is the same as the elastic tensor ((ij)(kl)).
#  So, we can rename this function to make it more general.
def get_permutations_delta(
    symmetry: str, delta_indices: str, start_dim: int = 0
) -> list[list[int]]:
    """
    Get the unique permutations of the indices to fully symmetrize a tensor.
    that is obtained by tensor product with delta tensors.

    For example, `symmetry = xxyyaabb`, and `delta_indices = ab` means:
        1. indices 1 and 2 are symmetric (both associated with `x`), and indices 3 and
           4 are symmetric (both associated with `y`)
        2. indices 5 and 6 are symmetric (both associated with `a`), and indices 7 and
           8 are symmetric (both associated with `b`)
        3. indices 5,6 and 7,8 are symmetric because both `a` and `b` are in
           `delta_indices`, meaning they are associated with delta tensors.

    So, we consider three types of symmetries:
        1. symmetries in other indices of the tensor
        2. minor symmetry in delta tensors
        3. major symmetry in delta tensors

    For 2 and 3, consider, for example:
        delta_ij delta_kl = delta_ji delta_lk = delta_kl delta_ij,
    So, we have both minor and major symmetries in delta tensors.

    The above example can be though as symmetrizing a tensor Z obtained as:
        Z = X \otimes Y \otimes \delta \otimes \delta
    where `X` and `Y` are rank-2 symmetric tensors, and `\delta` are delta tensors.

    As another example, consider `symmetry = xxxxaabb` and `delta_indices = ab`, then
    it can be thought as symmetrizing a tensor Z obtained as:
        Z = X \otimes \delta \otimes \delta \otimes \delta
    where `X` is a rank-4 symmetric tensor, and `\delta` are delta tensors.

    As can be seen, `delta_indices` does not necessarily need to be associated with
    deltas. It can be indices of the same symmetric tensor. For example, the permutation
    of
        Z = X \otimes X \otimes Y \otimes Y
    where `X` and `Y` are rank-2 symmetric tensors,
    can be obtained using `symmetry = xxaabb` and `delta_indices = ab`.


    Args:
        symmetry: A string that describes the symmetry already in the tensor. This
            should be used together with `delta_indices`. See below.
        delta_indices: The indices that are associated with the delta tensors, for which
            need to consider both minor and major symmetries.
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        Each inner tuple contains the permutation indices for symmetrization.
    """

    # TODO, this is not TorchScript compatible
    # If we want to use this in TorchScript,
    # We can create a data file to store the permutations and load it here.
    # start_dim can be easily handled by adding a constant.
    all_perms = itertools.permutations(
        torch.arange(start_dim, start_dim + len(symmetry))
    )

    prefix = list(range(start_dim))
    unique_perms: list[list[int]] = []
    unique_canonical_forms: list[str] = []

    # Filter permutations based on contraction pattern
    for perm in all_perms:
        perm_string = "".join(symmetry[i - start_dim] for i in perm)

        canonical_form = _canonize(perm_string, delta_indices)
        if canonical_form not in unique_canonical_forms:
            unique_perms.append(prefix + [int(i) for i in perm])
            unique_canonical_forms.append(canonical_form)

    return unique_perms


def _canonize(ps: str, di: str) -> str:
    """
    Convert a permutation string to its canonical form, such that equivalent
    permutation strings have the same representation.

    Major symmetry is based on first occurrence positions of each letter in the
    string. For example, `baba` and `fefe` are equivalent permutation strings, and
    both will be converted to `0101`.

    Args:
        ps: The permutation string to convert
        di: `delta_indices`

    Returns:
        The canonical form of the permutation string
    """
    # Do not need to canonize indices not in `delta_indices`.
    # For example, in `symmetry = xxyyaabb` and `delta_indices = ab`,
    # xxyy and yyxx are different.
    return "".join(str(ps.index(c)) if c in di else c for c in ps)


def get_permutations_2(m: int, num_delta: int, start_dim: int = 0) -> list[list[int]]:
    """

    Get the unique permutations of the tensor product of a symmetric tensor and deltas.

    For example, we know
    {U_rrss \delta_ij \delta_kl}
    = U_rrss \delta_ij \delta_kl
    + U_rsrs \delta_ij \delta_kl
    + U_rssr \delta_ij \delta_kl

    This is equivalent to
    1. First get V_ijkl = U_rrss \delta_ij \delta_kl
    2. Then permute V_ijkl to get V_ikjl and V_iklj
    3. Sum them up to get the result, i.e.
        {U_rrss \delta_ij \delta_kl} = V_ijkl + V_ikjl + V_iklj


    This function find the permutations of the indices in V.
    There are two types of symmetry to consider in the permutations:
    a. Minor symmetry: the symmetry of the two indices in each delta tensor.
       For example, V_ijkl = V_ijlk = V_jikl = V_jilk
    b. Major symmetry: the symmetry of indices between the deltas. For example,
       V_ijkl = V_klij

    In addition, we consider another symmetry:
    c. The symmetry of the remaining indices of the tensor, e.g. in U_rrst\delta_ij,
        the indices r and s are symmetric.

    Args:
        m: the rank of the symmetric tensor
        num_delta: the number of delta tensors to be contracted
        start_dim: the starting dimension to perform the operation. Dimensions before
            `start_dim` will not be used in the operation.

    Returns:
        Each inner list contains the permutation indices for symmetrization.
    """

    num_remain = m - 2 * num_delta
    assert num_remain >= 0, "The number of remaining indices must be non-negative."

    # Construct the symmetry pattern, e.g., zzaabb
    u_remain = "z" * num_remain
    delta = "".join(repeat_double_index(num_delta))
    symmetry = f"{u_remain}{delta}"

    delta_indices = letter_index(num_delta)

    perms = get_permutations_delta(symmetry, delta_indices, start_dim)

    return perms


def remove_trace_rule(m: int, d: int) -> str:
    """
    Get the contraction rule to remove the trace of a symmetric tensor.

    Note, d <= m/2.

    Args:
        m: rank of the symmetric tensor
        d: the number of delta

    Returns:
        rule: the contraction rule
        symmetry: the symmetry of the indices
    """
    u_contracted = "".join(repeat_double_index(d))
    u_remain = letter_index(m - 2 * d, start=d)
    delta = double_index(d, start=m - d)

    return (
        f"...{u_contracted}{u_remain},{','.join(delta)}->...{u_remain}{''.join(delta)}"
    )
