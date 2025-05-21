"""
Tensor operator to get Z = X \otimes Y.

This is based on our newly derived formulas (\label{eq:tp:even:H} and
\label{eq:tp:odd:H}) to get Z_l3 as: Z_l3 = H : X_l1 Y_l2, where : denotes contraction
with X and Y.
Unlike the equations in [LP89], which needs a loop to compute Z, the newly derived is
much more efficient as it just need a single tensor product.

[LP89] "Angular reduction in multiparticle matrix elements" by D. R. Lehman and W. C. Parke.
http://dx.doi.org/10.1063/1.528515
"""

from fractions import Fraction
from typing import Optional

import torch
from torch import Tensor

from natt.EGH import create_delta_epsilon_tensors
from natt.evaluate import evaluate_tensors
from natt.ops import simplify_linear_combination
from natt.symbolic import LinearCombination
from natt.symmetrize import get_permutations_delta
from natt.utils import (
    double_factorial,
    double_index,
    factorial,
    letter_index,
    repeat_double_index,
)


def get_H_numerical_even(
    l1: int, l2: int, l3: int, normalize: str = "unity"
) -> tuple[Tensor, str]:
    """
    Args:
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.
    """
    H, X_idx, Y_idx, Z_idx = get_H_even(l1, l2, l3)

    H = simplify_linear_combination(H)

    # We have three types indices, lower case for Z, upper case for X, and upper case
    # for Y. But the function evaluate_tensors() can only deal with two types of
    # indices: lower and upper case (cannot distinguish between X and Y). Then:
    # Q: Why we still can use it to evaluate H?
    # A: We take advantage of the fact that, by construction, all the indices in
    # X_idx are smaller than the indices in Y_idx, and that in `evaluate_tensors()`,
    # (actually `tp_delta_epsilon()`), the upper indices are sorted. As a result,
    # we have all the indices of X_idx # comes before Y_idx in H_numerical.
    # In other words, the indices of H_numerical is {Z_idx}{X_idx}{Y_idx}.
    # Then, we can use this to do H:XY.
    #
    # TODO, create a new function like evaluate_tensors to deal with this case.
    H_numerical = evaluate_tensors(H, mode="H")

    if normalize == "unity":
        c = coeff_C(l1, l2, l3)
        H_numerical *= c
    elif normalize == "none":
        pass
    else:
        supported = ["none", "unity"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )

    # Rule that can be used in einsum to obtain Z = einsum(rule, G, X, Y)
    rule = f"{Z_idx}{X_idx}{Y_idx},...{X_idx},...{Y_idx}->...{Z_idx}"

    return H_numerical, rule


def get_H_numerical_odd(
    l1: int,
    l2: int,
    l3: int,
    normalize: str = "unity",
) -> tuple[Tensor, str]:
    """

    Args:
        l1:
        l2:
        l3:
        normalize:

    Returns:
    """
    H, X_idx, Y_idx, Z_idx = get_H_odd(l1, l2, l3)
    H = simplify_linear_combination(H)

    H_numerical = evaluate_tensors(H, mode="H")

    if normalize == "unity":
        c = coeff_D(l1, l2, l3)
        H_numerical *= c
    elif normalize == "none":
        pass
    else:
        supported = ["none", "unity"]
        raise ValueError(
            f"Unknown normalization method: {normalize}. Supported are: {supported}."
        )

    # Rule that can be used in einsum to obtain Z = einsum(rule, G, X, Y)
    rule = f"{Z_idx}{X_idx}{Y_idx},...{X_idx},...{Y_idx}->...{Z_idx}"

    return H_numerical, rule


def get_H_even(l1: int, l2: int, l3: int) -> tuple[LinearCombination, str, str, str]:
    """
    Calculate the H operator tensor to obtain Z_l3 = H:XY, where l1 + l2 - l3 is even.

    Args:
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.

    Returns:
        H: Symbolic representation of the H tensor.
        X_idx: letters used as X indices in H.
        Y_idx: letters used as Y indices in H.
        Z_idx: letters used as Z indices in H.
    """
    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    k = (l1 + l2 - l3) // 2

    out = []
    for t in range(min(l1, l2) - k + 1):
        coeff = Fraction(
            (-2) ** t, double_factorial(2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2).item()
        )

        all_rules = get_H_rules_even(l1, l2, l3, t)

        # create tensor products of deltas for each rule
        tensors = [
            create_delta_epsilon_tensors(
                ru["ra"] + ru["sa"] + ru["aa"] + ru["rs"], factor=coeff
            )
            for ru in all_rules
        ]

        # extend them to sum up later
        out.extend(tensors)

    H = LinearCombination(*out)

    # Note, this should exactly the same as those in `get_H_rules_even()`
    X_idx = letter_index(l1, upper_case=True)
    Y_idx = letter_index(l2, start=l1, upper_case=True)
    Z_idx = letter_index(l3)

    return H, X_idx, Y_idx, Z_idx


def get_H_odd(l1: int, l2: int, l3: int) -> tuple[LinearCombination, str, str, str]:
    """
    Calculate the H operator tensor to obtain Z_l3 = H:XY, where l1 + l2 - l3 is odd.

    Args:
        l1: The rank of the first tensor X.
        l2: The rank of the second tensor Y.
        l3: The rank of the output tensor Z.
        normalize: The normalization method.
            If `unity`, the output is normalized such that the l3 fold contraction of
            the output tensor with a unit vector yields 1.
            If `none`, no normalization is applied.

    Returns:
        H: Symbolic representation of the H tensor.
        X_idx: letters used as X indices in H.
        Y_idx: letters used as Y indices in H.
        Z_idx: letters used as Z indices in H.
    """
    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    k = (l1 + l2 - l3 - 1) // 2

    out = []

    for t in range(min(l1, l2) - k):
        coeff = Fraction(
            (-2) ** t, double_factorial(2 * l3 - 1, 2 * l3 - 2 * t - 1 + 2).item()
        )

        all_rules = get_H_rules_odd(l1, l2, l3, t)

        # create tensor products of deltas for each rule
        tensors = [
            create_delta_epsilon_tensors(
                ru["ra"] + ru["sa"] + ru["aa"] + ru["rs"],
                epsilon=ru["rsa"][0],
                factor=coeff,
            )
            for ru in all_rules
        ]

        # extend them to sum up later
        out.extend(tensors)

    H = LinearCombination(*out)

    # Note, this should exactly the same as those in `get_H_rules_odd()`
    X_idx = letter_index(l1, upper_case=True)
    Y_idx = letter_index(l2, start=l1, upper_case=True)
    Z_idx = letter_index(l3)

    return H, X_idx, Y_idx, Z_idx


def get_H_rules_even(l1: int, l2: int, l3: int, t: int) -> list[dict[str, list[str]]]:
    """
    Get the rule for  { d_ra^{l1-k-t} d_sa^{l2-k-t} d_{aa}^t } d_rs^{k+t}.

    Args:
        l1:
        l2:
        l3:
        t:

    Returns:
        Each dict gives the indices for the Kronecker delta tensors d_ra, d_sa, d_aa,
        and d_rs, which can be used to create the left-hand-side of the einsum rule.
        The r_indices, s_indices, and a_indices can be used to create the
        right-hand-side of the einsum rule.
    """
    assert (l1 + l2 - l3) % 2 == 0, "l1 + l2 - l3 must be even"

    k = (l1 + l2 - l3) // 2

    # r indices for X, s indices for Y, a indices for Z
    r_idx = letter_index(l1, upper_case=True)
    s_idx = letter_index(l2, start=l1, upper_case=True)
    a_idx = letter_index(l3)

    _, symmetry, delta_indices = get_tp_even_rule(l1, l2, k, t)
    all_perms = get_permutations_delta(symmetry, delta_indices)

    n_ra = l1 - k - t
    n_sa = l2 - k - t
    n_aa = t

    # r indices in d_ra and d_rs
    r_ra_idx = r_idx[:n_ra]
    r_rs_idx = r_idx[n_ra:]

    # s indices in d_sa and d_rs
    s_sa_idx = s_idx[:n_sa]
    s_rs_idx = s_idx[n_sa:]

    # rs pairs
    rs_pairs = [r + s for r, s in zip(r_rs_idx, s_rs_idx)]

    all_rules = []
    for perm in all_perms:

        # Permute the a indices to symmetrize the output, namely considering the
        # curly braces {}. No need to permute the r and s indices
        #
        # Get p_a such that p_a[perm] -> a_idx. This is used because later when we do
        # tensor product to get Z = H:XY, the rule in the einsum will be sorted in the
        # right-hand-side. Then, p_a corresponds to the indices on the left-hand-side,
        # and a_idx (it is sorted here) corresponds to the indices on the
        # right-hand-side. Then we are essentially symmetrizing the tensors.
        #
        # The below is the same as
        # indices = [letters[perm.index(i)] for i in range(n)]
        # in get_G_rules_odd() and get_G_rules_even()
        p_a = [x for _, x in sorted(zip(perm, a_idx))]

        # a indices in d_ra, d_sa and d_aa
        a_ra_idx = p_a[:n_ra]
        a_sa_idx = p_a[n_ra : n_ra + n_sa]
        a_aa_idx = p_a[n_ra + n_sa :]

        # Pairs of indices for delta tensors
        ra_pairs = [r + a for r, a in zip(r_ra_idx, a_ra_idx)]
        sa_pairs = [s + a for s, a in zip(s_sa_idx, a_sa_idx)]
        aa_pairs = [a_aa_idx[2 * i] + a_aa_idx[2 * i + 1] for i in range(n_aa)]

        all_rules.append(
            {"ra": ra_pairs, "sa": sa_pairs, "aa": aa_pairs, "rs": rs_pairs}
        )

    return all_rules


def get_H_rules_odd(l1: int, l2: int, l3: int, t: int) -> list[dict[str, list[str]]]:
    """
    Get the rule for  { eps_rsa d_ra^{l1-k-t} d_sa^{l2-k-t} d_{aa}^t } d_rs^{k+t}.


    Args:
        l1:
        l2:
        l3:
        t:

    Returns:
        Rules for constructing the H operator for odd l1 + l2 - l3, e.g.
        {eps_ipA  d_jB d_kC  d_qD d_rE  D_FG} d_ls

    """

    assert (l1 + l2 - l3) % 2 == 1, "l1 + l2 - l3 must be odd"

    k = (l1 + l2 - l3 - 1) // 2

    # r indices for X, s indices for Y, a indices for Z
    r_idx = letter_index(l1, upper_case=True)
    s_idx = letter_index(l2, start=l1, upper_case=True)
    a_idx = letter_index(l3)

    _, symmetry, delta_indices = get_tp_odd_rule(l1, l2, k, t)
    all_perms = get_permutations_delta(symmetry, delta_indices)

    n_ra = l1 - k - t - 1
    n_sa = l2 - k - t - 1
    n_aa = t

    # r indices in epsilon, d_ra, and d_rs
    r_eps_idx = r_idx[0]
    r_ra_idx = r_idx[1 : n_ra + 1]
    r_rs_idx = r_idx[n_ra + 1 :]

    # s indices in epsilon, d_sa, and d_rs
    s_eps_idx = s_idx[0]
    s_sa_idx = s_idx[1 : n_sa + 1]
    s_rs_idx = s_idx[n_sa + 1 :]

    # rs pairs
    rs_pairs = [r + s for r, s in zip(r_rs_idx, s_rs_idx)]

    all_rules = []
    for perm in all_perms:

        # Permute the a indices to symmetrize the output, namely considering the
        # curly braces {}. No need to permute the r and s indices
        #
        # Get p_a such that p_a[perm] -> a_idx. This is used because later when we do
        # tensor product to get Z = H:XY, the rule in the einsum will be sorted in the
        # right-hand-side. Then, p_a corresponds to the indices on the left-hand-side,
        # and a_idx (it is sorted here) corresponds to the indices on the
        # right-hand-side. Then we are essentially symmetrizing the tensors.
        #
        # The below is the same as
        # indices = [letters[perm.index(i)] for i in range(n)]
        # in get_G_rules_odd() and get_G_rules_even()
        p_a = [x for _, x in sorted(zip(perm, a_idx))]

        # a indices in epsilon, d_ra, d_sa and d_aa
        a_eps_idx = p_a[0]
        a_ra_idx = p_a[1 : n_ra + 1]
        a_sa_idx = p_a[n_ra + 1 : n_ra + n_sa + 1]
        a_aa_idx = p_a[n_ra + n_sa + 1 :]

        # Pairs of indices for delta tensors
        rsa_triplet = [r_eps_idx + s_eps_idx + a_eps_idx]
        ra_pairs = [r + a for r, a in zip(r_ra_idx, a_ra_idx)]
        sa_pairs = [s + a for s, a in zip(s_sa_idx, a_sa_idx)]
        aa_pairs = [a_aa_idx[2 * i] + a_aa_idx[2 * i + 1] for i in range(n_aa)]

        all_rules.append(
            {
                "rsa": rsa_triplet,
                "ra": ra_pairs,
                "sa": sa_pairs,
                "aa": aa_pairs,
                "rs": rs_pairs,
            }
        )

    return all_rules


def get_tp_even_rule(l1: int, l2: int, k: int, t: int) -> tuple[str, str, str]:
    """
    Get the einsum rule when l1 + l2 - l3 is even.

    x_l1 \odot^{k+t} x_l2 \otimes I ^{\otimes^m}

    After contraction, the resultant tensor will have l1-k-t indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-k-t
    symmetric indices from y. It will have 2*t indices from I. Each two indices from I
    are symmetric.

    In total, the resultant tensor will have l3 = l1 + l2 - 2(k + t) tensor indices.

    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor.
            product. e.g. `xxxyyyaa` means the first three indices are symmetric, the
            next three indices are symmetric, and the last two indices are symmetric.
        delta_indices: The indices for the delta tensors.
    """

    # indices that are contracted
    xy_contracted = letter_index(k + t)

    # indices that are not contracted
    x_remain = letter_index(l1 - k - t, k + t)
    y_remain = letter_index(l2 - k - t, l1)

    # indices for contracting t of I
    delta = double_index(t, upper_case=True)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    rule = (
        f"...{xy_contracted}{x_remain},"
        f"...{xy_contracted}{y_remain}"
        f"{delta_left}"
        f"->...{x_remain}{y_remain}{delta_right}"
    )

    # l1-k-t remaining symmetric indices from x
    # l2-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    # TorchScript does not allow string multiplication, so we need to use `join`
    symmetry = (
        "".join(["a"] * len(x_remain))
        + "".join(["b"] * len(y_remain))
        + "".join(repeat_double_index(t, upper_case=True))
    )
    delta_indices = letter_index(t, upper_case=True)

    return rule, symmetry, delta_indices


def get_tp_odd_rule(l1: int, l2: int, k: int, t: int) -> tuple[str, str, str]:
    """
    Get the einsum rule when l1 + l2 - l3 is odd.

    epsilon : x_l1 \odot^{k+t} x_l2 \otimes I ^{\otimes^t}

    epsilon is the Levi-Civita symbol. It contracts away one index from x and one index
    from y. So, after contraction, the resultant tensor will have 1 index from epsilon.
    After contraction, the resultant tensor will have l1-1-k-t indices from x, and these
    indices are still symmetric. Similarly, the resultant tensor will have l2-1-k-t
    symmetric indices from y. It will have 2*m indices from I. Each two indices from I
    are symmetric.

    In total, the resultant tensor will have l3 = l1 + l2 - 1 - 2(k + t) indices.


    Returns:
        rule: The einsum rule for the tensor product
        symmetry: The symmetry information of the resultant tensor after the tensor
            product. e.g. `aabb` means the first two indices are symmetric, and the last
            two indices are symmetric.
        delta_indices: The indices for the delta tensors.
    """
    # example: epsilon_Uvw x_vabc  y_wabd I_AB -> UcdAB

    xy_contracted = letter_index(k + t)
    x_remain = letter_index(l1 - 1 - k - t, k + t)
    y_remain = letter_index(l2 - 1 - k - t, l1 - 1)

    # indices for contracting t of I
    delta = double_index(t, upper_case=True)
    delta_left = "," + ",".join(delta) if delta else ""
    delta_right = "".join(delta)

    # The ... are for the batch dimensions
    rule = (
        f"uvw,"  # indices for epsilon
        f"...v{xy_contracted}{x_remain},"
        f"...w{xy_contracted}{y_remain}"
        f"{delta_left}"
        f"->...u{x_remain}{y_remain}{delta_right}"
    )

    # 1 index from epsilon
    # l1-1-k-t remaining symmetric indices from x
    # l2-1-k-t remaining symmetric indices from y
    # 2t indices from all deltas. Each delta has 2 symmetric indices.
    symmetry = (
        "a"
        + "".join(["b"] * len(x_remain))
        + "".join(["c"] * len(y_remain))
        + "".join(repeat_double_index(t, upper_case=True))
    )
    delta_indices = letter_index(t, upper_case=True)

    return rule, symmetry, delta_indices


def coeff_C(l1: int, l2: int, l3: int, device: Optional[torch.device] = None):
    """Coefficient C for even L.

    The coefficient is obtained such at l3 fold contraction of the output tensor with
    A unit vector yields 1.

    Ref: Eq. 54 of [LP89]
    """
    L = l1 + l2 + l3
    L1 = L - 2 * l1 - 1
    L2 = L - 2 * l2 - 1
    L3 = L - 2 * l3 - 1

    return (
        factorial(l1, device)
        * factorial(l2, device)
        * double_factorial(2 * l3 - 1, device=device)
        * factorial((L1 + 1) // 2, device=device)
        * factorial((L2 + 1) // 2, device=device)
        / factorial(l3, device=device)
        / double_factorial(L1, device=device)
        / double_factorial(L2, device=device)
        / double_factorial(L3, device=device)
        / factorial(L // 2, device=device)
    )


def coeff_D(l1: int, l2: int, l3: int, device: Optional[torch.device] = None):
    """Coefficient D for odd L.

    The coefficient is obtained such at l3 fold contraction of the output tensor with
    A unit vector yields 1.

    Ref: Eq. 55 of [LP89]
    """
    L = l1 + l2 + l3
    L1 = L - 2 * l1 - 1
    L2 = L - 2 * l2 - 1
    L3 = L - 2 * l3 - 1

    return (
        2
        * factorial(l1, device)
        * factorial(l2, device)
        * double_factorial(2 * l3 - 1, device=device)
        * factorial(L1 // 2, device=device)
        * factorial(L2 // 2, device=device)
        / factorial(l3 - 1, device=device)
        / double_factorial(L1 + 1, device=device)
        / double_factorial(L2 + 1, device=device)
        / double_factorial(L3 + 1, device=device)
        / factorial((L + 1) // 2, device=device)
    )
