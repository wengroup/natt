from collections import Counter

import torch
from torch import Tensor

from natt.symbolic import Delta, Epsilon, LinearCombination, TensorProduct
from natt.symmetrize import symmetrize_and_remove_trace
from natt.utils import dij, eijk


def tp_delta_epsilon(tp: TensorProduct, mode: str) -> Tensor:
    """Get the tensor product of Kronecker delta and Levi-Civita tensors.

    Note, the order of the indices need to be taken care of.
    Upper-case letters are used to represent tensors in the n space (namely for
    tensors T and such), while lower-case letters are used to represent tensors in the
    j space (namely for tensors X). So:
    1. X = H T: H would consist of both lower case and upper-case letters, and its
       upper-case letters are to be contracted with T. We assume the contracting rule
       is something like X_ab = H_abABC T_ABC.
    2. T' = G X: G would consist of both lower case and upper-case letters, and its
       lower-case letters are to be contracted with X. We assume the contracting rule is
       something like T'_ABC = G_ABCab X_ab.
    3. T' = G H T = S T: S would consist of only upper-case letters. We assume the
       contracting rule is something like T'_ABC = S_ABCDEF T_DEF, where the first
       half of the indices are associated with the embedded tensor T', while the
       latter half of the indices are associated with the original tensor T.

    Args:
        tp: Tensor product of Kronecker delta and Levi-Civita tensors.
        mode: which mode to use, either `G`, `H`, or `S`. This determines how the
            output indices are ordered.

    Returns:
        Tensor product of Kronecker delta and Levi-Civita tensors.
    """
    delta_rules = []
    epsilon_rules = []
    for t in tp.components:
        if isinstance(t, Delta):
            delta_rules.append(t.indices)
        elif isinstance(t, Epsilon):
            epsilon_rules.append(t.indices)
        else:
            raise ValueError(f"Unknown tensor type: {type(t)}")

    left = ",".join(delta_rules + epsilon_rules)

    # Since the tensors only consists of delta and epsilon, the left rule should be OK,
    # but the right rule should be ordered according to the mode.
    right = "".join(delta_rules + epsilon_rules)
    lower = sorted([c for c in right if c.islower()])
    upper = sorted([c for c in right if c.isupper()])
    if mode == "G" or mode == "S":
        right = "".join(upper + lower)
    elif mode == "H":
        right = "".join(lower + upper)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rule = left + "->" + right

    d = dij()
    e = eijk()
    deltas = [d for _ in range(len(delta_rules))]
    epsilons = [e for _ in range(len(epsilon_rules))]
    data = deltas + epsilons

    product = torch.einsum(rule, *data)

    # multiply factor
    product = product * float(tp.factor)

    return product


def evaluate_tensors(tensors: LinearCombination, mode: str) -> Tensor:
    """
    Evaluate the tensor product of Kronecker delta and Levi-Civita tensors to get
    numerical values.
    """

    # Evaluate each tensor product
    output = 0
    for tp in tensors.components:
        if isinstance(tp, TensorProduct):
            output = output + tp_delta_epsilon(tp, mode)
        else:
            raise ValueError(f"Unknown tensor type: {type(tp)}")

    return output


def extract(H: LinearCombination, T: Tensor) -> Tensor:
    r"""
    Evaluate X^p,j = H^p(j|n) \odot^n T(n).

    Args:
        H:
        T:

    Returns:
    """

    d = dij()
    e = eijk()

    # TODO, we can use evaluate_tensors() in ghs.py to do the below?
    # Evaluate H and then tensor product with T
    output = []
    for tp in H:
        # create contraction rule
        indices = [t.indices for t in tp]
        delta_epsilon_rule = ",".join(indices)
        X_rule = "".join(sorted([i for i in "".join(indices) if i.islower()]))

        # For odd n-j and j != 0 the index for tau will appear twice, they should be
        # removed for the S rule. tau will be in epsilon and will be in E, see table
        # 4 in the writeup.
        # TODO, but double appearance can be eliminated, since one tau is in delta
        #  and the other tau is in epsilon. If we use simplify_linear_combination() to
        #  G at the top of the function, then we can get rid of the double appearance.
        #  and remove the checking on double appearance.
        upper = "".join([i for i in "".join(indices) if i.isupper()])
        T_rule = "".join(sorted([s for s, n in Counter(upper).items() if n == 1]))

        rule = f"{delta_epsilon_rule},{T_rule}->{X_rule}"

        # get delta and epsilon tensors for contraction
        delta_epsilon = []
        seen_epsilon = False
        for comp in tp:
            if isinstance(comp, Delta):
                delta_epsilon.append(d)
            elif isinstance(comp, Epsilon):
                if seen_epsilon:
                    raise ValueError("Only one epsilon tensor is allowed.")
                else:
                    seen_epsilon = True
                delta_epsilon.append(e)
            else:
                # tp only consists of delta and epsilon tensors
                raise ValueError(f"Unexpected type. {type(comp)}")

        # TODO, the rules tensor product epsilons and deltas can be precomputed and
        #  summed up. Then, we only need a single contraction.
        #
        # perform the contraction
        X = float(tp.factor) * torch.einsum(rule, *delta_epsilon, T)
        output.append(X)

    return torch.stack(output).sum(dim=0)


def embed(j: int, G: LinearCombination, X: Tensor = None, seed: int = 35) -> Tensor:
    r"""
    Evaluate S(n) = G(n|j) \odot^n X(j).

    Recall, in G, lower case indices are for r1, r2, ..., rj, and upper case indices
    are for s1, s2, ..., sn.

    Args:
        G: the contraction rule.
        X: the natural tensor X(j) to contract with G. If None, a random one is created.

    Return:
        S(n) in the space n.
    """

    if X is None:
        torch.manual_seed(seed)
        X = torch.randn(3**j).reshape([3] * j)
        X = symmetrize_and_remove_trace(X)

    d = dij()
    e = eijk()

    # TODO, we can use evaluate_tensors() in ghs.py to do the below?
    # Evaluate H and then tensor product with T
    output = []
    for tp in G:
        # create contraction rule
        indices = [t.indices for t in tp]
        delta_epsilon_rule = ",".join(indices)
        X_rule = "".join(sorted([i for i in "".join(indices) if i.islower()]))

        # For odd n-j and j != 0 the index for tau will appear twice, they should be
        # removed for the S rule. tau will be in epsilon and will be in E, see table
        # 4 in the writeup.
        # TODO, but double appearance can be eliminated, since one tau is in delta
        #  and the other tau is in epsilon. If we use simplify_linear_combination() to
        #  G at the top of the function, then we can get rid of the double appearance.
        #  and remove the checking on double appearance.
        upper = "".join([i for i in "".join(indices) if i.isupper()])
        S_rule = "".join(sorted([s for s, n in Counter(upper).items() if n == 1]))

        rule = f"{delta_epsilon_rule},{X_rule}->{S_rule}"

        # get delta and epsilon tensors for contraction
        delta_epsilon = []
        seen_epsilon = False
        for comp in tp:
            if isinstance(comp, Delta):
                delta_epsilon.append(d)
            elif isinstance(comp, Epsilon):
                if seen_epsilon:
                    raise ValueError("Only one epsilon tensor is allowed.")
                else:
                    seen_epsilon = True
                delta_epsilon.append(e)
            else:
                raise ValueError("Unexpected type.")

        # TODO, the rules for tensor product epsilons and deltas can be precomputed and
        #  summed up. Then, we only need a single contraction.
        # perform the contraction
        S = float(tp.factor) * torch.einsum(rule, *delta_epsilon, X)
        output.append(S)

    return torch.stack(output).sum(dim=0)


def embed2(G: LinearCombination, X: Tensor) -> Tensor:
    r"""
    Evaluate S(n) = G(n|j) \odot^n X(j).

    In G, lower case indices are for r1, r2, ..., rj, and upper case indices are for
    s1, s2, ..., sn. Here, the lower indices are to be contracted away.

    Args:
        G: the contraction rule.
        X: the natural tensor X(j) to contract with G.

    Return:
        S(n) in the space n.
    """
