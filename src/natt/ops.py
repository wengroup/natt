"""
Operations on cartesian tensors.
"""

import itertools
from collections import Counter, defaultdict
from fractions import Fraction

from natt.symbolic import (
    CartesianTensor,
    Delta,
    Epsilon,
    LinearCombination,
    Scalar,
    TensorProduct,
    Zero,
)


def contract_with_delta(delta: Delta, tensor: CartesianTensor) -> CartesianTensor:
    r"""
    Contract a tensor with a delta tensor.

    For example,
    \delta_ij T_ijk -> T_iik
    \delta_ai T_ijk -> T_ajk

    Args:
        delta: The delta tensor.
        tensor: A Cartesian tensor.

    Returns:
        The contracted tensor.
    """
    # check at least one of the indices is in common
    if not (set(tensor.indices) & set(delta.indices)):
        raise ValueError("Delta tensor does not have common indices with the tensor")

    for p, i in enumerate(delta):
        if i in tensor:
            other = delta[1] if p == 0 else delta[0]
            return tensor.__class__(
                tensor.indices.replace(i, other), tensor.factor, tensor.symbol
            )

    raise ValueError("Delta tensor does not have common indices with the tensor")


def contract_with_epsilon(epsilon: Epsilon, tensor: CartesianTensor) -> TensorProduct:
    r"""
    Contract a tensor with an epsilon tensor.

    \epsilon_aij T_ijk...n
    \epsilon_abi T_ijk...n

    Args:
        epsilon: The epsilon tensor.
        tensor: A Cartesian tensor.

    Returns:
        The contracted tensor.
    """
    # check at least one of the indices is in common
    if not (set(tensor.indices) & set(epsilon.indices)):
        raise ValueError("Epsilon tensor does not have common indices with the tensor")
    return TensorProduct(epsilon, tensor)


def contract_epsilon_delta(epsilon: Epsilon, delta: Delta):
    r"""
    Contract an epsilon tensor with a delta tensor.

    For example,
    \epsilon_ijk \delta_ij = \epsilon_iik = 0
    \epsilon_ijk \delta_il = \epsilon_ljk

    At least one of the indices must be repeated in the delta tensor

    Args:
        epsilon: The epsilon tensor, given by three indices.
        delta: The delta tensor, given by a pair of indices.

    Returns:
        The contracted tensor.
    """
    if len(set(epsilon) & set(delta)) == 2:
        return Zero()

    return contract_with_delta(delta, epsilon)


def contract_two_epsilon(epsilon1: Epsilon, epsilon2: Epsilon):
    """
    Contract two epsilon tensors.

    This implements:
    1. e_ijk e_pqk = d_ip d_jq - d_iq d_jp
    2. e_ijk e_pjk = 2 d_ip
    3. e_ijk e_ijk = 6

    Args:
        epsilon1: The first epsilon tensor.
        epsilon2: The second epsilon tensor.

    Returns:
        The contracted delta tensor.
    """

    def canonicalize_one(eps, idx):
        """
        Canonicalize the order of the indices.

        Does not change relative order of the three indices, but put the provided index
        at the last position.

        For example,
            (ijk, k) -> ijk
            (ijk, j) -> kij
            (ijk, i) -> jki
        """
        if idx == eps[0]:
            indices = eps[1] + eps[2] + eps[0]
            return Epsilon(indices, eps.factor)
        elif idx == eps[1]:
            indices = eps[2] + eps[0] + eps[1]
            return Epsilon(indices, eps.factor)
        else:
            return eps

    def canonicalize_two(eps, idx1, idx2):
        """
        Canonicalize the order of the indices.

        Put idx1 at the second position, idx2 at the third position.
        If relative order of the indices is changed, the sign is flipped.

        For example,
        (ijk, i, j) -> kij
        (ijk, j, i) -> -kij
        (ijk, j, k) -> ijk
        (ijk, k, j) -> -ijk
        (ijk, k, i) -> jki
        (ijk, i, k) -> -jki
        """
        if idx1 == eps[0] and idx2 == eps[1]:
            indices = eps[2] + eps[0] + eps[1]
            sign = 1

        elif idx1 == eps[1] and idx2 == eps[2]:
            indices = eps[0] + eps[1] + eps[2]
            sign = 1

        elif idx1 == eps[2] and idx2 == eps[0]:
            indices = eps[1] + eps[2] + eps[0]
            sign = 1

        elif idx1 == eps[1] and idx2 == eps[0]:
            indices = eps[2] + eps[1] + eps[0]
            sign = -1

        elif idx1 == eps[2] and idx2 == eps[1]:
            indices = eps[0] + eps[2] + eps[1]
            sign = -1

        elif idx1 == eps[0] and idx2 == eps[2]:
            indices = eps[1] + eps[0] + eps[2]
            sign = -1

        else:
            raise ValueError("Invalid indices")

        return Epsilon(indices, sign * eps.factor)

    # get number of repeated indices
    repeated = set(epsilon1) & set(epsilon2)

    if len(repeated) == 3:
        # Canonicalize the indices of epsilon2 such that it has the same indices as
        # epsilon1, and change the sign if necessary
        idx1 = epsilon1.indices[1]
        idx2 = epsilon1.indices[2]
        eps2 = canonicalize_two(epsilon2, idx1, idx2)
        factor = epsilon1.factor * eps2.factor
        return Scalar(6 * factor)

    elif len(repeated) == 2:
        idx1, idx2 = sorted(repeated)
        eps1 = canonicalize_two(epsilon1, idx1, idx2)
        eps2 = canonicalize_two(epsilon2, idx1, idx2)
        return Delta(eps1[0] + eps2[0], 2 * eps1.factor * eps2.factor)

    elif len(repeated) == 1:
        idx = repeated.pop()
        eps1 = canonicalize_one(epsilon1, idx)
        eps2 = canonicalize_one(epsilon2, idx)
        d1 = Delta(eps1[0] + eps2[0])
        d2 = Delta(eps1[1] + eps2[1])
        d3 = Delta(eps1[0] + eps2[1])
        d4 = Delta(eps1[1] + eps2[0])
        return LinearCombination(
            TensorProduct(d1, d2), TensorProduct(d3, d4, factor=-1)
        )

    else:
        raise ValueError("No repeated indices")


def symmetrize(
    tensor: CartesianTensor | TensorProduct, indices: str = None
) -> LinearCombination:
    """
    Symmetrize a tensor.

    Args:
        tensor: A Cartesian tensor.
        indices: The indices to symmetrize over. If None, all non-repeated indices are symmetrized.

    Returns:
        A list of tensors, each with a different permutation of the indices, each tensor
        is normalized by the number of permutations.
    """

    if indices is None:
        indices = [i for i, c in Counter(tensor.indices).items() if c == 1]
    else:
        # check provided indices are not repeated in the tensor
        for i in indices:
            if tensor.indices.count(i) != 1:
                raise ValueError(f"Index {i} must appear exactly once in the tensor")

    moveable_pos = [i for i, x in enumerate(tensor.indices) if x in indices]

    all_tensors = []
    permutations = list(itertools.permutations(moveable_pos))
    for perm in permutations:
        # candidate permute
        permute = list(range(len(tensor.indices)))
        # update permute positions
        for i, p in zip(moveable_pos, perm):
            permute[i] = p

        t = tensor.permute_indices(permute, factor=Fraction(1, len(permutations)))
        all_tensors.append(t)

    return LinearCombination(*all_tensors)


def is_zero(tensors: LinearCombination) -> bool:
    """
    Check whether a linear combination of tensors is zero.
    """
    # TODO, for now, we just check if the str representation of the positive ones
    #  and the negative ones are the same
    positive = []
    negative = []
    for t in tensors:
        if t.factor == 0:
            continue
        elif t.factor > 0:
            positive.append(t)
        else:
            negative.append(t)

    # flip the sign of the negative ones
    negative = [-1 * t for t in negative]

    pos_count = Counter([str(t) for t in positive])
    neg_count = Counter([str(t) for t in negative])

    return pos_count == neg_count


def simplify_epsilon(
    product: TensorProduct,
) -> tuple[TensorProduct | LinearCombination, bool]:
    """
    Evaluate product of two epsilon tensors in a tensor product.
    """

    epsilon_pos = [i for i, t in enumerate(product) if isinstance(t, Epsilon)]

    for i, j in itertools.combinations(epsilon_pos, 2):
        # check if they share at least one index
        if set(product[i].indices) & set(product[j].indices):
            out = contract_two_epsilon(product[i], product[j])

            # Remaining components after the two epsilon tensors are contracted
            remaining = [t for p, t in enumerate(product) if p not in [i, j]]

            # three identical indices, resulting in a scalar
            if isinstance(out, Scalar):
                return (
                    TensorProduct(out, *remaining, factor=product.factor),
                    True,
                )

            # two identical indices, resulting in a delta tensor
            elif isinstance(out, Delta):
                return (
                    TensorProduct(out, *remaining, factor=product.factor),
                    True,
                )

            # one identical index, resulting in linear combination of tensor products
            # of delta tensors e_ijk e_ilm = d_jl d_km - d_jm d_kl
            elif isinstance(out, LinearCombination):
                all_tp = []
                for tp in out:
                    new_tp = TensorProduct(
                        *tp.components,
                        *remaining,
                        factor=tp.factor * product.factor,
                    )
                    all_tp.append(new_tp)

                return LinearCombination(*all_tp), True

            else:
                raise ValueError("Invalid output")

    return product, False


def simplify_delta(product: TensorProduct) -> tuple[TensorProduct, bool]:
    """
    Evaluate product of a delta tensor with another tensor in a tensor product.

    This will contract all possible delta tensors in the product.

    Tensors like delta_ii will evaluated to 3.
    """
    # Positions of delta tensors in the product
    delta_pos = [i for i, t in enumerate(product) if isinstance(t, Delta)]

    # This is the dictionary of components, which will be updated as we contract
    components = {i: t for i, t in enumerate(product)}

    performed = False

    n = len(components)

    while delta_pos:
        i = delta_pos.pop()
        delta = components[i]

        new_components = components.copy()

        # delta_ii
        if delta.indices[0] == delta.indices[1]:
            out = Scalar(3)

            # Remove the delta from the components
            new_components.pop(i)

            # Add the resulting scalar to the components
            new_components[n] = out

            performed = True
            n += 1
            components = new_components
            continue

        # delta_ij
        for j, t in components.items():
            if j == i:
                continue

            # Perform contraction if delta and t share common indices
            if set(delta.indices) & set(t.indices):
                if isinstance(t, Epsilon):
                    out = contract_epsilon_delta(t, delta)
                else:
                    out = contract_with_delta(delta, t)

                # Remove the delta and t from the components
                new_components.pop(i)
                new_components.pop(j)

                # Add the contracted result, create a new index
                new_components[n] = out

                # Update delta_pos: if t is a delta tensor, remove it; if the contracted
                # out is a delta tensor, add it
                if isinstance(t, Delta):
                    delta_pos.remove(j)
                if isinstance(out, Delta):
                    delta_pos.append(n)

                performed = True
                n += 1
                break

        components = new_components

    if performed:
        return TensorProduct(*components.values(), factor=product.factor), True
    else:
        return product, False


def simplify(tp: TensorProduct) -> LinearCombination:
    """
    Simplify a tensor product by apply delta and epsilon rules.

    The simplification is done iteratively until no more simplification can be done.
    Zeros resulting from the simplification are removed.

    For example,
    d_ij e_imn d_nq T_qpr -> e_jmq T_qpr
    """

    # Iteratively simplify the tensor product
    performed = True
    simplified = LinearCombination(tp)
    while performed:
        double_epsilon = None
        double_epsilon_pos = None
        new_simplified = []
        performed = []
        for i, tp in enumerate(simplified):

            # Step 1: simplify epsilon first
            sim, perf = simplify_epsilon(tp)

            # Double epsilon contraction will return a LinearCombination
            if isinstance(sim, LinearCombination):
                if double_epsilon is not None:
                    raise ValueError(
                        "Double epsilon simplification already done. The current "
                        "Implementation does not support multiple double epsilon."
                    )
                double_epsilon_pos = i
                double_epsilon = sim

            # Step 2: If no epsilon simplification performed, then simplify delta
            if not perf:
                sim, perf = simplify_delta(tp)

            new_simplified.append(sim)
            performed.append(perf)

        # Double epsilon contraction will return a LinearCombination of sum of two
        # tensor products. We need to expand it to be produced with other components
        # in the input tp.
        if double_epsilon is not None:
            linear_comb = []
            for de in double_epsilon:
                # list of tensor products
                comb = new_simplified.copy()
                comb[double_epsilon_pos] = de
                new_tp = multiply(*comb)
                linear_comb.append(new_tp)
        else:
            linear_comb = new_simplified

        # prepare for the next iteration
        performed = any(performed)
        simplified = LinearCombination(*linear_comb)

    # Step 3: remove zeros
    simplified = LinearCombination(*[t for t in simplified if t.factor != 0])

    return simplified


def simplify_2(tensor: LinearCombination) -> LinearCombination:
    """Simplify a linear combination of tensors.
    1. Applying delta and epsilon rules.
    2. Removing zero tensors or tensor products.
    """
    simplified = []
    for t in tensor:
        if t.factor == 0:  # remove zeros
            continue
        if isinstance(t, CartesianTensor):
            simplified.append(t)
        elif isinstance(t, TensorProduct):
            out = simplify(t)
            simplified.extend(out)
        else:
            raise ValueError("Unexpected type")

    # Combine tensor products that are of the same form
    categorized = defaultdict(list)
    for t in simplified:
        if isinstance(t, CartesianTensor):
            raise ValueError(
                "@not implemented, should modify the `for tp_list in "
                "categorized.values()` block too"
            )
        elif isinstance(t, TensorProduct):
            t = t.canonize()
            rep = t.str_rep_without_factor()
            categorized[rep].append(t)
        else:
            raise ValueError("Unexpected type")

    # TODO, the logic is reimplemented in `combine_terms()`, but it is much
    #  simple here
    lin_comb = []
    for tp_list in categorized.values():
        factor = sum(tp.factor for tp in tp_list)
        components = tp_list[0].components
        tp = TensorProduct(*components, factor=factor)
        lin_comb.append(tp)
    simplified = LinearCombination(*lin_comb)

    return LinearCombination(*simplified)


if __name__ == "__main__":
    ###
    # Example 1
    # check e_aij T_ijkl, e_aij T_ikjl, and e_aij T_kijl are linearly dependent

    # basic tensors
    e = Epsilon("aij")
    tp1 = contract_with_epsilon(e, CartesianTensor("ijkl"))
    tp2 = contract_with_epsilon(e, CartesianTensor("ikjl"))
    tp3 = contract_with_epsilon(e, CartesianTensor("kijl"))

    # symmetrize the tensors
    s1 = symmetrize(tp1, indices="akl")
    s2 = symmetrize(tp2, indices="akl")
    s3 = symmetrize(tp3, indices="akl")

    tensors = s1 + -1 * s2 + s3

    evaluated = tensors.evaluate(
        {
            "a": "1",
            "i": "2",
            "j": "3",
            "k": "2",
            "l": "3",
        }
    )

    evaluated_non_zero = LinearCombination(*[t for t in evaluated if t.factor != 0])

    out = is_zero(evaluated_non_zero)

    print("Tensors", tensors)
    print("number of non-zeros:", len(evaluated_non_zero))
    print("evaluated non-zeros", evaluated_non_zero)
    print("Dependence:", out)


def multiply(
    *tensors: CartesianTensor | TensorProduct, factor: int | Fraction = 1
) -> TensorProduct:
    """
    Multiple tensors, tensor products to create a new tensor product.

    Args:
        *tensors: the tensors or tensor products to multiply.
        factor: Additional factor to be multiplied to the tensor product, default is 1.

    Returns:
        The new tensor product.
    """
    new_tensors = []
    factor = Fraction(factor)
    for t in tensors:
        if isinstance(t, CartesianTensor):
            new_tensors.append(t)
        elif isinstance(t, TensorProduct):
            new_tensors.extend(t.components)
            factor *= t.factor
        else:
            raise ValueError("Unexpected type")

    tp = TensorProduct(*new_tensors, factor=factor)

    return tp


def multiply_2(
    *tensors: CartesianTensor | TensorProduct | LinearCombination,
    factor: int | Fraction = 1,
) -> LinearCombination:
    """
    Multiply tensors, tensor products, linearly combined tensors to create a new Tensors object.

    Args:
        *tensors: the tensors or tensor products to multiply.
        factor: Additional factor to be multiplied to the tensor product, default is 1.

    Returns:
        The new tensor product.
    """
    # First, convert input to Tensors
    new_tensors = []
    for t in tensors:
        if isinstance(t, (CartesianTensor, TensorProduct)):
            new_tensors.append(LinearCombination(t))
        elif isinstance(t, LinearCombination):
            new_tensors.append(t)
        else:
            raise ValueError("Unexpected type")

    all_tp = []
    for prod in itertools.product(*new_tensors):
        all_tp.append(multiply(*prod, factor=factor))

    return LinearCombination(*all_tp)
