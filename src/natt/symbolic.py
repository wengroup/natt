"""
Symbolic representation of cartesian tensor, product of cartesian tensors, and linear
combinations of them.
"""

from collections import Counter, defaultdict
from fractions import Fraction
from typing import Union


class CartesianTensor:
    """
    A general Cartesian tensor T_ij...k.

    This allows for pairs of repeated indices, e.g. ii, jj, etc.

    Args:
        indices: The indices of the tensor.
        factor: The scalar factor multiplied to the tensor, default is 1.
        symbol: The symbol of the tensor, default is "T".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "T"):
        self._check_indices(indices)
        self._indices = indices

        if isinstance(factor, int):
            self._factor = Fraction(factor)
        elif isinstance(factor, Fraction):
            self._factor = factor
        else:
            raise ValueError("The `factor` must be a Fraction object")

        self._symbol = symbol

    @property
    def indices(self):
        return self._indices

    @property
    def symbol(self):
        return self._symbol

    @property
    def factor(self):
        return self._factor

    @property
    def rank(self):
        return len(self.indices)

    def permute_indices(
        self, permute: list[int], factor: int | Fraction = 1
    ) -> "CartesianTensor":
        """
        Permute the indices of the tensor.

        For example, if the tensor is T_ijk and permute is [1, 2, 0], the new tensor
        is T_jki.

        Args:
            permute: The new order of the indices.
            factor: The factor to be multiplied to the tensor, default is 1.

        Returns:
            The tensor with permuted indices.
        """
        indices = "".join([self.indices[i] for i in permute])
        if Counter(self.indices) != Counter(indices):
            raise ValueError("The new indices must contain the same indices")

        return self.__class__(indices, factor * self.factor, self.symbol)

    @staticmethod
    def _check_indices(indices: str):
        """Check indices are repeated at most twice.

        Only check for alphabetic indices, not integer indices. When the indices are
        evaluated, there can be many repeated values of the same integer.
        """
        for i in indices:
            if i.isalpha() and indices.count(i) > 2:
                raise ValueError("Indices can be repeated at most twice")

    def __contains__(self, item):
        return item in self.indices

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, index):
        return self.indices[index]

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return self.__class__(self.indices, self.factor * other, self.symbol)

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __eq__(self, other):
        idx2pos_self = defaultdict(list)
        idx2pos_other = defaultdict(list)
        for p, i in enumerate(self.indices):
            idx2pos_self[i].append(p)
        for p, i in enumerate(other.indices):
            idx2pos_other[i].append(p)

        # check if the indices are the same, ignoring repeated indices
        idx2pos_self = {k: v for k, v in idx2pos_self.items() if len(v) == 1}
        idx2pos_other = {k: v for k, v in idx2pos_other.items() if len(v) == 1}
        if idx2pos_self != idx2pos_other:
            return False

        return self.symbol == other.symbol and self.factor == other.factor

    def __str__(self):
        if self.factor == Fraction(1):
            return f"{self.symbol}_{self.indices}"
        else:
            return f"({self.factor}) {self.symbol}_{self.indices}"


class Scalar(CartesianTensor):
    """
    Zero rank tensor, a scalar.
    """

    def __init__(self, factor: int | Fraction = 1):
        super().__init__("", factor, "Const")

    def __str__(self):
        return f"{self.symbol}({self.factor})"


class Zero(Scalar):
    """The scalar zero."""

    def __init__(self):
        super().__init__(0)


class Delta(CartesianTensor):
    r"""
    The Kronecker delta tensor \delta.

    Args:
        indices: The indices of the delta tensor.
        factor: The scalar factor multiplied to the delta tensor, default is 1.
        symbol: The symbol of the delta tensor, default is "δ".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "δ"):
        assert len(indices) == 2, "The delta tensor must have two indices"
        super().__init__(indices, factor, symbol)

    def __eq__(self, other):
        # should not compare symbol, but just make sure it is a Delta tensor
        if not isinstance(other, Delta):
            return False

        if not self.factor == other.factor:
            return False

        if self.indices != other.indices and self.indices != other.indices[::-1]:
            return False

        return True


class Epsilon(CartesianTensor):
    r"""
    The Levi-Civita tensor \epsilon.

    Args:
        indices: The indices of the epsilon tensor.
        factor: The scalar factor multiplied to the epsilon tensor, default is 1.
        symbol: The symbol of the epsilon tensor, default is "ε".
    """

    def __init__(self, indices: str, factor: int | Fraction = 1, symbol: str = "ε"):
        assert len(indices) == 3, "The epsilon tensor must have three indices"
        super().__init__(indices, factor, symbol)

    def __eq__(self, other):
        # should not compare symbol, but just make sure it is a Epsilon tensor
        if not isinstance(other, Epsilon):
            return False

        if not self.factor == other.factor:
            return False

        # even permutations of indices are equal
        indices = other.indices
        if (
            self.indices != indices
            and self.indices != indices[1] + indices[2] + indices[0]
            and self.indices != indices[2] + indices[0] + indices[1]
        ):
            return False

        return True


class TensorProduct:
    """
    A representation of a tensor product of multiple tensors.

    Args:
        tensors: The constituting tensors.
        factor: Additional factor multiplied to the tensor product. Each tensor in the
            product can have its own factor. So, the overall factor is the product of
            the factors of the constituting tensors and this factor.
        combine_scalars: If True, combine scalars in the tensor product. For example, all
            scalars will be combined into the factor of the tensor product, and the scalars
            will be removed from the tensor product. Default is True.
    """

    def __init__(
        self,
        *tensors: CartesianTensor | Epsilon | Delta | Scalar,
        factor: int | Fraction = 1,
        combine_scalars: bool = True,
    ):
        self.combine_scalars = combine_scalars

        if not combine_scalars:
            self._factor = factor
            self._tensors = list(tensors)
        else:
            # get overall factor
            for t in tensors:
                factor *= t.factor
            self._factor = factor

            if self._factor == 0:
                self._tensors = [Zero()]

            else:
                # set the factor of the constituting tensors to 1
                self._tensors = []
                for t in tensors:
                    if isinstance(t, Scalar):
                        pass  # scalars already been included in the factor
                    else:
                        self._tensors.append(t.__class__(t.indices, 1, t.symbol))

    @property
    def factor(self):
        """The overall factor of the tensor product."""
        return self._factor

    @property
    def components(self):
        """Constituting tensors of the product, without considering the factor."""
        return self._tensors

    @property
    def indices(self):
        """The indices of the tensor product."""
        return "".join([t.indices for t in self._tensors])

    def permute_indices(
        self, permute: list[int], factor: int | Fraction = 1
    ) -> "TensorProduct":
        """
        Permute the indices of the tensor product.

        For example,
        if the tensor is D_ab T_ijk and permute is [2,4,0,1,3], the new tensor is
        D_ik T_abj.

        Args:
            permute: The new order of the indices.
            factor: Additional factor to be multiplied to the tensor, default is 1.

        Returns:
            The tensor product with permuted indices.
        """

        indices = self.indices

        i = 0
        tensors = []
        for t in self._tensors:
            perm = permute[i : i + len(t.indices)]
            permuted_indices = "".join([indices[p] for p in perm])
            nt = t.__class__(permuted_indices, t.factor, t.symbol)
            tensors.append(nt)
            i += len(t.indices)

        return TensorProduct(*tensors, factor=factor * self.factor)

    def canonize(self):
        """
        Canonicalize the tensor product.

        1. The canonized form will be like: delta_... epsilon_... T_...
        2. For a delta, the indices will be ordered, e.g. delta_ji -> delta_ij
        3. For an epsilon, the indices will be shifted such that the first index is the
           smallest one. e.g. epsilon_jik -> epsilon_ikj, epsilon_jki -> epsilon_ijk,
           while keeping the relative order of the indices.
        4. For a general tensor, no ordering of the indices are performed.
        5. For tensors of the same type (e.g. two delta tensors), they will be ordered
           by their first indices. For example, delta_ij delta_ab -> delta_ab delta_ij,
           since a < i. This is similarly for the epsilon tensors and general tensors.

        Returns:
            A canonized tensor product.
        """
        # TODO, this assumes the factor of each component is 1, which may not be true
        #  in general.

        deltas = []
        epsilons = []
        general = []
        for t in self._tensors:
            if isinstance(t, Delta):
                deltas.append(t)
            elif isinstance(t, Epsilon):
                epsilons.append(t)
            else:
                general.append(t)

        # canonize deltas
        all_indices = ["".join(sorted(t.indices)) for t in deltas]
        all_indices = sorted(all_indices)
        deltas = [Delta(indices) for indices in all_indices]

        # canonize epsilons
        def sort_circular(s: str):
            """
            Shift the three indices of the epsilon tensor such that the first index is
            the smallest one.

            E.g. kij -> ijk
            """
            # Find the index of the smallest character
            min_index = s.index(min(s))
            # Rotate the string to bring the smallest character to the front
            return s[min_index:] + s[:min_index]

        all_indices = [sort_circular(t.indices) for t in epsilons]
        all_indices = sorted(all_indices)
        epsilons = [Epsilon(indices) for indices in all_indices]

        # canonize general tensors
        all_indices = sorted([t.indices for t in general])
        general = [CartesianTensor(indices) for indices in all_indices]

        # Create the tensor product
        tensors = deltas + epsilons + general
        return TensorProduct(*tensors, factor=self.factor)

    def __eq__(self, other: Union[CartesianTensor, "TensorProduct"]):
        if len(self) != len(other):
            return False

        if self.factor != other.factor:
            return False

        # compare symbol and indices of the constituting tensors
        if not str(self.canonize()) == str(other.canonize()):
            return False

        return True

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return self.__class__(*self._tensors, factor=self.factor * other)

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        return self._tensors[item]

    def __len__(self):
        return len(self._tensors)

    def __str__(self):
        rep = self.str_rep_without_factor()
        if self.factor >= 0:
            factor = f"+{self.factor}"
        else:
            factor = self.factor
        return f"{factor}{rep}"

    def str_rep_without_factor(self):
        """Get the string representation of the tensor product without the factor."""
        rep = ""
        for t in self._tensors:
            # scalars will be included in the factor, so we skip them here
            if not isinstance(t, Scalar):
                rep += f" {t.symbol}_{t.indices}"

        return rep


class LinearCombination:
    """A linear combination of Cartesian tensors or tensor Product."""

    def __init__(self, *tensors: CartesianTensor | Delta | Epsilon | TensorProduct):
        self._tensors = tensors

    @property
    def components(self):
        """The constituting tensor products."""
        return self._tensors

    def to_str_list(self, including_zero: bool = False) -> list[str]:
        """
        Convert the tensors to string representation.

        Args:
            including_zero: If True, include zero tensors in the output.
        """
        return [str(t) for t in self._tensors if including_zero or t.factor != 0]

    def __eq__(self, other: "LinearCombination"):
        # TODO, we just implement the case that the constituting tensors are the same
        #  and in the same order. Of course, this is not general.

        if len(self) != len(other):
            return False

        for x, y in zip(self._tensors, other._tensors):
            if x != y:
                return False

        return True

    def __len__(self):
        return len(self._tensors)

    def __iter__(self):
        return iter(self._tensors)

    def __getitem__(self, item):
        return self._tensors[item]

    def __add__(self, other: "LinearCombination"):
        return LinearCombination(*self._tensors, *other._tensors)

    def __radd__(self, other: "LinearCombination"):
        # Handle the sum() case. Note, sum([X]) is expanded as 0 + X
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other: int | Fraction):
        """Multiply the tensor by a scalar."""
        return LinearCombination(*[t * other for t in self._tensors])

    def __rmul__(self, other: int | Fraction):
        return self.__mul__(other)

    def __str__(self):
        str_rep = self.to_str_list(including_zero=False)

        return "  ".join(str_rep)
