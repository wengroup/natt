"""
Example to convert ordinary tensor and natural tensor back and forth.
"""

import torch

from natt.GHS import get_G_H_S
from natt.sym import symmetrize
from natt.utils import is_symmetric, is_symmetric_traceless, is_traceless


def convert(rank=3, symmetry: str = None):
    """Converting between ordinary tensor and natural tensor.

    For an ordinary tensor T, convert it to natural tensors X (which will be
    symmetric traceless), embed each natural tensor X to be in the original space T' (
    which is not symmetric traceless).
    Or, T' can be directly obtained by applying S to T.

    Args:
        rank: rank of the tensor T
        symmetry: symmetry of the tensor T, e.g. "ij=ji" for a rank-2 tensor.
    """

    torch.manual_seed(35)

    # Create a random tensor T of the given rank and symmetry
    T = torch.randn(*[3] * rank)
    if symmetry is not None:
        T = symmetrize(T, symmetry)

    GHS = get_G_H_S(rank, symmetry)

    all_T_prime = []
    for j, out_j in GHS.items():

        for p, (H, G, S) in enumerate(zip(out_j["H"], out_j["G"], out_j["S"])):
            # X = H T
            X = torch.einsum(H["rule"], H["numerical"], T)
            assert is_symmetric_traceless(
                X
            ), "X is not symmetric traceless for j={j}, p={p}"

            # T' = G X
            T_p_1 = torch.einsum(G["rule"], G["numerical"], X)
            print(
                f"T' (j={j}, p={p}), symmetric:",
                is_symmetric(T_p_1),
                "traceless:",
                is_traceless(T_p_1),
            )

            # T' = S T
            T_p_2 = torch.einsum(S["rule"], S["numerical"], T)

            # T_p_1 and T_p_2 should be equal
            assert torch.allclose(T_p_1, T_p_2, rtol=1e-5, atol=1e-6), (
                f"T_p_1 and T_p_2 are not equal for j=" f"{j}, p={p}"
            )

            all_T_prime.append(T_p_1)

    sum_T_prime = torch.sum(torch.stack(all_T_prime), dim=0)

    assert torch.allclose(sum_T_prime, T)


if __name__ == "__main__":
    convert(rank=2)
    print("=" * 40)
    convert(rank=3)
