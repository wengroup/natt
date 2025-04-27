import torch

from natt.symmetrize import (
    get_permutations,
    get_permutations_2,
    remove_trace,
    remove_trace_rule,
    symmetrize,
)
from natt.utils import is_symmetric


def test_get_permutations():
    assert get_permutations("aaaa") == [[0, 1, 2, 3]]
    assert get_permutations("aaaa", start_dim=2) == [[0, 1, 2, 3, 4, 5]]

    ref = [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 2, 4],
        [0, 1, 3, 4, 2],
        [0, 3, 1, 2, 4],
        [0, 3, 1, 4, 2],
        [0, 3, 4, 1, 2],
        [3, 0, 1, 2, 4],
        [3, 0, 1, 4, 2],
        [3, 0, 4, 1, 2],
        [3, 4, 0, 1, 2],
    ]
    perms = get_permutations("aaabb")
    assert perms == ref

    perms = get_permutations("aaabb", start_dim=2)
    assert perms == [[0, 1] + [2 + i for i in sub] for sub in ref]


def test_get_permutations_2():
    perms = get_permutations_2(m=2, num_delta=1)
    assert perms == [[0, 1]]


def test_symmetrize(T2, T3, T4):
    for t in [T2, T3, T4]:
        for start_dim in range(2):
            sym = symmetrize(t, start_dim)
            is_symmetric(sym, start_dim)


def test_remove_trace_rule():
    rule = remove_trace_rule(5, 2)
    assert rule == "...aabbc,de,fg->...cdefg"


def test_remove_trace(T2, T3, T4):
    """
    Test traceless part, not the symmetric part.
    """
    # second rank tensor
    t2 = symmetrize(T2)
    t2_1 = remove_trace(t2, start_dim=0)
    assert torch.einsum("ii", t2_1) == 0.0

    t2_2 = t2.reshape(1, 1, 3, 3)
    t2_tl = remove_trace(t2_2, start_dim=2)
    assert t2_tl.shape == t2_2.shape

    assert torch.einsum("...ii", t2_tl) == 0.0

    # third rank tensor
    t3 = symmetrize(T3)
    t3 = t3.reshape(1, 1, 3, 3, 3)
    t3_tl = remove_trace(t3, start_dim=2)
    assert t3_tl.shape == t3.shape

    for rule in ["...iij", "...iji", "...jii"]:
        out = torch.einsum(rule, t3_tl)
        assert torch.allclose(out, torch.zeros(3), atol=1e-5)

    # fourth rank tensor
    t4 = symmetrize(T4)
    t4 = t4.reshape(1, 1, 3, 3, 3, 3)
    t4_tl = remove_trace(t4, start_dim=2)
    assert t4_tl.shape == t4.shape

    for rule in ["...iijk", "...ijik", "...ijki", "...jiik", "...jiki", "...jkii"]:
        out = torch.einsum(rule, t4_tl)
        assert torch.allclose(out, torch.zeros(3, 3), atol=1e-4)
