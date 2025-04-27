import torch

from natt.utils import double_index, get_trace


def test_multi_double_index():
    assert double_index(2) == ["ab", "cd"]
    assert double_index(3, start=1) == ["bc", "de", "fg"]


def test_get_trace():
    T2 = torch.arange(9).reshape(3, 3).to(torch.float)
    T3 = torch.arange(27).reshape(3, 3, 3).to(torch.float)

    trace = get_trace(T2, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([12.0]))

    trace = get_trace(T3, i=0, j=1)
    assert torch.allclose(trace, torch.tensor([36.0, 39.0, 42.0]))

    trace = get_trace(T3, i=1, j=2)
    assert torch.allclose(trace, torch.tensor([12.0, 39.0, 66.0]))
