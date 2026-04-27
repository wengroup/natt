# natt

`natt` is a Python package for constructing, analyzing, and embedding **natural tensors** — symmetric traceless tensors that form irreducible representations of the rotation group O(3).
It provides tools to decompose arbitrary Cartesian tensors into natural tensor components and to embed natural tensors back into the original Cartesian tensor space.

## Installation

It is recommended to create a virtual environment first (e.g. using conda) before installing `natt`:

```bash
git clone https://github.com/wengroup/natt.git
cd natt
pip install -e .
```

## Usage

The example below converts a rank-2 Cartesian tensor into its natural tensor components and reconstructs the original tensor.

```python
import torch
from natt.GHS import get_G_H_S

# Create a rank-2 tensor
T = torch.arange(9, dtype=torch.float).reshape(3, 3)

# Get the G, H, S matrices for the decomposition
rank = 2
symmetry=None # `None` means no additional symmetry; `ij=ji` means symmetric tensors... 
GHS = get_G_H_S(rank, symmetry)

all_T_prime = []
for j, out_j in GHS.items():
    for p, (H, G, S) in enumerate(zip(out_j["H"], out_j["G"], out_j["S"])):
        # Extract natural tensor component X = H T  (symmetric traceless)
        X = torch.einsum(H["rule"], H["numerical"], T)

        # Embed back: T' = G X  (contribution to T from angular momentum j)
        T_prime = torch.einsum(G["rule"], G["numerical"], X)
        all_T_prime.append(T_prime)

        print('-'*10 + f"j={j}, p={p}"+'-'*10)
        print(f"X=\n{X}")
        print(f"T_prime=\n{T_prime}")

# Sum of all embedded components should reconstruct the original tensor
T_sum = torch.stack(all_T_prime).sum(dim=0)
print(f"Original T=\n{T}")
print(f"Sum of T_prime=\n{T_sum}")

assert torch.allclose(T_sum, T, atol=1e-6)
```

See [`example/get_natural_tensor.py`](example/get_natural_tensor.py) for a complete working example.


## Citation
Chen, Q., Pattamatta, A.S.L., Wang, B., Srolovitz, D.J. and Wen, M., 2026. Atomistic Machine Learning with Irreducible Cartesian Natural Tensors. arXiv preprint arXiv:2510.04015.

```latex
@article{chen2026atomistic,
  title   = {Atomistic Machine Learning with Irreducible Cartesian Natural Tensors},
  author  = {Chen, Qun and Pattamatta, ASL and Wang, Boyu and Srolovitz, David J and Wen, 
  Mingjian},
  journal = {arXiv preprint arXiv:2510.04015},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.04015},
}
```
