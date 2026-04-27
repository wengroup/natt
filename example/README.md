# Generating Projector YAML Files

`natt` can generate three types of projectors, each stored as a YAML file.
The `generate_*.py` scripts in this directory generate the corresponding YAML files:

- `generate_tensor_product_projector.py` generates `tensor_product_projector.yaml`
- `generate_decomposition_and_reconstruction_projector.py` generates `decomposition_and_reconstruction_projector.yaml`
- `generate_unit_vector_projector.py` generates `unit_vector_projector.yaml`

Run a script directly to generate the corresponding YAML file, e.g.:

```bash
python generate_tensor_product_projector.py
```

Each script allows configuring which ranks or tensor types to include; edit the
parameters at the top of the script before running.

The three projector types are described below.


# unit_vector_projector.yaml

This projector `H` maps a unit vector to a natural tensor of a given rank, i.e. it
extracts the rank-l irreducible component from the outer products of a unit vector.
The file contains the symbolic and numerical values of `H` along with the einsum rule
to apply it.

The data is organized in the following way:

{rank-normalization:
    {"H_symbolic": symbolic expression for the projector H,
    "H_numerical": numerical values of the projector H,
    "rule": einsum rule to apply the projector H to get the natural tensor from the unit vector
    }
}

- In "rank-normalization", rank is a positive integer that indicates the rank of the
  natural tensor to be constructed, and normalization can be `none` or `unity`,
  indicating whether the projector H is normalized or not.


# tensor_product_projector.yaml

This projector `H` performs the Clebsch-Gordan-like coupling of two natural tensors of
ranks l1 and l2 into a new natural tensor of rank l3.
The file contains the symbolic and numerical values of `H` along with the einsum rule
to apply it.

The data is organized in the following way:

{l1-l2-l3-normalization:
    {"H_symbolic": symbolic expression for the projector H,
    "H_numerical": numerical values of the projector H,
    "rule": einsum rule to apply the projector H to get the natural tensor from the two input natural tensors
    }
}

- In "l1-l2-l3-normalization", l1 and l2 are the ranks of the two input natural tensors,
  l3 is the rank of the output natural tensor, and normalization can be `none` or
  `unity`, indicating whether the projector H is normalized or not.


# decomposition_and_reconstruction_projector.yaml

These projectors decompose a physical Cartesian tensor (e.g. polarizability, elasticity)
into its natural tensor components and reconstruct it back. Specifically:

- `H` extracts a natural tensor component from the physical tensor.
- `G` embeds a natural tensor component back into the physical tensor space.
- `S` is the normalization projector.

The file contains the symbolic and numerical values of `G`, `H`, and `S` for each
physical tensor and each natural tensor component (labeled by its weight).

The data is organized in the following way:

{physical_tensor_name:
    {"rank": rank of the physical tensor,
     "symmetry": symmetry of the physical tensor,
     "GHS":
        weight:
            {"G": [{"symbolic": symbolic expression for the projector G,
                   "numerical": numerical values of the projector G }],
             "H": [{"symbolic": symbolic expression for the projector H,
                   "numerical": numerical values of the projector H }],
             "S": [{"symbolic": symbolic expression for the projector S,]
                   "numerical": numerical values of the projector S }]
            }
    }
}

- "physical_tensor_name" is the name of the physical tensor, such as "polarizability" "
elasticity".
- "weight" is the weight of the natural tensor in the decomposition of the physical
  tensor, which is a positive integer.
