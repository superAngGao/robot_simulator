import numpy as np
import torch
import warp as wp

checks = {
    "numpy": ["empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "full", "array", "asarray", "copy"],
    "torch": ["empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "arange", "linspace", "logspace", "eye", "rand", "randn", "randint", "rand_like", "randn_like", "tensor", "as_tensor", "asarray", "from_numpy", "from_dlpack"],
    "warp": ["empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "array", "from_numpy", "to_torch", "from_torch"],
}
for name, attrs in checks.items():
    obj = {"numpy": np, "torch": torch, "warp": wp}[name]
    print(name, [(attr, hasattr(obj, attr)) for attr in attrs])
print("Tensor", [(attr, hasattr(torch.Tensor, attr)) for attr in ["cpu", "cuda", "to", "numpy", "item", "tolist", "contiguous", "new_empty", "new_zeros", "new_ones", "new_full", "resize_"]])
