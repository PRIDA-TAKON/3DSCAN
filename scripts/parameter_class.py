
import torch
import torch.optim as optim
from typing import Dict, Any, Callable, Union

class ParameterClass:
    """
    Manages parameters and their optimizer. 
    Replacement for missing taichi_splatting.misc.parameter_class.
    """
    def __init__(self, tensors: Dict[str, torch.Tensor], learning_rates: Dict[str, float], optimizer_cls: Callable, base_lr: float = 1.0):
        self.tensors = tensors
        self.learning_rates = learning_rates
        self.optimizer_cls = optimizer_cls
        self.base_lr = base_lr
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        param_groups = []
        for name, tensor in self.tensors.items():
            lr = self.learning_rates.get(name, self.base_lr)
            param_groups.append({'params': [tensor], 'lr': lr, 'name': name})
        return self.optimizer_cls(param_groups)

    @classmethod
    def create(cls, tensors, learning_rates, base_lr, optimizer):
        # Tensors might be a tensordict or dict
        # Ensure they are leaf tensors with requires_grad=True
        processed_tensors = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().clone().requires_grad_(True)
            processed_tensors[k] = v
        
        return cls(processed_tensors, learning_rates, optimizer, base_lr)

    def __getattr__(self, name):
        if name in self.tensors:
            return self.tensors[name]
        # Allow access to optimizer attributes if needed or fail
        raise AttributeError(f"'ParameterClass' object has no attribute '{name}'")
    
    def __getitem__(self, key):
         # If key is a string, return tensor
         if isinstance(key, str):
             return self.tensors[key]
             
         # Slicing/Indexing returns a subset wrapper
         new_tensors = {k: v[key] for k, v in self.tensors.items()}
         return ParameterClassSubset(new_tensors)

    @property
    def batch_size(self):
        # Assume all tensors have same first dim
        return [next(iter(self.tensors.values())).shape[0]]

    def append_tensors(self, new_tensors_dict):
         """
         Appends new tensors to the existing parameters and re-initializes the optimizer.
         Use efficient state merging if possible, but strict concatenation is safer for simple impl.
         """
         with torch.no_grad():
             for k, v in new_tensors_dict.items():
                 if k in self.tensors:
                    old_tensor = self.tensors[k]
                    # Concatenate along dim 0
                    new_tensor = torch.cat([old_tensor, v], dim=0)
                    new_tensor.requires_grad_(True)
                    self.tensors[k] = new_tensor
                 
         # Re-create optimizer with new tensors
         # Note: This resets momentum usually. 
         # In full Gaussian Splatting, we want to preserve momentum for existing points?
         # For this MVP fix, resetting is acceptable as densification happens periodically.
         self.optimizer = self._create_optimizer()
         return self

class ParameterClassSubset:
    """Helper for sliced parameters (detached from optimizer)"""
    def __init__(self, tensors):
        self.tensors = tensors
    
    def clone(self):
         # Returns same type
         return ParameterClassSubset({k: v.clone() for k, v in self.tensors.items()})

    def to_tensordict(self):
        return self.tensors
    
    def __getattr__(self, name):
         if name in self.tensors:
            return self.tensors[name]
         raise AttributeError(f"'ParameterClassSubset' has no attribute {name}")
