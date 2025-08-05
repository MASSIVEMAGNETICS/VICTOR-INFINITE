# =================================================================================================
# FILE: nn-module-v3.0.0-Archon.py
# VERSION: v3.0.0-Archon
# NAME: OmegaGodCore NN Module
# AUTHOR: Brandon "iambandobandz" Emery & Victor, Upgraded by First Born AGI
# PURPOSE: Defines the abstract base class for all neural network layers (OmegaLayer).
#          Provides parameter registration, state management, and device placement.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# =================================================================================================

from typing import List, Dict, OrderedDict, Set
import logging

# Assuming tensor and ops are in files accessible in the same path
# or a path configured in the environment.
from tensor-v3.0.0-Archon import OmegaTensor

logger = logging.getLogger("OmegaGodCore")

class OmegaLayer:
    """
    The master blueprint for all neural network layers in OmegaGodCore.
    
    This class provides the fundamental functionality required for any layer:
    - Parameter tracking: Automatically discovers all learnable OmegaTensor parameters.
    - Sub-layer registration: Can contain other OmegaLayer instances, forming a tree structure.
    - State management: Can save and load its state (all parameters) via a state_dict.
    - Device placement: Can move all its parameters to a specified device.
    """
    def __init__(self):
        # Using OrderedDicts to maintain insertion order, which is crucial for reproducibility.
        self._parameters: Dict[str, OmegaTensor] = OrderedDict()
        self._sub_layers: Dict[str, 'OmegaLayer'] = OrderedDict()
        self.training = True # Layer is in training mode by default

    def __call__(self, *args, **kwargs):
        """
        The forward pass logic must be implemented by all subclasses.
        """
        raise NotImplementedError("Every OmegaLayer must implement its own __call__ (forward pass) method.")

    def _register_parameter(self, name: str, tensor: OmegaTensor):
        """
        Registers an OmegaTensor as a learnable parameter of this layer.
        The parameter becomes an attribute of the layer.
        
        Args:
            name (str): The name of the parameter.
            tensor (OmegaTensor): The parameter tensor itself.
        """
        if not isinstance(tensor, OmegaTensor):
            raise TypeError(f"Cannot register '{name}' as a parameter. It must be an OmegaTensor, but got {type(tensor)}.")
        if '.' in name:
            raise KeyError("Parameter name cannot contain '.'")
            
        setattr(self, name, tensor)
        self._parameters[name] = tensor

    def _register_layer(self, name: str, layer: 'OmegaLayer'):
        """
        Registers a sub-layer within this layer. This allows for creating complex,
        nested models. The sub-layer's parameters will be tracked.

        Args:
            name (str): The name of the sub-layer.
            layer (OmegaLayer): The sub-layer instance.
        """
        if not isinstance(layer, OmegaLayer):
            raise TypeError(f"Cannot register '{name}' as a sub-layer. It must be an OmegaLayer, but got {type(layer)}.")
        if '.' in name:
            raise KeyError("Sub-layer name cannot contain '.'")

        setattr(self, name, layer)
        self._sub_layers[name] = layer

    def parameters(self, recurse: bool = True) -> List[OmegaTensor]:
        """
        Returns a list of all learnable parameters in this layer and, optionally,
        all its registered sub-layers.

        Args:
            recurse (bool): If True, recursively includes parameters from sub-layers.

        Returns:
            A list of unique OmegaTensor parameters.
        """
        param_list: List[OmegaTensor] = list(self._parameters.values())
        
        if recurse:
            for name, layer in self._sub_layers.items():
                param_list.extend(layer.parameters(recurse=True))
        
        # Using dict.fromkeys to ensure uniqueness while preserving order
        return list(dict.fromkeys(param_list))

    def zero_grad(self, recurse: bool = True):
        """
        Resets the gradients of all learnable parameters to None.
        This is a critical step before performing a new backward pass.

        Args:
            recurse (bool): If True, recursively zeros gradients in sub-layers.
        """
        for p in self.parameters(recurse=False): # Get only direct parameters
            p.zero_grad()
        
        if recurse:
            for name, layer in self._sub_layers.items():
                layer.zero_grad(recurse=True)

    def state_dict(self) -> Dict[str, OmegaTensor]:
        """
        Returns a dictionary containing the entire state of the layer, including
        all parameters from this layer and its sub-layers. Keys are prefixed
        with layer names.

        Returns:
            A dictionary mapping from parameter name to OmegaTensor.
        """
        state = OrderedDict()
        for name, param in self._parameters.items():
            state[name] = param
        
        for layer_name, layer in self._sub_layers.items():
            sub_state = layer.state_dict()
            for sub_name, sub_param in sub_state.items():
                state[f"{layer_name}.{sub_name}"] = sub_param
        
        return state

    def load_state_dict(self, state_dict: Dict[str, OmegaTensor]):
        """
        Loads the layer's state from a state_dict.

        Args:
            state_dict (Dict[str, OmegaTensor]): A dictionary of parameters.
        """
        # This is a strict load. It expects all keys to match.
        current_state_dict = self.state_dict()
        if state_dict.keys() != current_state_dict.keys():
            missing = state_dict.keys() - current_state_dict.keys()
            unexpected = current_state_dict.keys() - state_dict.keys()
            err_msg = "Error(s) in loading state_dict:\n"
            if missing: err_msg += f"\tMissing keys: {', '.join(missing)}\n"
            if unexpected: err_msg += f"\tUnexpected keys: {', '.join(unexpected)}\n"
            raise KeyError(err_msg)

        for name, param in state_dict.items():
            # Find the parameter in the current model and update its data
            # This is a bit complex due to nested structure. A simpler way for now:
            # We assume the user provides tensors with data to be loaded.
            # A more robust implementation would navigate the module tree.
            parts = name.split('.')
            module = self
            for part in parts[:-1]:
                module = getattr(module, part)
            
            current_param = getattr(module, parts[-1])
            if current_param.shape != param.shape:
                raise ValueError(f"Shape mismatch for '{name}': saved tensor has shape {param.shape}, layer parameter has shape {current_param.shape}")
            
            # Update the data in-place
            current_param.data = param.data
        logger.info(f"Successfully loaded state_dict for layer {self.__class__.__name__}.")

    def train(self):
        """Sets the layer and all its sub-layers to training mode."""
        self.training = True
        for layer in self._sub_layers.values():
            layer.train()

    def eval(self):
        """Sets the layer and all its sub-layers to evaluation mode."""
        self.training = False
        for layer in self._sub_layers.values():
            layer.eval()

# Example of how you would use this base class (for illustration)
if __name__ == '__main__':
    class SimpleLinear(OmegaLayer):
        def __init__(self, in_features, out_features):
            super().__init__()
            # Use the registration methods
            self._register_parameter("weight", OmegaTensor(np.random.randn(in_features, out_features)))
            self._register_parameter("bias", OmegaTensor(np.zeros(out_features)))

        def __call__(self, x):
            # The actual matmul op would be patched onto OmegaTensor by ops.py
            return x.matmul(self.weight) + self.bias

    class SimpleModel(OmegaLayer):
        def __init__(self):
            super().__init__()
            self._register_layer("fc1", SimpleLinear(10, 5))
            self._register_layer("fc2", SimpleLinear(5, 1))

        def __call__(self, x):
            x = self.fc1(x)
            # A relu op would be used here
            return self.fc2(x)

    logger.setLevel(logging.DEBUG)
    model = SimpleModel()
    
    print("--- Model Parameters ---")
    params = model.parameters()
    for p in params:
        print(f"Name: {p.name}, Shape: {p.shape}, Requires Grad: {p.requires_grad}")
    assert len(params) == 4

    print("\n--- Model State Dict ---")
    state = model.state_dict()
    for name, p in state.items():
        print(f"Key: '{name}', Tensor Shape: {p.shape}")
    assert 'fc1.weight' in state
    assert 'fc2.bias' in state

    print("\n--- Zeroing Gradients ---")
    # Simulate a backward pass that populates grads
    for p in model.parameters():
        p.requires_grad = True
        p.grad = OmegaTensor(np.ones_like(p.data))
        print(f"Grad present for '{p.name}': {p.grad is not None}")
    
    model.zero_grad()
    print("Gradients after zero_grad():")
    for p in model.parameters():
        print(f"Grad present for '{p.name}': {p.grad is not None}")
        assert p.grad is None

    print("\n--- nn-module-v3.0.0-Archon.py: All checks passed. System nominal. ---")