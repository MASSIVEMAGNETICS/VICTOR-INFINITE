# omegallama/registry.py

class OmegaBlockRegistry:
    """
    Global registry for all OmegaLlamaLayers block types.
    Enables plug-and-play extensibility for Attention, FFN, Adapter, MoE, etc.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register a block class under a given name.
        Usage:
            @OmegaBlockRegistry.register('flash_attn')
            class FlashAttention(...): ...
        """
        def wrapper(block_cls):
            cls._registry[name] = block_cls
            return block_cls
        return wrapper

    @classmethod
    def create(cls, name, **kwargs):
        """
        Instantiate a registered block by name.
        Args:
            name: The string key for the block type.
            kwargs: Arguments to pass to the block's constructor.
        Returns:
            An instance of the requested block.
        """
        if name not in cls._registry:
            raise ValueError(f"Block type '{name}' not registered in OmegaBlockRegistry.")
        return cls._registry[name](**kwargs)

    @classmethod
    def available_blocks(cls):
        """
        Returns a list of all registered block names.
        """
        return list(cls._registry.keys())