import torch
import torch.nn as nn
from typing import Dict
from lattice_utils import make_lattice_config


class LatticeBufferRegistry(nn.Module):
    """
    Manages and registers lattice geometry as unlearnable module buffers.
    
    Lattice tensors (spatial_diff, group_action_space, group_mult) are registered
    as buffers with persistent=False to avoid checkpointing overhead while ensuring
    they persist across forward passes without re-computation.
    """
    
    def __init__(self, lattice_type: str, **lattice_params):
        """
        Initialize the lattice buffer registry.
        
        Args:
            lattice_type: Type of lattice ("1d_dihedral", etc.)
            **lattice_params: Additional parameters for the lattice (e.g., n for 1D).
        """
        super().__init__()
        self.lattice_type = lattice_type
        self.lattice_params = lattice_params
        
        # Generate lattice configuration
        config = make_lattice_config(lattice_type, **lattice_params)
        
        # Store metadata
        self.num_sites = config["num_sites"]
        self.group_size = config["group_size"]
        
        # Register tensors as buffers (not trainable, no gradients required)
        self.register_buffer(
            "spatial_diff",
            config["spatial_diff"],
            persistent=False
        )
        self.register_buffer(
            "group_action_space",
            config["group_action_space"],
            persistent=False
        )
        self.register_buffer(
            "group_mult",
            config["group_mult"],
            persistent=False
        )
    
    def forward(self, x: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Return the registered lattice buffers.
        
        This is useful for introspection or when buffers need to be accessed
        without explicitly indexing the module attributes.
        
        Args:
            x: Dummy input (unused, for torch.jit compatibility).
        
        Returns:
            Dictionary with keys: "spatial_diff", "group_action_space", "group_mult".
        """
        return {
            "spatial_diff": self.spatial_diff,
            "group_action_space": self.group_action_space,
            "group_mult": self.group_mult,
        }
    
    def get_lattice_config(self) -> Dict:
        """
        Return full lattice configuration as a dictionary.
        
        Returns:
            Dictionary with lattice metadata and buffers.
        """
        return {
            "spatial_diff": self.spatial_diff,
            "group_action_space": self.group_action_space,
            "group_mult": self.group_mult,
            "num_sites": self.num_sites,
            "group_size": self.group_size,
            "lattice_type": self.lattice_type,
        }
    
    def to_device(self, device: torch.device) -> None:
        """
        Move all lattice buffers to a specified device.
        
        Args:
            device: Target device ("cpu", "cuda", etc.).
        """
        self.to(device)
    
    def verify_buffers(self) -> bool:
        """
        Verify that all lattice buffers are properly registered and have correct shapes.
        
        Returns:
            True if all buffers are valid, False otherwise.
        """
        # Check spatial_diff
        if self.spatial_diff.shape[0] != self.num_sites or \
           self.spatial_diff.shape[1] != self.num_sites:
            print(f"❌ spatial_diff shape mismatch: {self.spatial_diff.shape}")
            return False
        
        # Check group_action_space
        if self.group_action_space.shape[0] != self.group_size or \
           self.group_action_space.shape[1] != self.num_sites:
            print(f"❌ group_action_space shape mismatch: {self.group_action_space.shape}")
            return False
        
        # Check group_mult
        if self.group_mult.shape[0] != self.group_size or \
           self.group_mult.shape[1] != self.group_size:
            print(f"❌ group_mult shape mismatch: {self.group_mult.shape}")
            return False
        
        # Verify dtypes
        if self.spatial_diff.dtype != torch.long:
            print(f"❌ spatial_diff dtype mismatch: {self.spatial_diff.dtype}")
            return False
        if self.group_action_space.dtype != torch.long:
            print(f"❌ group_action_space dtype mismatch: {self.group_action_space.dtype}")
            return False
        if self.group_mult.dtype != torch.long:
            print(f"❌ group_mult dtype mismatch: {self.group_mult.dtype}")
            return False
        
        return True


if __name__ == "__main__":
    print("Testing LatticeBufferRegistry...")
    
    # Create registry for 1D Dihedral with n=16
    registry = LatticeBufferRegistry("1d_dihedral", n=16)
    print(f"✓ Registry created for 1D Dihedral with n=16")
    print(f"  - num_sites: {registry.num_sites}")
    print(f"  - group_size: {registry.group_size}")
    
    # Verify buffers
    if registry.verify_buffers():
        print(f"✓ All buffers verified successfully")
    else:
        print(f"❌ Buffer verification failed")
        exit(1)
    
    # Test forward pass
    buffers_dict = registry.forward()
    print(f"✓ Forward pass successful")
    print(f"  - spatial_diff: {buffers_dict['spatial_diff'].shape}")
    print(f"  - group_action_space: {buffers_dict['group_action_space'].shape}")
    print(f"  - group_mult: {buffers_dict['group_mult'].shape}")
    
    # Test config retrieval
    config = registry.get_lattice_config()
    print(f"✓ Config retrieval successful")
    print(f"  - lattice_type: {config['lattice_type']}")
    
    # Test device move
    registry.to_device(torch.device("cpu"))
    print(f"✓ Device move successful")
    
    print("\n✓ ALL LATTICE BUFFER REGISTRY TESTS PASSED")
