import torch
import torch.nn as nn
from typing import Dict
from lattice_utils import make_lattice_config


class LatticeBufferRegistry(nn.Module):
    """
    Manages and registers lattice geometry as unlearnable module buffers.
    """
    
    def __init__(self, lattice_type: str, **lattice_params):
        """
        Initialise the lattice buffer registry.
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
        """
        return {
            "spatial_diff": self.spatial_diff,
            "group_action_space": self.group_action_space,
            "group_mult": self.group_mult,
        }
    
    def get_lattice_config(self) -> Dict:
        """
        Return full lattice configuration as a dictionary.
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
        """
        self.to(device)
    
    def verify_buffers(self) -> bool:
        """
        Verify that all lattice buffers are properly registered and have correct shapes.
        """
        # Check spatial_diff
        if self.spatial_diff.shape[0] != self.num_sites or \
           self.spatial_diff.shape[1] != self.num_sites:
            return False
        
        # Check group_action_space
        if self.group_action_space.shape[0] != self.group_size or \
           self.group_action_space.shape[1] != self.num_sites:
            return False
        
        # Check group_mult
        if self.group_mult.shape[0] != self.group_size or \
           self.group_mult.shape[1] != self.group_size:
            return False
        
        # Verify dtypes
        if self.spatial_diff.dtype != torch.long:
            return False
        if self.group_action_space.dtype != torch.long:
            return False
        if self.group_mult.dtype != torch.long:
            return False
        
        return True