# from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import TerminationTermCfg



def ring_insertion_success(
    env: ManagerBasedRLEnv,
    xy_tolerance: float,
    z_threshold: float,
    ring_cfg: SceneEntityCfg = SceneEntityCfg("yellow_ring"),
    pole_cfg: SceneEntityCfg = SceneEntityCfg("rock_a_stack_base"),
) -> torch.Tensor:

    # Extract the ring and pole objects from the scene
    ring: RigidObject = env.scene[ring_cfg.name]
    pole: RigidObject = env.scene[pole_cfg.name]
    
    # Get world positions: (num_envs, 3) where columns are [x, y, z]
    ring_pos_w = ring.data.root_pos_w
    pole_pos_w = pole.data.root_pos_w
    
    # Calculate distance in x-y plane only
    xy_distance = torch.norm(ring_pos_w[:, :2] - pole_pos_w[:, :2], dim=1)
    
    # Check if ring is aligned in x-y
    is_aligned = xy_distance <= xy_tolerance
    
    # Check if ring is at appropriate height (inserted onto pole)
    is_at_height = ring_pos_w[:, 2] <= z_threshold
    
    # Return True when both conditions are met (successful insertion)
    return is_aligned & is_at_height


class DynamicTimeout(ManagerTermBase):
    """Dynamic timeout that extends when ring gets close to pole.
    
    Parameters:
        base_timeout: Initial timeout in steps (default: 250)
        extension: Additional steps granted when condition is met (default: 150)
        tolerance: Distance threshold to trigger extension (default: 0.02m)
        ring_cfg: Configuration for the ring object
        pole_cfg: Configuration for the pole object
    
    Note:
        The _has_extended flag ensures only ONE extension per episode.
        Episodes will NEVER exceed 400 steps.
    """
    
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # Get parameters
        self.base_timeout = cfg.params.get("base_timeout", 200)
        self.extension = cfg.params.get("extension", 100)
        self.tolerance = cfg.params.get("tolerance", 0.02)
        
        # Initialize state buffers for each environment
        self._has_extended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._current_max_steps = torch.full(
            (env.num_envs,), self.base_timeout, dtype=torch.int32, device=env.device
        )
        
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        base_timeout: int = 250,
        extension: int = 150,
        tolerance: float = 0.02,
        ring_cfg: SceneEntityCfg = SceneEntityCfg("yellow_ring"),
        pole_cfg: SceneEntityCfg = SceneEntityCfg("rock_a_stack_base"),
    ) -> torch.Tensor:
        """Check if episode should timeout with dynamic extension.
        
        Returns:
            Boolean tensor indicating which environments should timeout.
        """
        # Get ring and pole positions
        ring: RigidObject = env.scene[ring_cfg.name]
        pole: RigidObject = env.scene[pole_cfg.name]
        
        ring_pos_w = ring.data.root_pos_w
        pole_pos_w = pole.data.root_pos_w
        
        # Calculate distance in x-y plane
        xy_distance = torch.norm(ring_pos_w[:, :2] - pole_pos_w[:, :2], dim=1)
        
        # Check if ring is within tolerance and we haven't extended yet
        within_tolerance = xy_distance <= tolerance
        should_extend = within_tolerance & (~self._has_extended)
        
        # Extend timeout for environments that meet the condition (only once per episode)
        if should_extend.any():
            # Mark as extended (prevents any future extensions in this episode)
            self._has_extended[should_extend] = True
            # Extend the max steps (capped at 300 to ensure max length)
            new_max_steps = self.base_timeout + extension
            self._current_max_steps[should_extend] = min(new_max_steps, 400)
            
            # Print notification (only for first environment for simplicity)
            # if should_extend[0]:
                # print(f"[TIME EXTENSION] Ring is close. Extending episode by {extension} steps (now {self._current_max_steps[0]} total)")
        
        # Check if we've exceeded the current max steps (hard cap at 300)
        return env.episode_length_buf >= torch.minimum(self._current_max_steps, torch.tensor(400, device=env.device))
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the state for specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        
        # Reset extension flag and max steps for these environments
        self._has_extended[env_ids] = False
        self._current_max_steps[env_ids] = self.base_timeout

