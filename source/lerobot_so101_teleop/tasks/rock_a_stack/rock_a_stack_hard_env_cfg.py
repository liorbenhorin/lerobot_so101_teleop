import os
import math
import numpy as np

import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

import isaaclab.envs.mdp as mdp_lib

from . import mdp


# import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from lerobot_so101_teleop import assets
from lerobot_so101_teleop.mdp import (
    randomize_physics_material,
    reset_root_state_uniform
)

from ..base.base_env_cfg import BaseEnvCfg, LerobotSo101BaseSceneCfg, EventCfg

assets_path = os.path.dirname(os.path.abspath(assets.__file__))


@configclass
class LerobotSo101RockAStackHardSceneCfg(LerobotSo101BaseSceneCfg):

    camera_external = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            fisheye_nominal_height=480,
            fisheye_nominal_width=640,
            fisheye_optical_centre_x=320,
            fisheye_optical_centre_y=120,  # hmmm
            fisheye_max_fov=170,
            fisheye_polynomial_a=0,
            fisheye_polynomial_b=0.0015,
            fisheye_polynomial_c=0,
            fisheye_polynomial_d=0,
            fisheye_polynomial_e=0,
            fisheye_polynomial_f=0,
        ),
        # place the camera at the mount point
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.57, 0.0, 0.055),
            rot=euler_angles_to_quat(np.array([115, 0, 90]), degrees=True),
            convention="opengl",
        ),
    )

    rock_a_stack_base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rock_a_stack_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/rock-a-stack/base.usd",
        ),
        # to deconflict with the robot during initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.18, 0, 0.1),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )

    yellow_ring = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/yellow_ring",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/rock-a-stack/yellow_ring.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0.0, 0.05),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )


@configclass
class RockAStackEventCfg(EventCfg):
    """Configuration for events."""

    pass

    reset_physics_material_props = EventTerm(
        func=randomize_physics_material,
        mode="reset",
        params={"asset_names": ["yellow_ring", "rock_a_stack_base"]},
    )

    # Reset physics objects to their initial positions
    reset_base_plate = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.01, 0.03),
                "y": (-0.03, -0.05),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("rock_a_stack_base"),
        },
    )

    reset_yellow_ring = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (0.05, 0.08),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("yellow_ring"),
        },
    )


@configclass
class RockAStackHardEnvCfg(BaseEnvCfg):
    """Configuration for the rock-a-stack environment."""

    scene: LerobotSo101RockAStackHardSceneCfg = LerobotSo101RockAStackHardSceneCfg()
    events: RockAStackEventCfg = RockAStackEventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # create physics material
        physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        )
        self.physics_material_path = "/Looks/physicsMaterial"
        physics_material_cfg.func(self.physics_material_path, physics_material_cfg)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Dynamic timeout: starts at 200 steps, extends by 100 when ring gets close to pole
    time_out = DoneTerm(
        func=mdp.DynamicTimeout,
        time_out=True,
        params={
            "base_timeout": 250,  # Initial timeout in steps
            "extension": 150,  # Additional steps when condition is met
            "tolerance": 0.02,  # Distance threshold (2cm) to trigger extension
            "ring_cfg": SceneEntityCfg("yellow_ring"),
            "pole_cfg": SceneEntityCfg("rock_a_stack_base"),
        },
    )

    # Terminate on successful insertion (task success)
    success = DoneTerm(
        func=mdp.ring_insertion_success,
        time_out=False,  # This is a termination (task completion), not a timeout
        params={
            "xy_tolerance": 0.005,  # 5mm tolerance in x-y plane
            "z_threshold": 0.06,  # Ring must be at ~6cm height to be considered inserted
            "ring_cfg": SceneEntityCfg("yellow_ring"),
            "pole_cfg": SceneEntityCfg("rock_a_stack_base"),
        },
    )


@configclass
class RockAStackHardEnvEvalCfg(RockAStackHardEnvCfg):

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 400 / 60.0
