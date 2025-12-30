import os

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from lerobot_so101_teleop import assets
from lerobot_so101_teleop.mdp import (
    randomize_physics_material,
    reset_root_state_uniform,
    reset_rings_transform_state_uniform,
)


from ..base.base_env_cfg import BaseEnvCfg, LerobotSo101BaseSceneCfg, EventCfg

assets_path = os.path.dirname(os.path.abspath(assets.__file__))


@configclass
class LerobotSo101RockAStackSceneCfg(LerobotSo101BaseSceneCfg):

    rock_a_stack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RockAStack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/rock-a-stack-simple.usd",
        ),
        # to deconflict with the robot during initialization
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.2, 0, 0.2),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )

    base_plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RockAStack/Base",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0, 0.04),
            rot=(0.98098075, 0.0, 0.0, 0.19410504),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # rings
        for idx in range(1, 6):
            ring_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RockAStack/" + f"Ring{idx:02d}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.2, 0, 0.03),  # intial pose to not conflict with the robot
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            setattr(self, f"Ring{idx:02d}", ring_cfg)
            print(f"Ring{idx:02d}")


@configclass
class RockAStackEventCfg(EventCfg):
    """Configuration for events."""

    reset_physics_material_props = EventTerm(
        func=randomize_physics_material,
        mode="reset",
        params={
            "asset_names": [
                "Ring01",
                "Ring02",
                "Ring03",
                "Ring04",
                "Ring05",
                "base_plate",
            ]
        },
    )

    # Reset physics objects to their initial positions
    reset_base_plate = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.15 - 0.2, 0.3 - 0.2),
                # offset by -2 to not conflict with the robot
                "y": (0.0, -0.15),
                # only scatter on right side of the robot
                "z": (0.04, 0.04),
                # no need to scatter on z
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
                # randomize yaw
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("base_plate"),
        },
    )

    reset_ring_transform = EventTerm(
        func=reset_rings_transform_state_uniform,
        mode="reset",
        params={
            "asset_names": ["Ring01", "Ring02", "Ring03", "Ring04", "Ring05"],
            "x_y_pose_range": {
                "x": (-0.1, 0.1),
                # offset from the initial position to not conflict with the robot
                "y": (0.25, 0.1),
                # only scatter on left side of the robot
            },
        },
    )


@configclass
class RockAStackSimpleEnvCfg(BaseEnvCfg):
    """Configuration for the rock-a-stack environment."""

    scene: LerobotSo101RockAStackSceneCfg = LerobotSo101RockAStackSceneCfg()
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
