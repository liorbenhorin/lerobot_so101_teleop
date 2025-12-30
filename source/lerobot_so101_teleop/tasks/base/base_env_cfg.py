import os
import math
import numpy as np

from pxr import Gf, Sdf

from isaacsim.core.utils.rotations import euler_angles_to_quat


import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


# import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg

from lerobot_so101_teleop.assets.so101 import SO101_CFG

from lerobot_so101_teleop import assets
from lerobot_so101_teleop.mdp import (
    randomize_light_exposure,
    randomize_static_asset_orientation,
    JointPositionActionCfg,
    joint_pos,
    reset_joints_by_offset,
    set_robot_visual_material_props,
    image,
)

assets_path = os.path.dirname(os.path.abspath(assets.__file__))


@configclass
class LerobotSo101BaseSceneCfg(InteractiveSceneCfg):

    env_spacing = 4.0
    num_envs = 1

    # room
    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/room.usda",
        ),
    )
    # lights
    room_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Room/lights/DiskLight")
    env_light = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Room/lights/DomeLight")

    # action pad
    action_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ActionPad",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/action-pad.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.23535, 0, 0.0257),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Camera
    camera_ego = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyeKannalaBrandtK3",
            fisheye_nominal_height=480,
            fisheye_nominal_width=640,
            fisheye_optical_centre_x=320,
            fisheye_optical_centre_y=240,
            fisheye_max_fov=170,
            fisheye_polynomial_a=0,
            fisheye_polynomial_b=0.0028,
            fisheye_polynomial_c=0,
            fisheye_polynomial_d=0,
            fisheye_polynomial_e=0,
            fisheye_polynomial_f=0,
            f_stop=180,  # x10 of real
            focal_length=2.4,  # 10th of real
            focus_distance=0.05,  # 5cm in front of the camera
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.005, 0.06, -0.062),
            rot=euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True),
            convention="opengl",
        ),
    )

    camera_external = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyeKannalaBrandtK3",
            fisheye_nominal_height=480,
            fisheye_nominal_width=640,
            fisheye_optical_centre_x=320,
            fisheye_optical_centre_y=240,
            fisheye_max_fov=170,
            fisheye_polynomial_a=0,
            fisheye_polynomial_b=0.0028,
            fisheye_polynomial_c=0,
            fisheye_polynomial_d=0,
            fisheye_polynomial_e=0,
            fisheye_polynomial_f=0,
        ),
        # place the camera at the mount point
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.34412, 0.41122, 0.4191),
            rot=euler_angles_to_quat(np.array([45, 0, 180]), degrees=True),
            convention="opengl",
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_obs = ObsTerm(func=joint_pos)

        def __post_init__(self) -> None:
            self.enable_corruption = False

    @configclass
    class VisualCfg(ObsGroup):
        """Observations for policy group."""

        camera_ego = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_ego"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        camera_external = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("camera_external"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot position
    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Rotation",
                    "Pitch",
                    "Elbow",
                    "Wrist_Pitch",
                    "Wrist_Roll",
                    "Jaw",
                ],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    reset_room_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("room_light"),
        },
    )

    reset_env_light_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-2.0, 4.0),
            "asset_cfg": SceneEntityCfg("env_light"),
        },
    )

    reset_action_pad_orientation = EventTerm(
        func=randomize_static_asset_orientation,
        mode="reset",
        params={
            "pose_range": {"yaw": (math.pi - 0.2, math.pi + 0.2)},
            "asset_cfg": SceneEntityCfg("action_pad"),
        },
    )

    reset_sky_dome_orientation = EventTerm(
        func=randomize_static_asset_orientation,
        mode="reset",
        params={
            "pose_range": {"yaw": (math.pi / 2, -math.pi / 2)},
            "asset_cfg": SceneEntityCfg("env_light"),
        },
    )

    # define robot main color
    reset_set_robot_visual_material = EventTerm(
        func=set_robot_visual_material_props,
        mode="reset",
        params={"diffuse_color": (0.12, 0.52, 0.47)},  # RGB - blueish color
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    pass


##
# Environment configuration
##
@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: LerobotSo101BaseSceneCfg = LerobotSo101BaseSceneCfg()

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP settings
    rewards = None  # No rewards for teleoperation
    terminations = None  # No terminations for teleoperation

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5

        self.scene.num_envs = 1  # Always 1 env for teleoperation
        # viewer settings
        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = (0.15, 0.0, 0.12)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = (
            self.decimation
        )  # render every 2 frames to the viewport

        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = False
