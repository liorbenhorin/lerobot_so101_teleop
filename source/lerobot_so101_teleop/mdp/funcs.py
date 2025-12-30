import torch

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim import get_current_stage
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.prims import XFormPrim


def reset_rings_transform_state_uniform(
    env,
    env_ids: torch.Tensor,
    x_y_pose_range: dict,
    asset_names: list[str],
):
    assets: list[RigidObject | Articulation] = [
        env.scene[asset_name] for asset_name in asset_names
    ]

    ring_height = 0.03
    z_pos = 0
    shuffled_rings = torch.randperm(len(assets))
    for i in range(len(assets)):
        asset = assets[shuffled_rings[i]]
        root_states = asset.data.default_root_state[env_ids].clone()

        range_x = x_y_pose_range.get("x", (0.0, 0.0))
        range_y = x_y_pose_range.get("y", (0.0, 0.0))
        range_z = (z_pos, z_pos)

        range_list = [range_x, range_y, range_z]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device
        )

        positions = (
            root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        )
        orientations = root_states[:, 3:7]

        # set into the physics simulation
        asset.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        z_pos += ring_height


def randomize_physics_material(
    env, env_ids: torch.Tensor | None, asset_names: list[str] = None
):

    stage = get_current_stage()
    physics_material_prim = stage.GetPrimAtPath(env.cfg.physics_material_path)

    static_friction = math_utils.sample_uniform(0.0, 0.5, (1,), device="cpu").item()
    dynamic_friction = math_utils.sample_uniform(0.0, 0.5, (1,), device="cpu").item()
    restitution = math_utils.sample_uniform(0.0, 0.3, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        # apply physics material to the assets
        for asset_name in asset_names:
            asset = env.scene[asset_name]
            asset_prims = sim_utils.find_matching_prims(asset.cfg.prim_path)

            for asset_prim in asset_prims:
                sim_utils.bind_physics_material(
                    asset_prim.GetPath(), env.cfg.physics_material_path
                )
        # set physics material properties
        physics_material_prim.GetAttribute("physics:staticFriction").Set(
            static_friction
        )
        physics_material_prim.GetAttribute("physics:dynamicFriction").Set(
            dynamic_friction
        )
        physics_material_prim.GetAttribute("physics:restitution").Set(restitution)


def set_robot_visual_material_props(
    env, env_ids: torch.Tensor | None, diffuse_color: tuple[float, float, float]
):
    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
        material_prim = sim_utils.find_matching_prims(material_prim_path)[0]
        material_prim.GetAttribute("inputs:diffuse_color_constant").Set(diffuse_color)


def randomize_static_asset_orientation(
    env,
    env_ids: torch.Tensor | None,
    pose_range: dict | None,
    asset_cfg: SceneEntityCfg = None,
):

    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.unwrapped.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.unwrapped.device
    )
    orientations = math_utils.quat_from_euler_xyz(
        rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2]
    )

    asset_xform = XFormPrim(prim_paths_expr=asset_prim_path)

    with Sdf.ChangeBlock():
        asset_xform.set_local_poses(orientations=orientations)


def randomize_light_exposure(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)
