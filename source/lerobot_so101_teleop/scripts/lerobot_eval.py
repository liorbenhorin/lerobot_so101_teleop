# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Teleop agent.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

parser.add_argument(
    "--policy_path", type=str, default=None, help="Path to the policy checkpoint - Can be on HF Hub or local path"
)
parser.add_argument("--seed", type=int, default=101, help="Environment seed")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
# Add argument for renaming observation keys for compatibility between environment and policy
parser.add_argument(
    "--rename_map",
    type=str,
    required=False,
    default=None,
    help=(
        'JSON mapping for renaming camera keys to match policy/feature config: key is simulation feature name, value is policy feature name '
        'e.g. \'{"sim_name1": "policy_name1", "sim_name2": "policy_name2"}\'. '
    ),
)

# parser.add_argument(
#     "--repo_root", type=str, default=None, help="Repository root to store the dataset."
# )
# parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
args_cli.enable_cameras = True



# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import lerobot_so101_teleop.tasks  # noqa: F401
from lerobot_so101_teleop.utils.keyboard import KeyboardControl
from lerobot_so101_teleop.utils.lerobot_interface import LeRobotSO101Interface


def main():
    keyboard_control = KeyboardControl()

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")

    # cameras
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print(f"[Info]: No cameras found - videos will not be recorded")

    # lerobot interface
    rename_map = json.loads(args_cli.rename_map) if args_cli.rename_map else None
    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=None, # does not matter for evaluation
        id="leader_arm_1", # does not matter for evaluation
        cameras=cameras,
        fps=30,
        kind="follower",
        rename_map=rename_map, # only supports renaming cameras for now
    )
    robot_iface.init_device(visualize=True)
    robot_iface.make_policy(args_cli.policy_path)

    # reset environment
    obs, _ = env.reset()
    robot_iface.policy.reset()
    robot_iface.preprocessor.reset()
    robot_iface.postprocessor.reset()

    # simulate environment
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    initial_action = torch.tensor(
        [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465],
        device=env.unwrapped.device,
    )

    step = 0
    # total_reward = 0.0
    num_episodes = 0
    num_successes = 0
    success_rate = 0.0

    pbar = None

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            if step == 0:
                pbar = tqdm(
                    total=env.unwrapped.max_episode_length ,
                    desc=f"Rollout (ep {num_episodes + 1}, success: {success_rate:.1f}%)",
                    unit="step",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )

            if step < 10:  # warmup, not critical but helps
                actions[:] = initial_action # just a stand up pose for the arm

            else:
                observation_frame = robot_iface.sim_obs_to_policy_processor(
                    obs["policy"][0].clone(),
                    obs["visual"]
                )
                action_values = robot_iface.predict_action(observation_frame)
                actions[:] = robot_iface.prediction_to_sim_processor(
                    action_values, observation_frame, log=True
                )

            obs, rewards, terminated, truncated, info = env.step(actions)

            # reward_value = rewards.item() if rewards.numel() == 1 else rewards.mean().item()
            # total_reward += reward_value
            step += 1
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
            
            # Check for episode termination
            is_terminated = terminated.item() if terminated.numel() == 1 else terminated.any().item()
            is_truncated = truncated.item() if truncated.numel() == 1 else truncated.any().item()
            
            if is_terminated or is_truncated:
                if pbar is not None:
                    pbar.close()
                    pbar = None
                
                num_episodes += 1
                
                if is_terminated and not is_truncated:
                    num_successes += 1

                success_rate = (num_successes / num_episodes) * 100

                # Reset for next episode
                obs, _ = env.reset()
                robot_iface.policy.reset()
                robot_iface.preprocessor.reset()
                robot_iface.postprocessor.reset()
                step = 0

                continue

            # Manual reset with 'R' key
            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                if pbar is not None:
                    pbar.close()
                    pbar = None

                print(f"[MANUAL RESET] Episode interrupted at step {step}")
                obs, _ = env.reset()
                robot_iface.policy.reset()
                robot_iface.preprocessor.reset()
                robot_iface.postprocessor.reset()
                step = 0
                total_reward = 0.0
                continue

            if num_episodes >= args_cli.num_episodes:
                # Close progress bar if still open
                if pbar is not None:
                    pbar.close()
                    pbar = None
                print(f"[INFO]: Evaluated {args_cli.num_episodes} episodes")
                print(f"[INFO]: Success Rate: {num_successes}/{args_cli.num_episodes} ({success_rate:.1f}%)")
                env.close()
                simulation_app.close()

    env.close()


if __name__ == "__main__":

    main()

    while True:
        simulation_app.update()

    simulation_app.close()
