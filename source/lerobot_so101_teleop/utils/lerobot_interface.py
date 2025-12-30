import torch
import uuid


from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots import make_robot_from_config
from lerobot.processor import make_default_processors
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.policies.utils import make_robot_action
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# Ideally, we should make a base class for all robot interfaces,
# and inherit from it for each robot type.
class LeRobotSO101Interface:

    # USD Robot has joint ranges that are not ranging from -100 to 100
    SO101_USD_MAPPING = {
        "shoulder_pan": {"joint_min": -110, "joint_max": 110},
        "shoulder_lift": {"joint_min": -100, "joint_max": 100},
        "elbow_flex": {"joint_min": -100, "joint_max": 90},
        "wrist_flex": {"joint_min": -95, "joint_max": 95},
        "wrist_roll": {"joint_min": -160, "joint_max": 160},
        "gripper": {"joint_min": -10, "joint_max": 100},
    }

    # Joint order is the order of the joints in the USD articulation
    SO101_JOINT_ORDER = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    def __init__(
        self,
        device: str,
        port: str,
        id: str,
        cameras: dict,
        fps: int,
        kind: str = "leader",
        rename_map: dict = None,
    ):

        self.port = port
        self.id = id
        self.cameras = cameras
        self.device = device
        self.fps = fps
        self.kind = kind
        self.rename_map = rename_map

        self.joint_names = [joint.split(".")[0] for joint in self.SO101_JOINT_ORDER]
        self.joint_mins = torch.tensor(
            [self.SO101_USD_MAPPING[name]["joint_min"] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.joint_maxs = torch.tensor(
            [self.SO101_USD_MAPPING[name]["joint_max"] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )

    def make_cameras_cfg(self):
        cameras = {}
        # need to rename cameras to match policy/feature config
        for camera in self.cameras.keys():
            camera_name = camera
            if self.rename_map:
                camera_name = self.rename_map[camera]
            cameras[camera_name] = OpenCVCameraConfig(
                index_or_path="null",  # we are in simulation :)
                fps=self.fps,
                width=self.cameras[camera]["width"],
                height=self.cameras[camera]["height"],
            )

        return cameras

    def make_cfg(self):
        if self.kind == "leader":
            return SO101LeaderConfig(port=self.port, id=self.id)
        elif self.kind == "follower":
            cameras = self.make_cameras_cfg()
            return SO101FollowerConfig(port=self.port, id=self.id, cameras=cameras)

    def init_device(self, visualize: bool = False):
        self.cfg = self.make_cfg()
        self.robot = make_robot_from_config(self.cfg)

        random_session_name = f"eval_{uuid.uuid4().hex[:8]}"
        if visualize:
            init_rerun(session_name=random_session_name)

        print(f"[INFO]: Connected to the Arm at {self.port} with id {self.id}")

    def connect(self):
        self.robot.connect()

    def get_raw_actions_tensor(self, real_action):
        return torch.tensor(
            [real_action[joint] for joint in self.SO101_JOINT_ORDER],
            dtype=torch.float32,
            device=self.device,
        )

    def get_mapped_actions_vectorized(self, raw_values):
        normalized = torch.zeros_like(raw_values)
        normalized[:-1] = (
            raw_values[:-1] + 100
        ) / 200.0  # first 5 joints: -100-100 -> 0-1
        normalized[-1] = raw_values[-1] / 100.0  # gripper: 0-100 -> 0-1

        # Map to joint ranges (degrees)
        mapped_deg = self.joint_mins + normalized * (self.joint_maxs - self.joint_mins)

        # Convert to radians
        return mapped_deg * torch.pi / 180

    def get_raw_actions_from_radians(self, raw_values):
        # Convert from radians to degrees
        mapped_deg = raw_values * 180 / torch.pi

        # Reverse the joint range mapping
        normalized = (mapped_deg - self.joint_mins) / (
            self.joint_maxs - self.joint_mins
        )

        # Reverse the normalization
        raw_degrees = torch.zeros_like(normalized)
        raw_degrees[:-1] = (
            normalized[:-1] * 200.0 - 100
        )  # first 5 joints: 0-1 -> -100-100
        raw_degrees[-1] = normalized[-1] * 100.0  # gripper: 0-1 -> 0-100

        return raw_degrees

    def make_policy(
        self,
        name_or_path: str,
    ):

        _, self.robot_action_processor, self.robot_observation_processor = (
            make_default_processors()
        )

        self.dataset_features = combine_feature_dicts(
            # Observation features (joint positions + camera images)
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(
                    observation=self.robot.observation_features
                ),
                use_videos=True,  # MUST be True to include cameras!
            ),
            # Action features (target joint positions)
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_action_processor,
                initial_features=create_initial_features(
                    action=self.robot.action_features
                ),
                use_videos=True,
            ),
        )

        policy_config = PreTrainedConfig.from_pretrained(name_or_path)
        policy_config.pretrained_path = name_or_path
        policy_config.device = self.device

        self.dataset_meta = DummyDatasetMeta(self.dataset_features, self.robot.name)

        self.policy = make_policy(policy_config, ds_meta=self.dataset_meta)

        print(f"[INFO]: Policy loaded")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=name_or_path,
            dataset_stats={},  # No normalization stats (policy handles it)
            preprocessor_overrides={
                "device_processor": {"device": policy_config.device},
            },
        )

        print(f"[INFO]: Preprocessor and postprocessor loaded")

    def sim_obs_to_policy_processor(self, sim_observation: torch.Tensor, visual_obs: dict) -> dict:
        # TODO: makes no sense to copy to host here, but this is whay predict_action expects

        state: torch.Tensor = self.get_raw_actions_from_radians(sim_observation)

        state_np = state.cpu().numpy()

        sim_observation = {}
        sim_observation["shoulder_pan.pos"] = state_np[0]
        sim_observation["shoulder_lift.pos"] = state_np[1]
        sim_observation["elbow_flex.pos"] = state_np[2]
        sim_observation["wrist_flex.pos"] = state_np[3]
        sim_observation["wrist_roll.pos"] = state_np[4]
        sim_observation["gripper.pos"] = state_np[5]

        for camera in self.cameras.keys():
            img: torch.Tensor = (
                visual_obs[f"camera_{camera}"][0].clone()
            )
            # need to rename camera to match policy/feature config
            camera_name = camera
            if self.rename_map:
                camera_name = self.rename_map[camera]
            sim_observation[camera_name] = img.cpu().detach().numpy()

        obs_processed = self.robot_observation_processor(sim_observation)
        observation_frame = build_dataset_frame(
            self.dataset_features,  # Defines structure of observations/actions
            obs_processed,  # Raw observation data
            prefix=OBS_STR,  # Adds "observation." prefix to keys
        )

        return observation_frame

    def predict_action(self, observation_frame: dict) -> dict:
        action_values = predict_action(
            observation=observation_frame,
            policy=self.policy,
            device=get_safe_torch_device(self.policy.config.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.policy.config.use_amp,
            task=None,
            robot_type=self.robot.robot_type,
        )
        return action_values

    def prediction_to_sim_processor(
        self, action_values: dict, observation_frame: dict, log: bool = False
    ) -> dict:
        robot_action = make_robot_action(action_values, self.dataset_features)
        robot_action_to_send = self.robot_action_processor((robot_action, None))

        motor_actions = {
            k: v for k, v in robot_action_to_send.items() if k.endswith(".pos")
        }
        sim_motor_actions: torch.Tensor = self.get_raw_actions_tensor(motor_actions)

        if log:
            log_rerun_data(observation=observation_frame, action=robot_action)

        mapped_sim_motor_actions: torch.Tensor = self.get_mapped_actions_vectorized(
            sim_motor_actions
        )
        return mapped_sim_motor_actions

    # TODO: we should return a single object instead of a tuple
    def real_to_sim_obs_processor(
        self, sim_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_action: torch.Tensor = self.get_raw_actions_tensor(sim_obs)
        mapped_action: torch.Tensor = self.get_mapped_actions_vectorized(real_action)
        return real_action, mapped_action

    # TODO: we should return a single object instead of a tuple
    def sim_to_real_dataset_processor(
        self, policy_obs: torch.Tensor, visual_obs: dict
    ) -> tuple[torch.Tensor, dict]:
        real_obs = self.get_raw_actions_from_radians(policy_obs)
        visual_buffers = {}
        for camera in self.cameras.keys():
            visual_buffers[camera] = visual_obs[f"camera_{camera}"][0]

        return real_obs, visual_buffers


class DummyDatasetMeta:
    def __init__(self, features, robot_type):
        self.features = features
        self.stats = {}
        self.robot_type = robot_type


if __name__ == "__main__":
    lerobot_cfg = {"port": "/dev/ttyACM0", "id": "leader_arm_1"}
    lerobot_interface = LeRobotSO101Interface(cfg=lerobot_cfg)
    while True:
        real_action = lerobot_interface.teleop_dev.get_action()
        print(type(real_action))
        print(real_action)
        print(list(real_action.keys()))
