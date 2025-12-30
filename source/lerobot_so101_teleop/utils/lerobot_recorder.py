import os
import threading
import queue

import torch
import numpy as np
from tqdm import tqdm

import omni.kit.app
from carb.eventdispatcher import get_eventdispatcher, Event


from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotRecorder:

    STOP_RECORDING_EVENT: str = "lerobot_so101_teleop.stop_recording"
    CANCEL_RECORDING_EVENT: str = "lerobot_so101_teleop.cancel_recording"

    def __init__(
        self,
        task_name: str,
        repo_id: str,
        dataset_root: str,
        fps: int,
        device: str,
        cameras: dict,
    ):

        self.fps = fps
        self.dt = 1 / self.fps

        self.camera_features_template = {
            "dtype": "video",
            "shape": (0, 0, 3),
            "names": ["height", "width", "channels"],
        }

        self.FOLLOWER_OBS_FEATURES = {
            "observation.state": {
                "dtype": "float32",
                "fps": self.fps,
                "shape": (6,),
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            }
        }

        self.cameras = cameras

        for camera_name in self.cameras.keys():
            camera = self.cameras[camera_name]

            features = self.camera_features_template.copy()

            features["shape"] = (camera["height"], camera["width"], 3)

            self.FOLLOWER_OBS_FEATURES.update(
                {
                    f"observation.images.{camera_name}": features,
                }
            )

        self.LEADER_ACTION_FEATURES = {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "fps": self.fps,
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            }
        }

        self.SO101_ACTION_NAMES = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.dataset_features = {
            **self.FOLLOWER_OBS_FEATURES,
            **self.LEADER_ACTION_FEATURES,
        }
        self.num_recorded_episodes = 0

        self.device = device
        self.capcity = 3 * 60 * self.fps
        self.current_frame = 0

        self.action_buffers_tensor = None
        self.observation_buffer_tensor = None
        self.rgb_buffer_tensors = {}

        self.episode_queue = queue.Queue()
        self.episode_processor_stop_event = threading.Event()

        self.stop_recording_sub = get_eventdispatcher().observe_event(
            observer_name="stop_recording_observer",
            event_name=self.STOP_RECORDING_EVENT,
            on_event=self.save_episode,
        )

        self.cancel_recording_sub = get_eventdispatcher().observe_event(
            observer_name="cancel_recording_observer",
            event_name=self.CANCEL_RECORDING_EVENT,
            on_event=self.cancel_recording,
        )

        self.episode_processor_thread = threading.Thread(
            target=self.async_episode_processor,
            daemon=True,
        )
        self.episode_processor_thread.start()

    def check_dataset_exists(self):
        if os.path.exists(self.dataset_root):
            return True
        else:
            return False

    def _init_existing_dataset(self):
        self.dataset = LeRobotDataset(
            self.repo_id,
            root=self.dataset_root,
        )
        print(f"[INFO]: Existing dataset initialized - {self.dataset.root}")
        return

    def init_dataset(self):
        if self.check_dataset_exists():
            try:
                self._init_existing_dataset()
                return
            except:
                raise ValueError(
                    f"[ERROR]: Dataset folder exists but cannot be initialized at {self.dataset_root}"
                )

        self.dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self.dataset_features,
            root=self.dataset_root,
            robot_type="so101_follower",
        )

        print(f"[INFO]: New dataset initialized - {self.dataset.root}")

    def allocate_buffers(self):
        self.action_buffers_tensor = torch.zeros(
            (self.capcity, 6), dtype=torch.float32, device=self.device
        )
        self.observation_buffer_tensor = torch.zeros(
            (self.capcity, 6), dtype=torch.float32, device=self.device
        )

        for camera_name in self.cameras.keys():
            self.rgb_buffer_tensors[camera_name] = torch.zeros(
                (
                    self.capcity,
                    self.cameras[camera_name]["height"],
                    self.cameras[camera_name]["width"],
                    3,
                ),
                dtype=torch.uint8,
                device=self.device,
            )

    def push_frame_to_buffer(self, action, observation, visual_buffers):
        if self.current_frame >= self.capcity:
            # TODO: extand tensors to increase the buffer capacity if reached
            print(
                f"[INFO]: Reached the maximum capacity of the buffer. Skipping frame {self.current_frame}"
            )
            return

        if self.observation_buffer_tensor is None:
            self.allocate_buffers()

        self.action_buffers_tensor[self.current_frame] = action.clone()
        self.observation_buffer_tensor[self.current_frame] = observation.clone()

        for camera_name in self.cameras.keys():
            self.rgb_buffer_tensors[camera_name][self.current_frame] = visual_buffers[
                camera_name
            ].clone()

        self.current_frame += 1

    def add_dataset_frame(self, action, observation, rgb_buffers, frame_index):
        # action = np.array(
        #     [action[name] for name in self.SO101_ACTION_NAMES], dtype=np.float32
        # )
        # action
        # for name in self.SO101_ACTION_NAMES:
        frame = {
            "action": action,
            "observation.state": observation,
            "task": self.task_name,
            # "timestamp": frame_index * self.dt,
        }
        for camera_name in self.cameras.keys():
            frame[f"observation.images.{camera_name}"] = rgb_buffers[camera_name]

        self.dataset.add_frame(frame)

    def save_episode(self, event: Event):
        if event.event_name == self.STOP_RECORDING_EVENT:
            

            # batch copy to cpu
            print(f"[INFO]: Copy episode to CPU...")

            cpu_action_buffers_tensor = self.action_buffers_tensor.to("cpu").numpy()
            cpu_obs_buffer_tensor = self.observation_buffer_tensor.to("cpu").numpy()

            cpu_rgb_buffer_tensors = {}
            for camera_name in self.cameras.keys():
                cpu_rgb_buffer_tensors[camera_name] = (
                    self.rgb_buffer_tensors[camera_name].to("cpu").numpy()
                )

            self.episode_queue.put({
                "action_buffers": cpu_action_buffers_tensor.copy(),
                "observation_buffer_tensor": cpu_obs_buffer_tensor.copy(),
                "rgb_buffer_tensors": cpu_rgb_buffer_tensors.copy(),
                "total_frames": self.current_frame,
            })

            print(f"[INFO]: Episode added to queue.")

            self.action_buffers_tensor = None
            self.observation_buffer_tensor = None
            self.rgb_buffer_tensor = {}

            self.current_frame = 0
            print("[INFO]: Cleared buffers")


    def cancel_recording(self, event: Event):
        if event.event_name == self.CANCEL_RECORDING_EVENT:
            print(f"[INFO]: Cancelled recording.")
            self.action_buffers_tensor = None
            self.observation_buffer_tensor = None
            self.rgb_buffer_tensor = {}
            self.current_frame = 0
            print("[INFO]: Cleared buffers")
            print(f"[INFO]: Episode cancelled.")

    def async_episode_processor(self):
        while not self.episode_processor_stop_event.is_set():
            try:
                episode = self.episode_queue.get(timeout=1)
                print(f"[INFO]: [ASYNC] received episode from queue...")

                action_buffers = episode["action_buffers"]
                observation_buffer_tensor = episode["observation_buffer_tensor"]
                rgb_buffer_tensors = episode["rgb_buffer_tensors"]
                total_frames = episode["total_frames"]
                
                for frame_index in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
                    rgb_buffers = {}
                    for camera_name in self.cameras.keys():
                        rgb_buffers[camera_name] = rgb_buffer_tensors[camera_name][frame_index]

                    self.add_dataset_frame(
                        action_buffers[frame_index],
                        observation_buffer_tensor[frame_index],
                        rgb_buffers,
                        frame_index,
                    )

                self.dataset.save_episode()
                self.dataset.finalize()
                self._init_existing_dataset()
                self.episode_queue.task_done()

                self.num_recorded_episodes += 1
                print(f"[INFO]: Episode {self.num_recorded_episodes} saved.")

                if self.episode_queue.empty():
                    print(f"[INFO]: No more episodes in queue. Stopping processor thread...")
                else:
                    print(f"[INFO]: Additional {self.episode_queue.qsize()} episodes in queue.")
                  

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async processing: {e}")
                continue

    def _del__(self):
        # stop the episode processor thread
        self.episode_processor_stop_event.set()
        self.episode_processor_thread.join(timeout=0.1)
        self.episode_processor_thread = None
        print("processor thread joined")
