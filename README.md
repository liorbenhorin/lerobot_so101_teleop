# Lerobot SO-101 Teleop Isaac Lab environment


[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)


![](./banner.jpg)


[](https://github.com/user-attachments/assets/39d67d5d-b2f9-4184-9aa8-4c874476afd0)


[example.webm](https://github.com/user-attachments/assets/f66d9502-2cda-4ac9-b0ff-f750d6b19e2d)

Sample Environment for the LeRobot SO-101 Robot in Isaac Lab to collect demonstrations in a simulation. This can be utilized to later procedurally scale up datasets using various methods of domain randomization and style transfer techniques ✨ 🤖

Episoded are recorded directly to Lerobot Dataset, and can quickly be uploaded to HuggingFace Hub 🚀



## Installation

- Create a Conda environment
    ```bash
    conda create -y -n lerobot-isaac-lab python=3.11
    ```

- In your the new environment, Install the following:

    
    ```bash
    conda install ffmpeg=7.1.1 -c conda-forge

    pip install uv

    # install Isaac Sim
    uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    # run isaac sim and accept the EULA
    isaacsim
  
    # Install Isaac Lab
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    git checkout v2.3.0
    ./isaaclab.sh -i
    cd ..

    # Install this repo!
    git clone https://github.com/liorbenhorin/lerobot_so101_teleop.git
    cd lerobot_so101_teleop
    git lfs fetch # assets
    uv pip install -e source/lerobot_so101_teleop

    # Fix numpy dependency
    uv pip install numpy==1.26.0 lxml==4.9.4 packaging==23.2
    ```

- Make sure your Lerobot _Leader_ arm has been [calibrated](https://huggingface.co/docs/lerobot/en/so101#calibrate).

- Edit `source/lerobot_so101_teleop/scripts/lerobot_agent.py` by changing `lerobot_cfg` to match your port and id of your setup. (You have this information from the calibration step)

- Verify that the extension is correctly installed by listing the available tasks:

    ```bash
    list_envs
    ```
    You should see a several environments named like `Lerobot-So101-Teleop-*`


- Run the environment

    ```bash
    lerobot_agent --task Lerobot-So101-Teleop-Rock-A-Stack-Simple
    ```
- Get familiar with the teleop feeling

## Record a dataset

- Run the environment and include dataset repo-id and repo-root (They will get created if not already exists)

    ```bash
    lerobot_agent --task Lerobot-So101-Teleop-Rock-A-Stack-Hard \
    --repo_id ${HF_USER}/so101_teleop \
    --repo_root $(pwd)/datasets/so101_teleop \
    --task_name "Pick up the yellow ring and put it on the pole"
    ```

    - Click `S` to start/stop the recording. 
    - Click `C` to cancel current recording (Very useful!).
    - Reset the environment `R` will also stop the recording.
    - Episodes are queued for processing, while you work.
    - When you are done recodring, look for this massage in the console: `[INFO]: No more episodes in queue. Stopping processor thread...`
    - Exit the simulation with `Ctrl+C`.

## Playback & Upload

- To playback dataset episodes, use lerobot rerun visualizer
    ```bash
    lerobot-dataset-viz \
    --repo-id ${HF_USER}/so101_teleop \
    --root $(pwd)/datasets/so101_teleop \
    --episode-index 0 # or other episode
    ```

- To push your dataset to HuggingFace Hub (Optional)
    ```bash
    lerobot_push_dataset \
    --repo-id hf-repo-id/so101_teleop \
    --root $(pwd)/datasets/so101_teleop \
    --tags robotics teleop rock-a-stack \ # separate by spaces
    ```

## Evaluate policy

We assume you already trained a model based on the data collected (either in simulation or in real) and have a model ready to use

```bash
    lerobot_eval \
    --task Lerobot-So101-Teleop-Rock-A-Stack-Hard-Eval \
    --policy_path ${HF_USER}/your_policy
```

If your policy has differnt camera names, you can also provide a mapping.
The key is the simulated camera name and the value is the policy camera name.

```bash
    --rename_map '{"ego":"front", "external":"top"}'
```

## Creating new Tasks

To learn how to add new tasks, [follow this guide](source/lerobot_so101_teleop/docs/tasks.md)

## Contributors

- Thank you [LycheeAI](https://lycheeai-hub.com/) for making the SO101 arm available in USD format  💚 [https://github.com/MuammerBay/so-arm101-ros2-bridge/tree/main/IsaacSim_USD](https://github.com/MuammerBay/so-arm101-ros2-bridge/tree/main/IsaacSim_USD)

- Contributions are welcome via pull requests

## Limitations

- Simulated teleoperation can be hard - but training makes perfect 🏅


