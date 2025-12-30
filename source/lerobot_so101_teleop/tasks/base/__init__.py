import gymnasium as gym


##
# Register Gym environments.
##


gym.register(
    id="Lerobot-So101-Teleop-Base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:BaseEnvCfg",
    },
)
