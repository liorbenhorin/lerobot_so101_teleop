import gymnasium as gym


##
# Register Gym environments.
##


gym.register(
    id="Lerobot-So101-Teleop-Rock-A-Stack-Simple",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rock_a_stack_env_cfg:RockAStackSimpleEnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-Teleop-Rock-A-Stack-Hard",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rock_a_stack_hard_env_cfg:RockAStackHardEnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-Teleop-Rock-A-Stack-Hard-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rock_a_stack_hard_env_cfg:RockAStackHardEnvEvalCfg",
    },
)