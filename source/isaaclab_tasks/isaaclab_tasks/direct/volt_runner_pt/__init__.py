import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-VoltRunner-Pt-Direct-v0",
    entry_point="isaaclab_tasks.direct.volt_runner_pt.volt_runner_pt_env:VoltRunnerPtEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.volt_runner_pt.volt_runner_pt_env:VoltRunnerPtEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VoltRunnerPtPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-VoltRunner-Pt-Align-Direct-v0",
    entry_point="isaaclab_tasks.direct.volt_runner_pt.volt_runner_pt_align_env:VoltRunnerPtAlignEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.volt_runner_pt.volt_runner_pt_align_env:VoltRunnerPtAlignEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VoltRunnerPtPPORunnerCfg",
    },
)