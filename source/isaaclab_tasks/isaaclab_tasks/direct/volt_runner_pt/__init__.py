import gymnasium as gym

from . import agents
from .volt_runner_pt_slow_env import VoltRunnerPtSlowEnvCfg
from .volt_runner_pt_align_slow_env import VoltRunnerPtAlignSlowEnvCfg
from .volt_runner_pt_raster_transfer_slow_env import (
    VoltRunnerPtRasterTransferSlowEnvCfg,
)


# ============================================================
# Original VoltRunner Pt task
# ============================================================

gym.register(
    id="Isaac-VoltRunner-Pt-Direct-v0",
    entry_point=(
        "isaaclab_tasks.direct.volt_runner_pt."
        "volt_runner_pt_env:VoltRunnerPtEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.direct.volt_runner_pt."
            "volt_runner_pt_env:VoltRunnerPtEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "VoltRunnerPtPPORunnerCfg"
        ),
    },
)


# ============================================================
# Original VoltRunner Pt Align task
# ============================================================

gym.register(
    id="Isaac-VoltRunner-Pt-Align-Direct-v0",
    entry_point=(
        "isaaclab_tasks.direct.volt_runner_pt."
        "volt_runner_pt_align_env:VoltRunnerPtAlignEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.direct.volt_runner_pt."
            "volt_runner_pt_align_env:VoltRunnerPtAlignEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "VoltRunnerPtPPORunnerCfg"
        ),
    },
)


# ============================================================
# Slow velocity VoltRunner Pt task
# ============================================================

gym.register(
    id="Isaac-VoltRunner-Pt-Slow-Direct-v0",
    entry_point=(
        "isaaclab_tasks.direct.volt_runner_pt."
        "volt_runner_pt_slow_env:VoltRunnerPtSlowEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VoltRunnerPtSlowEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "VoltRunnerPtPPORunnerCfg"
        ),
    },
)


# ============================================================
# Slow velocity VoltRunner Pt Align task
# ============================================================

gym.register(
    id="Isaac-VoltRunner-Pt-Align-Slow-Direct-v0",
    entry_point=(
        "isaaclab_tasks.direct.volt_runner_pt."
        "volt_runner_pt_align_slow_env:"
        "VoltRunnerPtAlignSlowEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VoltRunnerPtAlignSlowEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "VoltRunnerPtPPORunnerCfg"
        ),
    },
)


# ============================================================
# Raster-transfer slow alignment task
#
# Raster Search itself remains rule-based.
# PPO begins from several initial Pt ranges representing
# the state after Raster Search detects the receiver.
# ============================================================

gym.register(
    id="Isaac-VoltRunner-Pt-Raster-Transfer-Slow-Direct-v0",
    entry_point=(
        "isaaclab_tasks.direct.volt_runner_pt."
        "volt_runner_pt_raster_transfer_slow_env:"
        "VoltRunnerPtRasterTransferSlowEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VoltRunnerPtRasterTransferSlowEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "VoltRunnerPtRasterTransferSlowPPORunnerCfg"
        ),
    },
)
