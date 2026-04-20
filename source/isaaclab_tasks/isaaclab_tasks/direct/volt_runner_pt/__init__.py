# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Volt Runner Pt direct RL task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-VoltRunner-Pt-Direct-v0",
    entry_point=f"{__name__}.volt_runner_pt_env:VoltRunnerPtEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.volt_runner_pt_env:VoltRunnerPtEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:VoltRunnerPtPPORunnerCfg",
    },
)