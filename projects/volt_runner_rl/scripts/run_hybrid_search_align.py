# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid play script: raster search first, then PPO alignment."""

import argparse
import sys
import os
import time
import importlib.metadata as metadata

from packaging import version

# 원본 play.py 폴더를 sys.path에 추가해서 cli_args import가 되게 만든다.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RSL_RL_SCRIPT_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl")
)
if _RSL_RL_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPT_DIR)

from isaaclab.app import AppLauncher

# local imports from IsaacLab original rsl_rl script folder
import cli_args  # isort: skip


#CHECKPOINT_PATH = "/home/lim/IsaacLab/logs/rsl_rl/volt_runner_pt/2026-04-22_15-00-44/model_1999.pt"
#CHECKPOINT_PATH = "/home/lim/IsaacLab/logs/rsl_rl/volt_runner_pt/2026-04-28_18-42-49/model_1999.pt"
#CHECKPOINT_PATH = "/home/lim/IsaacLab/logs/rsl_rl/volt_runner_pt/2026-04-29_11-48-26/model_1999.pt"
CHECKPOINT_PATH = "/home/lim/IsaacLab/logs/rsl_rl/volt_runner_pt/2026-04-29_12-44-07/model_1999.pt"
DEFAULT_TASK = "Isaac-VoltRunner-Pt-Direct-v0"

P_ALIGN = 0.20
ALIGN_CONFIRM_STEPS = 3


class RasterSearchController:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        lane_spacing_x: float,
        search_vx: float,
        search_vy: float,
        pos_tol: float,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.lane_spacing_x = lane_spacing_x
        self.search_vx = search_vx
        self.search_vy = search_vy
        self.pos_tol = pos_tol

        self.mode = "SWEEP_Y"
        self.dir_y = +1
        self.target_y = self.y_max
        self.target_x = self.x_min

    def reset(self, current_x: float):
        self.mode = "SWEEP_Y"
        self.dir_y = +1
        self.target_y = self.y_max
        self.target_x = max(self.x_min, min(current_x, self.x_max))

    def update(self, x: float, y: float):
        if self.mode == "SWEEP_Y":
            if abs(y - self.target_y) <= self.pos_tol:
                self.target_x = min(self.target_x + self.lane_spacing_x, self.x_max)
                self.mode = "STEP_X"
                return 0.0, 0.0, 0.0
            return 0.0, self.search_vy * self.dir_y, 0.0

        if self.mode == "STEP_X":
            if abs(x - self.target_x) <= self.pos_tol:
                self.dir_y *= -1
                self.target_y = self.y_max if self.dir_y > 0 else self.y_min
                self.mode = "SWEEP_Y"
                return 0.0, 0.0, 0.0
            return self.search_vx, 0.0, 0.0

        return 0.0, 0.0, 0.0


def clamp_normalized_action(vx_cmd: float, vy_cmd: float, wz_cmd: float, env_cfg):
    ax = max(-1.0, min(1.0, vx_cmd / env_cfg.action_scale_vx))
    ay = max(-1.0, min(1.0, vy_cmd / env_cfg.action_scale_vy))
    aw = max(-1.0, min(1.0, wz_cmd / env_cfg.action_scale_wz))
    return ax, ay, aw


parser = argparse.ArgumentParser(description="Play hybrid raster+PPO policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# 강제로 runtime env + 새 PPO checkpoint 고정
args_cli.task = DEFAULT_TASK
args_cli.checkpoint = CHECKPOINT_PATH
args_cli.num_envs = 1

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

installed_version = metadata.version("rsl-rl-lib")

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    env_unwrapped = env.unwrapped

    raster = RasterSearchController(
        x_min=-0.65,
        x_max=0.65,
        y_min=-0.35,
        y_max=0.35,
        lane_spacing_x=0.08,
        search_vx=0.12,
        search_vy=0.20,
        pos_tol=0.02,
    )

    root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
    env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
    local_x = (root_pos[0] - env_origin[0]).item()

    raster.reset(local_x)
    hybrid_mode = "SEARCH"
    align_counter = 0
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            policy_obs = obs["policy"]
            pt = policy_obs[0, 0].item()

            root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
            env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
            local_x = (root_pos[0] - env_origin[0]).item()
            local_y = (root_pos[1] - env_origin[1]).item()

            if hybrid_mode == "SEARCH":
                vx_cmd, vy_cmd, wz_cmd = raster.update(local_x, local_y)

                if pt >= P_ALIGN:
                    align_counter += 1
                else:
                    align_counter = 0

                if align_counter >= ALIGN_CONFIRM_STEPS:
                    hybrid_mode = "ALIGN"
                    print("=" * 80)
                    print(f"SWITCH TO ALIGN at step {timestep}")
                    print(f"robot=({local_x:.4f}, {local_y:.4f}), pt={pt:.4f}")
                    print("=" * 80)
                    actions = policy(obs)
                else:
                    ax, ay, aw = clamp_normalized_action(vx_cmd, vy_cmd, wz_cmd, env_unwrapped.cfg)
                    actions = torch.tensor([[ax, ay, aw]], dtype=torch.float32, device=env_unwrapped.device)
            else:
                actions = policy(obs)

            obs, _, dones, _ = env.step(actions)

            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

            if torch.any(dones):
                root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
                env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
                local_x = (root_pos[0] - env_origin[0]).item()

                raster.reset(local_x)
                hybrid_mode = "SEARCH"
                align_counter = 0

        timestep += 1

        if timestep % 10 == 0:
            receiver_x = env_unwrapped.receiver_x[0].item()
            receiver_y = env_unwrapped.receiver_y[0].item()
            print("-" * 80)
            print(f"step={timestep}, mode={hybrid_mode}")
            print(f"robot=({local_x:.4f}, {local_y:.4f})")
            print(f"receiver=({receiver_x:.4f}, {receiver_y:.4f})")
            print(f"pt={pt:.4f}, align_counter={align_counter}")

        if args_cli.video and timestep == args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()