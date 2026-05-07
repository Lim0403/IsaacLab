# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate PPO ALIGN-only policy.

This script evaluates only the PPO alignment policy.
No raster search is used.

Evaluation target:
- task: Isaac-VoltRunner-Pt-Align-Direct-v0
- checkpoint: model_1999.pt

Outputs:
- step_log.csv
- episode_metrics.csv
- summary_metrics.csv
- summary_metrics.json
- plots/trajectory_ep_xxxx.png
- plots/pt_over_time_ep_xxxx.png
- plots/action_over_time_ep_xxxx.png
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import importlib.metadata as metadata
from pathlib import Path

from packaging import version

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RSL_RL_SCRIPT_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "..", "scripts", "reinforcement_learning", "rsl_rl")
)
if _RSL_RL_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPT_DIR)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


CHECKPOINT_PATH = "/home/lim/IsaacLab/logs/rsl_rl/volt_runner_pt/2026-04-29_12-44-07/model_1999.pt"
DEFAULT_TASK = "Isaac-VoltRunner-Pt-Align-Direct-v0"


parser = argparse.ArgumentParser(description="Evaluate PPO align-only policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--num_episodes", type=int, default=100, help="Number of evaluation episodes.")
parser.add_argument("--plot_episodes", type=int, default=5, help="Number of episodes to save plots for.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for evaluation results.")
parser.add_argument(
    "--max_eval_steps",
    type=int,
    default=200,
    help="Maximum number of evaluation steps per episode.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Force ALIGN-only env + trained checkpoint.
args_cli.task = DEFAULT_TASK
args_cli.checkpoint = CHECKPOINT_PATH
args_cli.num_envs = 1

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

installed_version = metadata.version("rsl-rl-lib")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        print(f"[WARN] No rows to save: {path}")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved: {path}")


def reset_env_and_get_obs(env):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def get_local_robot_xy(env_unwrapped):
    root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
    env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
    local_x = float((root_pos[0] - env_origin[0]).item())
    local_y = float((root_pos[1] - env_origin[1]).item())
    return local_x, local_y


def read_env0_state_before_step(
    env_unwrapped,
    obs,
    action,
    episode_id: int,
    step: int,
    time_s: float,
):
    """Read env_0 state before env.step() to avoid reset-contaminated final states."""
    policy_obs = obs["policy"]
    pt = float(policy_obs[0, 0].detach().cpu().item())
    delta_pt = float(policy_obs[0, 1].detach().cpu().item())

    robot_x, robot_y = get_local_robot_xy(env_unwrapped)

    receiver_x = float(env_unwrapped.receiver_x[0].detach().cpu().item())
    receiver_y = float(env_unwrapped.receiver_y[0].detach().cpu().item())

    dx = receiver_x - robot_x
    dy = receiver_y - robot_y
    distance_error = math.sqrt(dx * dx + dy * dy)

    action_x = float(action[0, 0].detach().cpu().item())
    action_y = float(action[0, 1].detach().cpu().item())
    action_wz = float(action[0, 2].detach().cpu().item())

    vx_cmd = action_x * float(env_unwrapped.cfg.action_scale_vx)
    vy_cmd = action_y * float(env_unwrapped.cfg.action_scale_vy)
    wz_cmd = action_wz * float(env_unwrapped.cfg.action_scale_wz)

    curr_vx = float(env_unwrapped.curr_vx[0].detach().cpu().item())
    curr_vy = float(env_unwrapped.curr_vy[0].detach().cpu().item())
    curr_wz = float(env_unwrapped.curr_wz[0].detach().cpu().item())
    speed_norm = math.sqrt(curr_vx * curr_vx + curr_vy * curr_vy + curr_wz * curr_wz)

    success_now = (
        pt >= float(env_unwrapped.cfg.success_pt_threshold)
        and speed_norm < float(env_unwrapped.cfg.success_speed_epsilon)
    )

    out_of_bounds = (
        abs(robot_x) > float(env_unwrapped.cfg.workspace_size_x) / 2.0
        or abs(robot_y) > float(env_unwrapped.cfg.workspace_size_y) / 2.0
    )

    return {
        "episode": episode_id,
        "step": step,
        "time": time_s,
        "robot_x": robot_x,
        "robot_y": robot_y,
        "receiver_x": receiver_x,
        "receiver_y": receiver_y,
        "distance_error": distance_error,
        "pt": pt,
        "delta_pt": delta_pt,
        "action_x": action_x,
        "action_y": action_y,
        "action_wz": action_wz,
        "vx_cmd": vx_cmd,
        "vy_cmd": vy_cmd,
        "wz_cmd": wz_cmd,
        "curr_vx": curr_vx,
        "curr_vy": curr_vy,
        "curr_wz": curr_wz,
        "speed_norm": speed_norm,
        "reward": 0.0,
        "success_now": bool(success_now),
        "eval_success": False,
        "eval_timeout": False,
        "eval_out_of_bounds": bool(out_of_bounds),
        "eval_done": False,
        "env_done": False,
    }


def compute_episode_metrics(episode_id: int, rows: list[dict]) -> dict:
    if not rows:
        return {
            "episode": episode_id,
            "success": False,
            "final_pt": 0.0,
            "peak_pt": 0.0,
            "time_to_success": math.nan,
            "final_alignment_error": math.nan,
            "episode_length": 0,
            "cumulative_reward": 0.0,
            "timeout": False,
            "out_of_bounds": False,
            "path_length": 0.0,
            "mean_control_effort": math.nan,
        }

    final_row = rows[-1]

    success = any(r.get("eval_success", False) for r in rows)
    timeout = any(r.get("eval_timeout", False) for r in rows)
    out_of_bounds = any(r.get("eval_out_of_bounds", False) for r in rows)

    time_to_success = math.nan
    for r in rows:
        if r.get("eval_success", False):
            time_to_success = r["time"]
            break

    path_length = 0.0
    for i in range(1, len(rows)):
        dx = rows[i]["robot_x"] - rows[i - 1]["robot_x"]
        dy = rows[i]["robot_y"] - rows[i - 1]["robot_y"]
        path_length += math.sqrt(dx * dx + dy * dy)

    control_efforts = [
        r["action_x"] ** 2 + r["action_y"] ** 2 + r["action_wz"] ** 2
        for r in rows
    ]

    return {
        "episode": episode_id,
        "success": bool(success),
        "final_pt": final_row["pt"],
        "peak_pt": max(r["pt"] for r in rows),
        "time_to_success": time_to_success,
        "final_alignment_error": final_row["distance_error"],
        "episode_length": len(rows),
        "cumulative_reward": sum(r["reward"] for r in rows),
        "timeout": bool(timeout and not success),
        "out_of_bounds": bool(out_of_bounds and not success),
        "path_length": path_length,
        "mean_control_effort": float(np.mean(control_efforts)) if control_efforts else math.nan,
    }


def compute_summary_metrics(episode_metrics: list[dict]) -> dict:
    n = len(episode_metrics)
    if n == 0:
        return {}

    success_rows = [m for m in episode_metrics if m["success"]]
    success_times = [m["time_to_success"] for m in success_rows if not math.isnan(m["time_to_success"])]

    def mean(values):
        values = list(values)
        if len(values) == 0:
            return math.nan
        return float(np.mean(values))

    return {
        "num_episodes": n,
        "success_rate": sum(bool(m["success"]) for m in episode_metrics) / n,
        "mean_final_pt": mean(m["final_pt"] for m in episode_metrics),
        "mean_peak_pt": mean(m["peak_pt"] for m in episode_metrics),
        "mean_time_to_success_success_only": mean(success_times),
        "mean_episode_reward": mean(m["cumulative_reward"] for m in episode_metrics),
        "mean_final_alignment_error": mean(m["final_alignment_error"] for m in episode_metrics),
        "mean_episode_length": mean(m["episode_length"] for m in episode_metrics),
        "timeout_rate": sum(bool(m["timeout"]) for m in episode_metrics) / n,
        "out_of_bounds_rate": sum(bool(m["out_of_bounds"]) for m in episode_metrics) / n,
        "mean_path_length": mean(m["path_length"] for m in episode_metrics),
        "mean_control_effort": mean(m["mean_control_effort"] for m in episode_metrics),
    }


def plot_trajectory(rows: list[dict], out_path: Path, workspace_x: float, workspace_y: float):
    if not rows:
        return

    xs = [r["robot_x"] for r in rows]
    ys = [r["robot_y"] for r in rows]
    receiver_x = rows[0]["receiver_x"]
    receiver_y = rows[0]["receiver_y"]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, linewidth=2, label="Robot trajectory")
    plt.scatter(xs[0], ys[0], marker="o", s=70, label="Start")
    plt.scatter(xs[-1], ys[-1], marker="x", s=90, label="Final")
    plt.scatter(receiver_x, receiver_y, marker="*", s=140, label="Receiver")

    half_x = workspace_x / 2.0
    half_y = workspace_y / 2.0

    plt.axvline(-half_x, linestyle="--", linewidth=1)
    plt.axvline(half_x, linestyle="--", linewidth=1)
    plt.axhline(-half_y, linestyle="--", linewidth=1)
    plt.axhline(half_y, linestyle="--", linewidth=1)

    plt.xlim(-half_x - 0.1, half_x + 0.1)
    plt.ylim(-half_y - 0.1, half_y + 0.1)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("PPO align-only trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pt_over_time(rows: list[dict], out_path: Path, success_threshold: float):
    if not rows:
        return

    times = [r["time"] for r in rows]
    pts = [r["pt"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(times, pts, linewidth=2, label="Pt")
    plt.axhline(success_threshold, linestyle="--", linewidth=1.5, label=f"Threshold={success_threshold:.2f}")

    success_times = [r["time"] for r in rows if r.get("eval_success", False)]
    if success_times:
        plt.axvline(success_times[0], linestyle="-.", linewidth=1.5, label="Success")

    plt.xlabel("Time [s]")
    plt.ylabel("Pt")
    plt.title("Pt over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_action_over_time(rows: list[dict], out_path: Path):
    if not rows:
        return

    times = [r["time"] for r in rows]
    ax = [r["action_x"] for r in rows]
    ay = [r["action_y"] for r in rows]
    aw = [r["action_wz"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(times, ax, linewidth=1.5, label="action_x")
    plt.plot(times, ay, linewidth=1.5, label="action_y")
    plt.plot(times, aw, linewidth=1.5, label="action_wz")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized action")
    plt.title("Action over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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

    output_dir = (
        Path(args_cli.output_dir).expanduser().resolve()
        if args_cli.output_dir
        else Path(log_dir) / "align_only_evaluation_model_1999"
    )
    output_dir = ensure_dir(output_dir)
    plots_dir = ensure_dir(output_dir / "plots")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval_align_only"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
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

    all_step_rows = []
    episode_metrics = []

    episode_id = 0
    episode_step = 0
    episode_rows = []

    eval_success_counter = 0
    global_timestep = 0

    print("=" * 80)
    print("[INFO] PPO ALIGN-only evaluation started")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Target episodes: {args_cli.num_episodes}")
    print(f"[INFO] Max eval steps per episode: {args_cli.max_eval_steps}")
    print("=" * 80)

    while simulation_app.is_running() and episode_id < args_cli.num_episodes:
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)

            row = read_env0_state_before_step(
                env_unwrapped=env_unwrapped,
                obs=obs,
                action=actions,
                episode_id=episode_id,
                step=episode_step,
                time_s=episode_step * dt,
            )

            if row["success_now"]:
                eval_success_counter += 1
            else:
                eval_success_counter = 0

            eval_success = eval_success_counter >= int(env_unwrapped.cfg.success_hold_steps)
            eval_timeout = episode_step >= int(args_cli.max_eval_steps) - 1
            eval_out_of_bounds = bool(row["eval_out_of_bounds"])

            eval_done = eval_success or eval_timeout or eval_out_of_bounds

            row["eval_success"] = bool(eval_success)
            row["eval_timeout"] = bool(eval_timeout and not eval_success)
            row["eval_out_of_bounds"] = bool(eval_out_of_bounds and not eval_success)
            row["eval_done"] = bool(eval_done)

            if not eval_done:
                obs, rewards, dones, _ = env.step(actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

                reward_value = float(rewards[0].detach().cpu().item())
                env_done_value = bool(dones[0].detach().cpu().item())

                row["reward"] = reward_value
                row["env_done"] = env_done_value

                # Critical fix:
                # If the underlying environment ends before max_episode_length
                # and it is not out-of-bounds, interpret it as env-side success_done.
                if env_done_value:
                    row["eval_done"] = True

                    env_max_step = int(env_unwrapped.max_episode_length) - 1
                    is_env_timeout = episode_step >= env_max_step
                    is_env_oob = bool(row["eval_out_of_bounds"])

                    if (not is_env_timeout) and (not is_env_oob):
                        row["eval_success"] = True
                        row["eval_timeout"] = False
                        row["eval_out_of_bounds"] = False
                    elif is_env_oob:
                        row["eval_success"] = False
                        row["eval_timeout"] = False
                        row["eval_out_of_bounds"] = True
                    else:
                        row["eval_success"] = False
                        row["eval_timeout"] = True
                        row["eval_out_of_bounds"] = False
            else:
                row["reward"] = 0.0
                row["env_done"] = False

            episode_rows.append(row)
            all_step_rows.append(row)

            if row["eval_done"]:
                metrics = compute_episode_metrics(episode_id, episode_rows)
                episode_metrics.append(metrics)

                print(
                    f"[EVAL] ep={episode_id:04d} "
                    f"success={metrics['success']} "
                    f"final_pt={metrics['final_pt']:.4f} "
                    f"peak_pt={metrics['peak_pt']:.4f} "
                    f"tts={metrics['time_to_success']} "
                    f"err={metrics['final_alignment_error']:.4f} "
                    f"len={metrics['episode_length']} "
                    f"reward={metrics['cumulative_reward']:.2f} "
                    f"timeout={metrics['timeout']} "
                    f"oob={metrics['out_of_bounds']}"
                )

                if episode_id < args_cli.plot_episodes:
                    plot_trajectory(
                        episode_rows,
                        plots_dir / f"trajectory_ep_{episode_id:04d}.png",
                        workspace_x=float(env_unwrapped.cfg.workspace_size_x),
                        workspace_y=float(env_unwrapped.cfg.workspace_size_y),
                    )
                    plot_pt_over_time(
                        episode_rows,
                        plots_dir / f"pt_over_time_ep_{episode_id:04d}.png",
                        success_threshold=float(env_unwrapped.cfg.success_pt_threshold),
                    )
                    plot_action_over_time(
                        episode_rows,
                        plots_dir / f"action_over_time_ep_{episode_id:04d}.png",
                    )

                obs = reset_env_and_get_obs(env)

                eval_success_counter = 0
                episode_id += 1
                episode_step = 0
                episode_rows = []
            else:
                episode_step += 1

        global_timestep += 1

        if global_timestep % 20 == 0:
            print(
                f"global_step={global_timestep}, "
                f"episode={episode_id}, "
                f"episode_step={episode_step}, "
                f"pt={row['pt']:.4f}, "
                f"err={row['distance_error']:.4f}, "
                f"speed={row['speed_norm']:.4f}, "
                f"success_counter={eval_success_counter}"
            )

        if args_cli.video and global_timestep == args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    save_csv(output_dir / "step_log.csv", all_step_rows)
    save_csv(output_dir / "episode_metrics.csv", episode_metrics)

    summary = compute_summary_metrics(episode_metrics)
    summary_rows = [{"metric": k, "value": v} for k, v in summary.items()]
    save_csv(output_dir / "summary_metrics.csv", summary_rows)

    with open(output_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("=" * 80)
    print("[INFO] Summary metrics")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
