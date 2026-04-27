# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from pxr import Gf, Sdf, UsdGeom


@configclass
class VoltRunnerPtAlignEnvCfg(DirectRLEnvCfg):
    """Configuration for the Volt Runner Pt ALIGN environment."""

    # env
    decimation = 12
    episode_length_s = 20.0
    action_space = 3
    observation_space = 13
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene 
    #down is visual best,,, 
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(
    # num_envs=64, env_spacing=4.0, replicate_physics=False, clone_in_fabric=False
    # )

    #real train use it
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # full workspace: 2.0 m x 1.2 m
    workspace_size_x = 2.0
    workspace_size_y = 1.2

    # central raster/search region: 1.30 m x 0.70 m
    search_area_size_x = 1.30
    search_area_size_y = 0.70

    # robot footprint approximation
    robot_size_x = 0.24
    robot_size_y = 0.20

    # receiver spawn region = search region inset by robot size
    receiver_x_min = -0.53
    receiver_x_max = 0.53
    receiver_y_min = -0.25
    receiver_y_max = 0.25
    receiver_z = 0.203

    # ALIGN training spawn:
    # robot starts near receiver, not at the front search line
    align_spawn_offset_x_min = -0.08
    align_spawn_offset_x_max = 0.08
    align_spawn_offset_y_min = -0.08
    align_spawn_offset_y_max = 0.08

    # receiver marker size
    receiver_radius = 0.06
    receiver_thickness = 0.002

    # action scaling
    action_scale_vx = 0.30
    action_scale_vy = 0.25
    action_scale_wz = 0.45

    # thresholds
    detect_pt_threshold = 0.10
    success_pt_threshold = 0.90
    success_hold_steps = 10
    success_speed_epsilon = 0.02

    # reward weights
    reward_alpha_dpt = 2.0
    reward_beta_pt = 0.3
    reward_gamma_action = 0.03
    reward_delta_time = 0.01
    reward_success_bonus = 5.0
    reward_fail_penalty = -5.0

    # dummy pt model parameter
    pt_sigma = 0.04

    # robot asset
    robot_usd_path = "/home/lim/Desktop/New Folder/vvv_flat.usd"

    # robot cfg
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path=robot_usd_path,
        ),
        actuators={},
    )


class VoltRunnerPtAlignEnv(DirectRLEnv):
    cfg: VoltRunnerPtAlignEnvCfg

    def __init__(self, cfg: VoltRunnerPtAlignEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.pt = torch.zeros((self.num_envs,), device=self.device)
        self.prev_pt = torch.zeros((self.num_envs,), device=self.device)
        self.delta_pt = torch.zeros((self.num_envs,), device=self.device)

        self.success_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.curr_distance = torch.zeros((self.num_envs,), device=self.device)
        self.prev_distance = torch.zeros((self.num_envs,), device=self.device)

        self.receiver_x = torch.zeros((self.num_envs,), device=self.device)
        self.receiver_y = torch.zeros((self.num_envs,), device=self.device)

        self.pt_history = torch.zeros((self.num_envs, 5), device=self.device)

        self.curr_vx = torch.zeros((self.num_envs,), device=self.device)
        self.curr_vy = torch.zeros((self.num_envs,), device=self.device)
        self.curr_wz = torch.zeros((self.num_envs,), device=self.device)

    def _create_workspace_panel(self):
        stage = self.scene.stage
        env_prim_path = "/World/envs/env_0"
        panel_path = f"{env_prim_path}/WorkspacePanel"

        panel = UsdGeom.Cube.Define(stage, panel_path)
        panel.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.001))
        panel.AddScaleOp().Set(
            Gf.Vec3f(
                self.cfg.workspace_size_x / 2.0,
                self.cfg.workspace_size_y / 2.0,
                0.001,
            )
        )

        prim = stage.GetPrimAtPath(panel_path)
        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(0.2, 0.6, 0.9)]
        )

    def _create_receiver_marker(self):
        stage = self.scene.stage
        env_prim_path = "/World/envs/env_0"
        marker_path = f"{env_prim_path}/ReceiverMarker"

        marker = UsdGeom.Cylinder.Define(stage, marker_path)
        marker.CreateRadiusAttr(self.cfg.receiver_radius)
        marker.CreateHeightAttr(self.cfg.receiver_thickness)
        marker.CreateAxisAttr("Z")
        marker.AddTranslateOp().Set(
            Gf.Vec3d(
                0.0,
                0.0,
                self.cfg.receiver_z,
            )
        )

        prim = stage.GetPrimAtPath(marker_path)
        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(0.1, 0.9, 0.1)]
        )

    def _update_receiver_marker_env0(self):
        if self.num_envs < 1:
            return

        stage = self.scene.stage
        marker_path = "/World/envs/env_0/ReceiverMarker"
        prim = stage.GetPrimAtPath(marker_path)
        if not prim.IsValid():
            return

        marker_geom = UsdGeom.Cylinder(prim)
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()

        target = Gf.Vec3d(
            float(self.receiver_x[0].item()),
            float(self.receiver_y[0].item()),
            self.cfg.receiver_z,
        )

        translate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            translate_op = marker_geom.AddTranslateOp()

        translate_op.Set(target)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self._create_workspace_panel()
        self._create_receiver_marker()

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_position_terms(self):
        root_pos = self.robot.data.root_pos_w

        x = root_pos[:, 0] - self.scene.env_origins[:, 0]
        y = root_pos[:, 1] - self.scene.env_origins[:, 1]

        dx = self.receiver_x - x
        dy = self.receiver_y - y
        dist = torch.sqrt(dx * dx + dy * dy)

        return x, y, dx, dy, dist

    def _compute_pt(self, dist: torch.Tensor) -> torch.Tensor:
        sigma = self.cfg.pt_sigma
        active_radius = 0.08

        pt = torch.exp(-(dist * dist) / (2.0 * sigma * sigma))
        pt = torch.where(dist <= active_radius, pt, torch.zeros_like(pt))
        return pt.clamp(0.0, 1.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        vx = self.actions[:, 0] * self.cfg.action_scale_vx
        vy = self.actions[:, 1] * self.cfg.action_scale_vy
        wz = self.actions[:, 2] * self.cfg.action_scale_wz

        self.curr_vx[:] = vx
        self.curr_vy[:] = vy
        self.curr_wz[:] = wz

        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        root_vel[:, 0] = vx
        root_vel[:, 1] = vy
        root_vel[:, 5] = wz

        self.robot.write_root_velocity_to_sim(root_vel)

    def _get_observations(self) -> dict:
        _, _, _, _, dist = self._compute_position_terms()

        self.curr_distance = dist
        self.pt = self._compute_pt(dist)
        self.delta_pt = self.pt - self.prev_pt

        self.pt_history[:, :-1] = self.pt_history[:, 1:].clone()
        self.pt_history[:, -1] = self.pt

        obs = torch.stack(
            (
                self.pt,
                self.delta_pt,
                self.pt_history[:, 0],
                self.pt_history[:, 1],
                self.pt_history[:, 2],
                self.pt_history[:, 3],
                self.pt_history[:, 4],
                self.curr_vx,
                self.curr_vy,
                self.curr_wz,
                self.prev_actions[:, 0],
                self.prev_actions[:, 1],
                self.prev_actions[:, 2],
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        speed_norm = torch.sqrt(self.curr_vx**2 + self.curr_vy**2 + self.curr_wz**2)
        action_norm_sq = torch.sum(self.actions * self.actions, dim=-1)

        x = self.robot.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        y = self.robot.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]

        out_of_bounds = (
            (torch.abs(x) > self.cfg.workspace_size_x / 2.0)
            | (torch.abs(y) > self.cfg.workspace_size_y / 2.0)
        )

        success_now = (
            (self.pt >= self.cfg.success_pt_threshold)
            & (speed_norm < self.cfg.success_speed_epsilon)
        )

        reward = (
            self.cfg.reward_alpha_dpt * self.delta_pt
            + self.cfg.reward_beta_pt * self.pt
            - self.cfg.reward_gamma_action * action_norm_sq
            - self.cfg.reward_delta_time
        )

        reward = reward + torch.where(
            success_now,
            torch.full_like(reward, self.cfg.reward_success_bonus),
            torch.zeros_like(reward),
        )

        reward = reward + torch.where(
            out_of_bounds,
            torch.full_like(reward, self.cfg.reward_fail_penalty),
            torch.zeros_like(reward),
        )

        self.prev_pt = self.pt.clone()
        self.prev_distance = self.curr_distance.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _, _, dist = self._compute_position_terms()
        self.curr_distance = dist
        self.pt = self._compute_pt(dist)

        speed_norm = torch.sqrt(self.curr_vx**2 + self.curr_vy**2 + self.curr_wz**2)

        out_of_bounds = (
            (torch.abs(x) > self.cfg.workspace_size_x / 2.0)
            | (torch.abs(y) > self.cfg.workspace_size_y / 2.0)
        )

        success_now = (
            (self.pt >= self.cfg.success_pt_threshold)
            & (speed_norm < self.cfg.success_speed_epsilon)
        )

        self.success_counter = torch.where(
            success_now,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter),
        )

        success_done = self.success_counter >= self.cfg.success_hold_steps
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = out_of_bounds | success_done
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        self.pt[env_ids] = 0.0
        self.prev_pt[env_ids] = 0.0
        self.delta_pt[env_ids] = 0.0
        self.pt_history[env_ids] = 0.0

        self.success_counter[env_ids] = 0

        self.curr_distance[env_ids] = 0.0
        self.prev_distance[env_ids] = 0.0

        self.curr_vx[env_ids] = 0.0
        self.curr_vy[env_ids] = 0.0
        self.curr_wz[env_ids] = 0.0

        # receiver spawn: same as runtime env
        rand_receiver_x = sample_uniform(
            self.cfg.receiver_x_min,
            self.cfg.receiver_x_max,
            (len(env_ids), 1),
            self.device,
        ).squeeze(-1)
        rand_receiver_y = sample_uniform(
            self.cfg.receiver_y_min,
            self.cfg.receiver_y_max,
            (len(env_ids), 1),
            self.device,
        ).squeeze(-1)

        self.receiver_x[env_ids] = rand_receiver_x
        self.receiver_y[env_ids] = rand_receiver_y

        root_state = self.robot.data.default_root_state[env_ids].clone()

        # robot spawn: near receiver for ALIGN training
        rand_offset_x = sample_uniform(
            self.cfg.align_spawn_offset_x_min,
            self.cfg.align_spawn_offset_x_max,
            (len(env_ids), 1),
            self.device,
        ).squeeze(-1)
        rand_offset_y = sample_uniform(
            self.cfg.align_spawn_offset_y_min,
            self.cfg.align_spawn_offset_y_max,
            (len(env_ids), 1),
            self.device,
        ).squeeze(-1)

        spawn_x = self.receiver_x[env_ids] + rand_offset_x
        spawn_y = self.receiver_y[env_ids] + rand_offset_y

        # workspace clamp
        x_limit = self.cfg.workspace_size_x / 2.0 - self.cfg.robot_size_x / 2.0
        y_limit = self.cfg.workspace_size_y / 2.0 - self.cfg.robot_size_y / 2.0
        spawn_x = torch.clamp(spawn_x, -x_limit, x_limit)
        spawn_y = torch.clamp(spawn_y, -y_limit, y_limit)

        root_state[:, 0] = self.scene.env_origins[env_ids, 0] + spawn_x
        root_state[:, 1] = self.scene.env_origins[env_ids, 1] + spawn_y
        root_state[:, 7:] = 0.0

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        dx = self.receiver_x[env_ids] - spawn_x
        dy = self.receiver_y[env_ids] - spawn_y
        dist = torch.sqrt(dx * dx + dy * dy)
        pt_now = self._compute_pt(dist)

        self.curr_distance[env_ids] = dist
        self.prev_distance[env_ids] = dist
        self.pt[env_ids] = pt_now
        self.prev_pt[env_ids] = pt_now
        self.pt_history[env_ids, :] = pt_now.unsqueeze(-1)

        if 0 in [int(i) for i in env_ids]:
            self._update_receiver_marker_env0()