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
from pxr import Gf, Sdf, UsdGeom


@configclass
class VoltRunnerPtEnvCfg(DirectRLEnvCfg):
    """Configuration for the Volt Runner Pt environment."""

    # env
    decimation = 2
    episode_length_s = 20.0
    action_space = 3
    observation_space = 10
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # workspace settings
    workspace_size_x = 2.0
    workspace_size_y = 1.2

    # robot reset range inside workspace
    reset_pos_x_min = -0.8
    reset_pos_x_max = 0.8
    reset_pos_y_min = -0.4
    reset_pos_y_max = 0.4

    # receiver coil position
    receiver_x = 0.0
    receiver_y = 0.0
    receiver_z = 0.203

    # receiver marker size
    receiver_radius = 0.06
    receiver_thickness = 0.002

    # action scaling
    action_scale_vx = 0.5
    action_scale_vy = 0.5
    action_scale_wz = 1.5

    # success condition
    success_pt_threshold = 0.90
    success_hold_steps = 10

    # reward weights
    reward_alpha_dpt = 2.0
    reward_beta_pt = 1.0
    reward_gamma_action = 0.05
    reward_delta_time = 0.01
    reward_success_bonus = 5.0
    reward_oob_penalty = -5.0

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


class VoltRunnerPtEnv(DirectRLEnv):
    cfg: VoltRunnerPtEnvCfg

    def __init__(self, cfg: VoltRunnerPtEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.pt = torch.zeros((self.num_envs,), device=self.device)
        self.prev_pt = torch.zeros((self.num_envs,), device=self.device)
        self.success_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _create_workspace_panel(self):
        """Create a colored rectangular panel showing the valid work area."""
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
        """Create a receiver coil marker."""
        stage = self.scene.stage

        env_prim_path = "/World/envs/env_0"
        marker_path = f"{env_prim_path}/ReceiverMarker"

        marker = UsdGeom.Cylinder.Define(stage, marker_path)
        marker.CreateRadiusAttr(self.cfg.receiver_radius)
        marker.CreateHeightAttr(self.cfg.receiver_thickness)
        marker.CreateAxisAttr("Z")
        marker.AddTranslateOp().Set(
            Gf.Vec3d(
                self.cfg.receiver_x,
                self.cfg.receiver_y,
                self.cfg.receiver_z,
            )
        )

        prim = stage.GetPrimAtPath(marker_path)
        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(0.1, 0.9, 0.1)]
        )

    def _setup_scene(self):
        # create robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add workspace panel
        self._create_workspace_panel()

        # add receiver marker
        self._create_receiver_marker()

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # register articulation in scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # TODO: apply body velocity [vx, vy, wz] directly to robot root
        pass

    def _get_observations(self) -> dict:
        # placeholder observation
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # placeholder reward
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._ALL_INDICES
        super()._reset_idx(env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.pt[env_ids] = 0.0
        self.prev_pt[env_ids] = 0.0
        self.success_counter[env_ids] = 0