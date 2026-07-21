# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .volt_runner_pt_align_slow_env import (
    VoltRunnerPtAlignSlowEnv,
    VoltRunnerPtAlignSlowEnvCfg,
)


@configclass
class VoltRunnerPtRasterTransferSlowEnvCfg(VoltRunnerPtAlignSlowEnvCfg):
    """
    Raster Search가 약한 Pt를 검출한 이후의 PPO 정렬 상태를
    학습하기 위한 저속 환경.

    Raster 경로 자체는 PPO가 학습하지 않는다.

    PPO는 여러 초기 Pt 범위에서 시작하여 다음을 학습한다.

    1. 낮은 Pt에서 상승 방향 탐색
    2. Pt 상승 방향 유지
    3. Pt가 감소하면 부드럽게 방향 수정
    4. 높은 Pt에서 감속 및 정지
    """

    # =========================================================
    # Environment
    # =========================================================

    # physics 120 Hz / decimation 12 = policy 10 Hz
    decimation = 12

    # 저속 이동을 고려해 40초로 설정
    episode_length_s = 40.0

    # 기존 ROS 노드 및 모델 구조와 호환
    action_space = 3
    observation_space = 13
    state_space = 0

    # =========================================================
    # PPO action scaling
    # 실제 적용 명령과 동일한 값
    # =========================================================

    action_scale_vx = 0.020
    action_scale_vy = 0.020

    # 첫 비교 학습에서는 yaw 회전 비활성화
    action_scale_wz = 0.0

    # =========================================================
    # Initial Pt groups
    # =========================================================

    # Group 1: 약한 신호
    initial_pt_group1_min = 0.25
    initial_pt_group1_max = 0.40
    initial_pt_group1_ratio = 0.25

    # Group 2: 실제 문제 집중 구간
    initial_pt_group2_min = 0.40
    initial_pt_group2_max = 0.55
    initial_pt_group2_ratio = 0.40

    # Group 3: 중심 접근 구간
    initial_pt_group3_min = 0.55
    initial_pt_group3_max = 0.68
    initial_pt_group3_ratio = 0.25

    # Group 4: 미세 정렬 구간
    initial_pt_group4_min = 0.68
    initial_pt_group4_max = 0.74

    # =========================================================
    # Success condition
    # 실제 로봇 실행 기준과 우선 일치
    # =========================================================

    success_pt_threshold = 0.75
    success_hold_steps = 3

    # 초기 학습에서는 성공 정지 조건을 약간 완화
    success_speed_epsilon = 0.007

    # =========================================================
    # Reward weights
    # =========================================================

    # Pt 증가량 보상
    reward_alpha_dpt = 10.0

    # 현재 Pt 보상
    reward_beta_pt = 0.15

    # action 크기 페널티
    reward_gamma_action = 0.015

    # 시간 페널티
    reward_delta_time = 0.005

    # 성공 보너스
    reward_success_bonus = 6.0

    # 영역 이탈 페널티
    reward_fail_penalty = -5.0

    # action 급변 페널티
    reward_smooth_action = 0.10

    # Pt 감소 페널티
    reward_pt_decrease = 2.0

    # 낮은 Pt에서 정지할 때의 약한 페널티
    reward_low_stop_penalty = -0.05

    # 현재 Pt가 에피소드 초기 Pt보다 낮을 때 지속적으로 부과
    reward_below_initial_pt = 1.2

    # Pt가 거의 사라진 영역에서 배회하지 않도록 부과
    reward_very_low_pt = 0.20
    very_low_pt_threshold = 0.10

    # =========================================================
    # Command smoothing
    #
    # applied = alpha * target
    #         + (1-alpha) * previous_applied
    #
    # 1.0이면 즉시 적용
    # 작을수록 부드럽게 적용
    # =========================================================

    command_smoothing_alpha = 0.50


class VoltRunnerPtRasterTransferSlowEnv(VoltRunnerPtAlignSlowEnv):
    cfg: VoltRunnerPtRasterTransferSlowEnvCfg

    def __init__(
        self,
        cfg: VoltRunnerPtRasterTransferSlowEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # 실제 시뮬레이터에 적용되는 smoothing action
        self.applied_actions = torch.zeros(
            (self.num_envs, self.cfg.action_space),
            device=self.device,
        )

        # =====================================================
        # Episode metric buffers
        # =====================================================

        # 에피소드 시작 Pt
        self.episode_initial_pt = torch.zeros(
            (self.num_envs,),
            device=self.device,
        )

        # 에피소드 내 최고 Pt
        self.episode_peak_pt = torch.zeros(
            (self.num_envs,),
            device=self.device,
        )

        # 에피소드 내 action 변화량 누적
        self.episode_action_change_sum = torch.zeros(
            (self.num_envs,),
            device=self.device,
        )

        # Pt가 감소한 스텝 수
        self.episode_pt_decrease_steps = torch.zeros(
            (self.num_envs,),
            device=self.device,
        )

        # metric 계산용 실제 누적 step 수
        self.episode_metric_steps = torch.zeros(
            (self.num_envs,),
            device=self.device,
        )

        # 초기 Pt 그룹 번호
        # 0: 0.25~0.40
        # 1: 0.40~0.60
        # 2: 0.60~0.80
        # 3: 0.80~0.90
        self.initial_pt_group = torch.zeros(
            (self.num_envs,),
            dtype=torch.long,
            device=self.device,
        )

        # 마지막 종료 원인 저장
        self.last_success_done = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )

        self.last_time_out = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )

        self.last_out_of_bounds = torch.zeros(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device,
        )

    def _sample_initial_pt(
        self,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        설정된 비율에 따라 초기 Pt와 그룹 번호를 반환한다.

        Group 0: Pt 0.25~0.40, 25%
        Group 1: Pt 0.40~0.60, 40%
        Group 2: Pt 0.60~0.80, 25%
        Group 3: Pt 0.80~0.90, 10%
        """

        selector = torch.rand(
            (count,),
            device=self.device,
        )

        initial_pt = torch.empty(
            (count,),
            device=self.device,
        )

        group_id = torch.empty(
            (count,),
            dtype=torch.long,
            device=self.device,
        )

        group1_end = self.cfg.initial_pt_group1_ratio

        group2_end = (
            group1_end
            + self.cfg.initial_pt_group2_ratio
        )

        group3_end = (
            group2_end
            + self.cfg.initial_pt_group3_ratio
        )

        group1_mask = selector < group1_end

        group2_mask = (
            (selector >= group1_end)
            & (selector < group2_end)
        )

        group3_mask = (
            (selector >= group2_end)
            & (selector < group3_end)
        )

        group4_mask = selector >= group3_end

        group1_count = int(group1_mask.sum().item())
        group2_count = int(group2_mask.sum().item())
        group3_count = int(group3_mask.sum().item())
        group4_count = int(group4_mask.sum().item())

        if group1_count > 0:
            initial_pt[group1_mask] = sample_uniform(
                self.cfg.initial_pt_group1_min,
                self.cfg.initial_pt_group1_max,
                (group1_count, 1),
                self.device,
            ).squeeze(-1)

            group_id[group1_mask] = 0

        if group2_count > 0:
            initial_pt[group2_mask] = sample_uniform(
                self.cfg.initial_pt_group2_min,
                self.cfg.initial_pt_group2_max,
                (group2_count, 1),
                self.device,
            ).squeeze(-1)

            group_id[group2_mask] = 1

        if group3_count > 0:
            initial_pt[group3_mask] = sample_uniform(
                self.cfg.initial_pt_group3_min,
                self.cfg.initial_pt_group3_max,
                (group3_count, 1),
                self.device,
            ).squeeze(-1)

            group_id[group3_mask] = 2

        if group4_count > 0:
            initial_pt[group4_mask] = sample_uniform(
                self.cfg.initial_pt_group4_min,
                self.cfg.initial_pt_group4_max,
                (group4_count, 1),
                self.device,
            ).squeeze(-1)

            group_id[group4_mask] = 3

        initial_pt = initial_pt.clamp(
            1.0e-4,
            0.9999,
        )

        return initial_pt, group_id

    def _pt_to_distance(
        self,
        initial_pt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian Pt 모델 역함수.

        Pt = exp(-d^2 / (2 sigma^2))

        d = sigma * sqrt(-2 ln(Pt))
        """

        sigma = float(self.cfg.pt_sigma)

        distance = sigma * torch.sqrt(
            -2.0
            * torch.log(
                initial_pt.clamp(
                    1.0e-4,
                    0.9999,
                )
            )
        )

        return distance

    def _apply_action(self) -> None:
        """
        PPO action을 smoothing한 뒤 실제 root velocity에 적용한다.

        action 출력 구조는 3차원을 유지하지만,
        현재 학습에서는 wz를 항상 0으로 만든다.
        """

        alpha = float(
            self.cfg.command_smoothing_alpha
        )

        target_actions = self.actions.clone()

        # yaw action 비활성화
        target_actions[:, 2] = 0.0

        self.applied_actions[:] = (
            alpha * target_actions
            + (1.0 - alpha) * self.applied_actions
        )

        vx = (
            self.applied_actions[:, 0]
            * self.cfg.action_scale_vx
        )

        vy = (
            self.applied_actions[:, 1]
            * self.cfg.action_scale_vy
        )

        wz = torch.zeros_like(vx)

        self.curr_vx[:] = vx
        self.curr_vy[:] = vy
        self.curr_wz[:] = wz

        root_vel = torch.zeros(
            (self.num_envs, 6),
            device=self.device,
        )

        root_vel[:, 0] = vx
        root_vel[:, 1] = vy
        root_vel[:, 5] = 0.0

        self.robot.write_root_velocity_to_sim(
            root_vel
        )

    def _get_rewards(self) -> torch.Tensor:
        """
        Reward 계산과 episode metric 누적을 함께 수행한다.
        """

        speed_norm = torch.sqrt(
            self.curr_vx**2
            + self.curr_vy**2
        )

        # -----------------------------------------------------
        # Action magnitude
        # -----------------------------------------------------

        action_penalty = (
            self.actions[:, 0] ** 2
            + self.actions[:, 1] ** 2
        )

        # -----------------------------------------------------
        # Action change
        # -----------------------------------------------------

        action_delta = (
            self.actions
            - self.prev_actions
        )

        action_change = torch.sqrt(
            action_delta[:, 0] ** 2
            + action_delta[:, 1] ** 2
            + action_delta[:, 2] ** 2
        )

        smooth_action_penalty = (
            action_delta[:, 0] ** 2
            + action_delta[:, 1] ** 2
            + action_delta[:, 2] ** 2
        )

        # -----------------------------------------------------
        # Bounds
        # -----------------------------------------------------

        x = (
            self.robot.data.root_pos_w[:, 0]
            - self.scene.env_origins[:, 0]
        )

        y = (
            self.robot.data.root_pos_w[:, 1]
            - self.scene.env_origins[:, 1]
        )

        out_of_bounds = (
            (
                torch.abs(x)
                > self.cfg.workspace_size_x / 2.0
            )
            |
            (
                torch.abs(y)
                > self.cfg.workspace_size_y / 2.0
            )
        )

        # -----------------------------------------------------
        # Success
        # -----------------------------------------------------

        success_now = (
            (
                self.pt
                >= self.cfg.success_pt_threshold
            )
            &
            (
                speed_norm
                < self.cfg.success_speed_epsilon
            )
        )

        # -----------------------------------------------------
        # Basic reward
        # -----------------------------------------------------

        reward = (
            self.cfg.reward_alpha_dpt
            * self.delta_pt

            + self.cfg.reward_beta_pt
            * self.pt

            - self.cfg.reward_gamma_action
            * action_penalty

            - self.cfg.reward_smooth_action
            * smooth_action_penalty

            - self.cfg.reward_delta_time
        )

        # -----------------------------------------------------
        # Pt decrease penalty
        # -----------------------------------------------------

        pt_decrease = torch.clamp(
            -self.delta_pt,
            min=0.0,
        )

        reward = (
            reward
            - self.cfg.reward_pt_decrease
            * pt_decrease
        )

        # -----------------------------------------------------
        # Below-initial Pt penalty
        #
        # 한 번 코일에서 멀어진 뒤 delta Pt가 0이 되더라도,
        # 초기 Pt보다 낮은 상태가 계속되면 매 step 페널티를 준다.
        # -----------------------------------------------------

        below_initial_pt = torch.clamp(
            self.episode_initial_pt - self.pt,
            min=0.0,
        )

        reward = (
            reward
            - self.cfg.reward_below_initial_pt
            * below_initial_pt
        )

        # -----------------------------------------------------
        # Very-low Pt penalty
        #
        # Pt가 거의 0인 영역에서 오래 배회하는 정책을 억제한다.
        # -----------------------------------------------------

        very_low_pt_penalty = torch.where(
            self.pt < self.cfg.very_low_pt_threshold,
            torch.full_like(
                reward,
                self.cfg.reward_very_low_pt,
            ),
            torch.zeros_like(reward),
        )

        reward = reward - very_low_pt_penalty

        # -----------------------------------------------------
        # High Pt progress reward
        # -----------------------------------------------------

        peak_denominator = max(
            self.cfg.success_pt_threshold - 0.60,
            1.0e-6,
        )

        peak_progress = torch.clamp(
            (
                self.pt - 0.60
            )
            / peak_denominator,
            0.0,
            1.0,
        )

        peak_reward = (
            2.0
            * peak_progress**2
        )

        reward = reward + peak_reward

        # -----------------------------------------------------
        # High Pt bonus
        # -----------------------------------------------------

        high_pt_bonus = torch.where(
            self.pt
            >= self.cfg.success_pt_threshold,

            torch.full_like(
                reward,
                1.5,
            ),

            torch.zeros_like(
                reward,
            ),
        )

        reward = reward + high_pt_bonus

        # -----------------------------------------------------
        # Low Pt stop penalty
        # -----------------------------------------------------

        low_stop_penalty = torch.where(
            (
                self.pt
                < self.cfg.success_pt_threshold
            )
            &
            (
                speed_norm
                < self.cfg.success_speed_epsilon
            ),

            torch.full_like(
                reward,
                self.cfg.reward_low_stop_penalty,
            ),

            torch.zeros_like(
                reward,
            ),
        )

        reward = reward + low_stop_penalty

        # -----------------------------------------------------
        # Success bonus
        # -----------------------------------------------------

        reward = reward + torch.where(
            success_now,

            torch.full_like(
                reward,
                self.cfg.reward_success_bonus,
            ),

            torch.zeros_like(
                reward,
            ),
        )

        # -----------------------------------------------------
        # Failure penalty
        # -----------------------------------------------------

        reward = reward + torch.where(
            out_of_bounds,

            torch.full_like(
                reward,
                self.cfg.reward_fail_penalty,
            ),

            torch.zeros_like(
                reward,
            ),
        )

        # =====================================================
        # Episode metrics
        # =====================================================

        self.episode_peak_pt[:] = torch.maximum(
            self.episode_peak_pt,
            self.pt,
        )

        self.episode_action_change_sum += (
            action_change
        )

        self.episode_pt_decrease_steps += (
            self.delta_pt < 0.0
        ).float()

        self.episode_metric_steps += 1.0

        # 다음 step의 delta Pt 계산 기준
        self.prev_pt = self.pt.clone()

        self.prev_distance = (
            self.curr_distance.clone()
        )

        return reward

    def _get_dones(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        종료 조건을 계산하고 종료 원인을 metric buffer에 저장한다.
        """

        x, y, _, _, dist = (
            self._compute_position_terms()
        )

        self.curr_distance = dist

        self.pt = self._compute_pt(
            dist
        )

        speed_norm = torch.sqrt(
            self.curr_vx**2
            + self.curr_vy**2
        )

        out_of_bounds = (
            (
                torch.abs(x)
                > self.cfg.workspace_size_x / 2.0
            )
            |
            (
                torch.abs(y)
                > self.cfg.workspace_size_y / 2.0
            )
        )

        success_now = (
            (
                self.pt
                >= self.cfg.success_pt_threshold
            )
            &
            (
                speed_norm
                < self.cfg.success_speed_epsilon
            )
        )

        self.success_counter = torch.where(
            success_now,

            self.success_counter + 1,

            torch.zeros_like(
                self.success_counter,
            ),
        )

        success_done = (
            self.success_counter
            >= self.cfg.success_hold_steps
        )

        time_out = (
            self.episode_length_buf
            >= self.max_episode_length - 1
        )

        terminated = (
            out_of_bounds
            | success_done
        )

        self.last_success_done[:] = success_done
        self.last_time_out[:] = time_out
        self.last_out_of_bounds[:] = out_of_bounds

        return terminated, time_out

    def _write_episode_metrics(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        """
        종료된 environment들의 episode metric을
        self.extras["log"]에 기록한다.

        RSL-RL wrapper가 이 값을 TensorBoard scalar로 저장한다.
        """

        if env_ids.numel() == 0:
            return

        # 실제 episode를 한 번이라도 진행한 환경만 사용
        valid_mask = (
            self.episode_metric_steps[env_ids]
            > 0.0
        )

        finished_ids = env_ids[valid_mask]

        if finished_ids.numel() == 0:
            return

        steps = torch.clamp(
            self.episode_metric_steps[finished_ids],
            min=1.0,
        )

        success_values = (
            self.last_success_done[finished_ids]
            .float()
        )

        timeout_values = (
            self.last_time_out[finished_ids]
            .float()
        )

        out_of_bounds_values = (
            self.last_out_of_bounds[finished_ids]
            .float()
        )

        initial_pt_values = (
            self.episode_initial_pt[finished_ids]
        )

        final_pt_values = (
            self.pt[finished_ids]
        )

        peak_pt_values = (
            self.episode_peak_pt[finished_ids]
        )

        pt_gain_values = (
            final_pt_values
            - initial_pt_values
        )

        action_change_values = (
            self.episode_action_change_sum[
                finished_ids
            ]
        )

        mean_action_change_values = (
            action_change_values
            / steps
        )

        pt_decrease_fraction_values = (
            self.episode_pt_decrease_steps[
                finished_ids
            ]
            / steps
        )

        episode_length_values = steps

        log = {
            "Episode/success_rate":
                success_values.mean(),

            "Episode/timeout_rate":
                timeout_values.mean(),

            "Episode/out_of_bounds_rate":
                out_of_bounds_values.mean(),

            "Episode/initial_pt":
                initial_pt_values.mean(),

            "Episode/final_pt":
                final_pt_values.mean(),

            "Episode/peak_pt":
                peak_pt_values.mean(),

            "Episode/pt_gain":
                pt_gain_values.mean(),

            "Episode/action_change_sum":
                action_change_values.mean(),

            "Episode/mean_action_change":
                mean_action_change_values.mean(),

            "Episode/pt_decrease_fraction":
                pt_decrease_fraction_values.mean(),

            "Episode/length":
                episode_length_values.mean(),
        }

        # -----------------------------------------------------
        # Initial Pt group-specific metrics
        # -----------------------------------------------------

        group_names = (
            "pt_025_040",
            "pt_040_055",
            "pt_055_068",
            "pt_068_074",
        )

        for group_index, group_name in enumerate(
            group_names
        ):
            group_mask = (
                self.initial_pt_group[finished_ids]
                == group_index
            )

            if torch.any(group_mask):
                group_success = (
                    success_values[group_mask]
                    .mean()
                )

                group_final_pt = (
                    final_pt_values[group_mask]
                    .mean()
                )

                group_pt_gain = (
                    pt_gain_values[group_mask]
                    .mean()
                )

                log[
                    f"Group/{group_name}_success_rate"
                ] = group_success

                log[
                    f"Group/{group_name}_final_pt"
                ] = group_final_pt

                log[
                    f"Group/{group_name}_pt_gain"
                ] = group_pt_gain

        self.extras["log"] = log

    def _reset_idx(
        self,
        env_ids: Sequence[int] | None,
    ):
        """
        종료 metric을 먼저 기록한 뒤 새로운 episode를 초기화한다.

        Receiver는 search 영역에 랜덤 생성한다.

        Robot은 목표 초기 Pt를 먼저 샘플링하고,
        해당 Pt에 대응하는 거리와 랜덤 방향각으로 배치한다.
        """

        if env_ids is None:
            env_ids_tensor = (
                self.robot._ALL_INDICES
            )
        else:
            env_ids_tensor = torch.as_tensor(
                env_ids,
                dtype=torch.long,
                device=self.device,
            )

        # =====================================================
        # Reset 전에 종료된 episode metric 기록
        # =====================================================

        if hasattr(
            self,
            "episode_metric_steps",
        ):
            self._write_episode_metrics(
                env_ids_tensor
            )

        # Parent AlignSlow reset은 임의 ±3 cm spawn까지 수행하므로
        # 호출하지 않는다.
        #
        # DirectRLEnv 기본 reset만 호출한 뒤
        # 아래에서 새 spawn을 직접 적용한다.
        DirectRLEnv._reset_idx(
            self,
            env_ids_tensor,
        )

        count = env_ids_tensor.numel()

        # =====================================================
        # Buffer reset
        # =====================================================

        self.actions[env_ids_tensor] = 0.0
        self.prev_actions[env_ids_tensor] = 0.0

        if hasattr(
            self,
            "applied_actions",
        ):
            self.applied_actions[
                env_ids_tensor
            ] = 0.0

        self.pt[env_ids_tensor] = 0.0
        self.prev_pt[env_ids_tensor] = 0.0
        self.delta_pt[env_ids_tensor] = 0.0

        self.pt_history[
            env_ids_tensor
        ] = 0.0

        self.success_counter[
            env_ids_tensor
        ] = 0

        self.curr_distance[
            env_ids_tensor
        ] = 0.0

        self.prev_distance[
            env_ids_tensor
        ] = 0.0

        self.curr_vx[
            env_ids_tensor
        ] = 0.0

        self.curr_vy[
            env_ids_tensor
        ] = 0.0

        self.curr_wz[
            env_ids_tensor
        ] = 0.0

        if hasattr(
            self,
            "episode_action_change_sum",
        ):
            self.episode_action_change_sum[
                env_ids_tensor
            ] = 0.0

            self.episode_pt_decrease_steps[
                env_ids_tensor
            ] = 0.0

            self.episode_metric_steps[
                env_ids_tensor
            ] = 0.0

            self.last_success_done[
                env_ids_tensor
            ] = False

            self.last_time_out[
                env_ids_tensor
            ] = False

            self.last_out_of_bounds[
                env_ids_tensor
            ] = False

        # =====================================================
        # Receiver spawn
        # =====================================================

        rand_receiver_x = sample_uniform(
            self.cfg.receiver_x_min,
            self.cfg.receiver_x_max,
            (count, 1),
            self.device,
        ).squeeze(-1)

        rand_receiver_y = sample_uniform(
            self.cfg.receiver_y_min,
            self.cfg.receiver_y_max,
            (count, 1),
            self.device,
        ).squeeze(-1)

        self.receiver_x[
            env_ids_tensor
        ] = rand_receiver_x

        self.receiver_y[
            env_ids_tensor
        ] = rand_receiver_y

        # =====================================================
        # Initial Pt sampling
        # =====================================================

        (
            target_initial_pt,
            sampled_group_id,
        ) = self._sample_initial_pt(
            count
        )

        spawn_distance = (
            self._pt_to_distance(
                target_initial_pt
            )
        )

        # Receiver 주변 모든 방향에서 시작
        theta = sample_uniform(
            -math.pi,
            math.pi,
            (count, 1),
            self.device,
        ).squeeze(-1)

        spawn_x = (
            self.receiver_x[env_ids_tensor]
            + spawn_distance
            * torch.cos(theta)
        )

        spawn_y = (
            self.receiver_y[env_ids_tensor]
            + spawn_distance
            * torch.sin(theta)
        )

        # =====================================================
        # Workspace clamp
        # =====================================================

        x_limit = (
            self.cfg.workspace_size_x / 2.0
            - self.cfg.robot_size_x / 2.0
        )

        y_limit = (
            self.cfg.workspace_size_y / 2.0
            - self.cfg.robot_size_y / 2.0
        )

        spawn_x = torch.clamp(
            spawn_x,
            -x_limit,
            x_limit,
        )

        spawn_y = torch.clamp(
            spawn_y,
            -y_limit,
            y_limit,
        )

        # =====================================================
        # Robot state
        # =====================================================

        root_state = (
            self.robot.data.default_root_state[
                env_ids_tensor
            ].clone()
        )

        root_state[:, 0] = (
            self.scene.env_origins[
                env_ids_tensor,
                0,
            ]
            + spawn_x
        )

        root_state[:, 1] = (
            self.scene.env_origins[
                env_ids_tensor,
                1,
            ]
            + spawn_y
        )

        # Raster -> RL 전환 시 zero command를 먼저 보내므로
        # PPO는 정지 상태에서 시작
        root_state[:, 7:] = 0.0

        self.robot.write_root_pose_to_sim(
            root_state[:, :7],
            env_ids_tensor,
        )

        self.robot.write_root_velocity_to_sim(
            root_state[:, 7:],
            env_ids_tensor,
        )

        joint_pos = (
            self.robot.data.default_joint_pos[
                env_ids_tensor
            ].clone()
        )

        joint_vel = (
            self.robot.data.default_joint_vel[
                env_ids_tensor
            ].clone()
        )

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            None,
            env_ids_tensor,
        )

        # =====================================================
        # Initial Pt calculation
        # =====================================================

        dx = (
            self.receiver_x[env_ids_tensor]
            - spawn_x
        )

        dy = (
            self.receiver_y[env_ids_tensor]
            - spawn_y
        )

        dist = torch.sqrt(
            dx * dx
            + dy * dy
        )

        pt_now = self._compute_pt(
            dist
        )

        self.curr_distance[
            env_ids_tensor
        ] = dist

        self.prev_distance[
            env_ids_tensor
        ] = dist

        self.pt[
            env_ids_tensor
        ] = pt_now

        self.prev_pt[
            env_ids_tensor
        ] = pt_now

        self.delta_pt[
            env_ids_tensor
        ] = 0.0

        # Raster에서 이미 Pt history를 수집한 상태를 단순 모사
        self.pt_history[
            env_ids_tensor,
            :,
        ] = pt_now.unsqueeze(-1)

        # =====================================================
        # Episode metric initialization
        # =====================================================

        if hasattr(
            self,
            "episode_initial_pt",
        ):
            self.episode_initial_pt[
                env_ids_tensor
            ] = pt_now

            self.episode_peak_pt[
                env_ids_tensor
            ] = pt_now

            self.initial_pt_group[
                env_ids_tensor
            ] = sampled_group_id

        if 0 in [
            int(i)
            for i in env_ids_tensor
        ]:
            self._update_receiver_marker_env0()
