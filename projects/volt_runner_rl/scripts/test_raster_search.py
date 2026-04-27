from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class RasterSearchController:
    def __init__(
        self,
        y_min: float,
        y_max: float,
        x_min: float,
        x_max: float,
        lane_spacing_x: float,
        search_vy: float,
        search_vx: float,
        pos_tol: float,
        detect_pt_threshold: float,
        detect_confirm_steps: int,
    ):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.lane_spacing_x = lane_spacing_x
        self.search_vy = search_vy
        self.search_vx = search_vx
        self.pos_tol = pos_tol
        self.detect_pt_threshold = detect_pt_threshold
        self.detect_confirm_steps = detect_confirm_steps

        self.mode = "SWEEP_Y"
        self.dir_y = +1
        self.target_y = y_max
        self.target_x = None
        self.detect_counter = 0
        self.detected_once = False

    def reset(self, current_x: float):
        self.mode = "SWEEP_Y"
        self.dir_y = +1
        self.target_y = self.y_max
        self.target_x = max(self.x_min, min(current_x, self.x_max))
        self.detect_counter = 0
        self.detected_once = False

    def update(self, x: float, y: float, pt: float):
        if pt >= self.detect_pt_threshold:
            self.detect_counter += 1
        else:
            self.detect_counter = 0

        if self.detect_counter >= self.detect_confirm_steps:
            if not self.detected_once:
                self.detected_once = True
                return "ALIGN_READY", 0.0, 0.0, 0.0, True
            return "ALIGN_READY", 0.0, 0.0, 0.0, False

        if self.mode == "SWEEP_Y":
            if abs(y - self.target_y) <= self.pos_tol:
                self.mode = "STEP_X"
                next_x = self.target_x + self.lane_spacing_x
                self.target_x = min(next_x, self.x_max)
                return self.mode, 0.0, 0.0, 0.0, False

            vx = 0.0
            vy = self.search_vy * self.dir_y
            wz = 0.0
            return self.mode, vx, vy, wz, False

        elif self.mode == "STEP_X":
            if abs(x - self.target_x) <= self.pos_tol:
                self.dir_y *= -1
                self.target_y = self.y_max if self.dir_y > 0 else self.y_min
                self.mode = "SWEEP_Y"
                return self.mode, 0.0, 0.0, 0.0, False

            vx = self.search_vx
            vy = 0.0
            wz = 0.0
            return self.mode, vx, vy, wz, False

        return self.mode, 0.0, 0.0, 0.0, False


def clamp_normalized_action(vx, vy, wz, env_cfg):
    ax = max(-1.0, min(1.0, vx / env_cfg.action_scale_vx))
    ay = max(-1.0, min(1.0, vy / env_cfg.action_scale_vy))
    aw = max(-1.0, min(1.0, wz / env_cfg.action_scale_wz))
    return ax, ay, aw


def main():
    task_name = "Isaac-VoltRunner-Pt-Direct-v0"

    print("Parsing env config...")
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=1)

    print("Creating environment...")
    env = gym.make(task_name, cfg=env_cfg)
    env_unwrapped = env.unwrapped

    print("ENV_OK:", env)
    print("Window will stay open. Close it with Ctrl+C in terminal.")

    # 중앙 탐색영역 130 x 70 cm
    search_x_min = -0.65
    search_x_max = 0.65
    search_y_min = -0.35
    search_y_max = 0.35

    controller = RasterSearchController(
        y_min=search_y_min,
        y_max=search_y_max,
        x_min=search_x_min,
        x_max=search_x_max,
        lane_spacing_x=0.08,      # x 방향 줄 간격
        search_vy=0.20,           # 먼저 y로 길게 sweep
        search_vx=0.12,           # x로 한 칸 전진
        pos_tol=0.02,
        detect_pt_threshold=0.07,
        detect_confirm_steps=3,
    )

    obs, _ = env.reset()

    root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
    env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
    local_x = (root_pos[0] - env_origin[0]).item()
    local_y = (root_pos[1] - env_origin[1]).item()
    controller.reset(local_x)

    step_count = 0

    try:
        while simulation_app.is_running():
            policy_obs = obs["policy"][0].detach().cpu()
            pt = policy_obs[0].item()

            root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
            env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
            local_x = (root_pos[0] - env_origin[0]).item()
            local_y = (root_pos[1] - env_origin[1]).item()

            mode, vx_cmd, vy_cmd, wz_cmd, just_detected = controller.update(local_x, local_y, pt)

            if just_detected:
                print("=" * 70)
                print(f"DETECTED at step {step_count}")
                print(f"local_x={local_x:.4f}, local_y={local_y:.4f}, pt={pt:.4f}")
                print("Raster search finished. Later, PPO policy will take over here.")

            if mode == "ALIGN_READY":
                action = torch.tensor([[0.0, 0.0, 0.0]], device=env_unwrapped.device)
            else:
                ax, ay, aw = clamp_normalized_action(vx_cmd, vy_cmd, wz_cmd, env_cfg)
                action = torch.tensor([[ax, ay, aw]], device=env_unwrapped.device)

            obs, reward, terminated, truncated, info = env.step(action)
            simulation_app.update()
            step_count += 1

            if step_count % 10 == 0:
                receiver_x = env_unwrapped.receiver_x[0].item()
                receiver_y = env_unwrapped.receiver_y[0].item()
                print("-" * 70)
                print(f"step={step_count}, mode={mode}")
                print(f"robot=({local_x:.4f}, {local_y:.4f})")
                print(f"receiver=({receiver_x:.4f}, {receiver_y:.4f})")
                print(f"pt={pt:.4f}, reward={reward[0].item():.4f}")
                print(f"cmd=({vx_cmd:.3f}, {vy_cmd:.3f}, {wz_cmd:.3f})")

            if terminated[0].item() or truncated[0].item():
                print("=" * 70)
                print(f"RESET EVENT at step {step_count}")

                obs, _ = env.reset()

                root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
                env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()
                local_x = (root_pos[0] - env_origin[0]).item()
                controller.reset(local_x)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()