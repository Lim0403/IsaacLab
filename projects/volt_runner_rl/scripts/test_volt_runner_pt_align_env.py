from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    task_name = "Isaac-VoltRunner-Pt-Align-Direct-v0"

    print("Parsing env config...")
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=4)

    print("Creating environment...")
    env = gym.make(task_name, cfg=env_cfg)
    env_unwrapped = env.unwrapped

    print("ENV_OK:", env)

    # reset 1회
    obs, _ = env.reset()

    # 0 action으로 한 번만 step 해서 상태 안정화
    zero_action = torch.zeros((env_unwrapped.num_envs, 3), device=env_unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(zero_action)
    simulation_app.update()

    print("\n" + "=" * 90)
    print("ALIGN ENV RESET CHECK")
    print("=" * 90)

    root_pos = env_unwrapped.robot.data.root_pos_w.detach().cpu()
    env_origins = env_unwrapped.scene.env_origins.detach().cpu()
    receiver_x = env_unwrapped.receiver_x.detach().cpu()
    receiver_y = env_unwrapped.receiver_y.detach().cpu()

    for i in range(env_unwrapped.num_envs):
        robot_x = (root_pos[i, 0] - env_origins[i, 0]).item()
        robot_y = (root_pos[i, 1] - env_origins[i, 1]).item()

        rx = receiver_x[i].item()
        ry = receiver_y[i].item()

        dx = rx - robot_x
        dy = ry - robot_y
        dist = (dx * dx + dy * dy) ** 0.5

        ok_dx = abs(dx) <= 0.15 + 1e-6
        ok_dy = abs(dy) <= 0.15 + 1e-6
        ok_dist = dist <= 0.2125

        print(f"[env_{i}]")
        print(f"  robot    = ({robot_x:.4f}, {robot_y:.4f})")
        print(f"  receiver = ({rx:.4f}, {ry:.4f})")
        print(f"  dx, dy   = ({dx:.4f}, {dy:.4f})")
        print(f"  dist     = {dist:.4f}")
        print(f"  check    = abs(dx)<=0.15:{ok_dx}, abs(dy)<=0.15:{ok_dy}, dist<=0.2125:{ok_dist}")
        print("-" * 90)

    print("Window stays open. Close with Ctrl+C.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()