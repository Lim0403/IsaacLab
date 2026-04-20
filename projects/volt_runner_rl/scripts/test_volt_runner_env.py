from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import time
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    task_name = "Isaac-VoltRunner-Pt-Direct-v0"

    print("Parsing env config...")
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=1)

    print("Creating environment...")
    env = gym.make(task_name, cfg=env_cfg)

    print("ENV_OK:", env)
    print("Window will stay open. Close it with Ctrl+C in terminal.")

    try:
        while simulation_app.is_running():
            env.reset()
            simulation_app.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()