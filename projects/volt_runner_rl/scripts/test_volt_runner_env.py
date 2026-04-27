from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
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

    obs, _ = env.reset()
    step_count = 0

    try:
        while simulation_app.is_running():
            action = torch.tensor([[0.0, 0.0, 0.0]], device=env.unwrapped.device)

            obs, reward, terminated, truncated, info = env.step(action)
            simulation_app.update()

            step_count += 1

            if step_count % 20 == 0:
                policy_obs = obs["policy"][0].detach().cpu()

                pt = policy_obs[0].item()
                dpt = policy_obs[1].item()
                pt_hist = policy_obs[2:7].tolist()
                vx = policy_obs[7].item()
                vy = policy_obs[8].item()
                wz = policy_obs[9].item()
                prev_ax = policy_obs[10].item()
                prev_ay = policy_obs[11].item()
                prev_aw = policy_obs[12].item()

                rew_val = reward[0].item()
                term_val = bool(terminated[0].item())
                trunc_val = bool(truncated[0].item())

                env_unwrapped = env.unwrapped
                receiver_x = env_unwrapped.receiver_x[0].item()
                receiver_y = env_unwrapped.receiver_y[0].item()

                root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
                env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()

                local_x = (root_pos[0] - env_origin[0]).item()
                local_y = (root_pos[1] - env_origin[1]).item()

                print("=" * 70)
                print(f"step: {step_count}")
                print(f"obs_dim={policy_obs.shape[0]}")
                print(f"local_x={local_x:.4f}, local_y={local_y:.4f}")
                print(f"receiver_x={receiver_x:.4f}, receiver_y={receiver_y:.4f}")
                print(f"pt={pt:.4f}, dpt={dpt:.4f}")
                print(f"pt_history={[round(x, 4) for x in pt_hist]}")
                print(f"vel=({vx:.4f}, {vy:.4f}, {wz:.4f})")
                print(f"prev_action=({prev_ax:.4f}, {prev_ay:.4f}, {prev_aw:.4f})")
                print(f"reward={rew_val:.4f}")
                print(f"terminated={term_val}, truncated={trunc_val}")

            if terminated[0].item() or truncated[0].item():
                env_unwrapped = env.unwrapped
                receiver_x = env_unwrapped.receiver_x[0].item()
                receiver_y = env_unwrapped.receiver_y[0].item()

                root_pos = env_unwrapped.robot.data.root_pos_w[0].detach().cpu()
                env_origin = env_unwrapped.scene.env_origins[0].detach().cpu()

                local_x = (root_pos[0] - env_origin[0]).item()
                local_y = (root_pos[1] - env_origin[1]).item()

                print("-" * 70)
                print(f"RESET EVENT at step {step_count}")
                print(f"before reset local_x={local_x:.4f}, local_y={local_y:.4f}")
                print(f"before reset receiver_x={receiver_x:.4f}, receiver_y={receiver_y:.4f}")
                print(f"terminated={bool(terminated[0].item())}, truncated={bool(truncated[0].item())}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()