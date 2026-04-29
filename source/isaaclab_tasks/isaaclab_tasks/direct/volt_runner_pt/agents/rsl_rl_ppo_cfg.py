from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class VoltRunnerPtPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # PPO가 한 번 업데이트하기 전에 각 env에서 모을 step 수
    # 예: num_envs=128이면 128 * 24 = 3072 transitions 수집 후 update
    num_steps_per_env = 24

    # PPO update를 총 몇 번 반복할지
    # 총 학습 transition 수 = num_envs * num_steps_per_env * max_iterations
    max_iterations = 2000

    # 몇 iteration마다 checkpoint를 저장할지
    save_interval = 100

    # 로그와 모델이 저장될 실험 이름
    # 저장 위치 예: logs/rsl_rl/volt_runner_pt/날짜_시간/
    experiment_name = "volt_runner_pt"

    # observation 자동 정규화 사용 여부
    # 현재 obs는 Pt/action/velocity 위주로 범위가 제한되어 있어 False로 둠
    empirical_normalization = False

    # -----------------------------
    # Actor-Critic network settings
    # -----------------------------
    policy = RslRlPpoActorCriticCfg(
        # 초기 action noise 표준편차
        # 학습 초반 exploration 정도를 결정함
        init_noise_std=0.5,

        # Actor MLP 구조
        # obs(13차원) -> 128 -> 128 -> action(3차원)
        actor_hidden_dims=[128, 128],

        # Critic MLP 구조
        # obs(13차원) -> 128 -> 128 -> value(1차원)
        critic_hidden_dims=[128, 128],

        # hidden layer activation function
        activation="elu",
    )

    # -----------------------------
    # PPO algorithm hyperparameters
    # -----------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        # critic value loss 비중
        value_loss_coef=1.0,

        # critic value update도 너무 크게 바뀌지 않도록 clipping
        use_clipped_value_loss=True,

        # PPO policy clipping 범위
        # 정책이 한 번에 너무 크게 바뀌는 것을 방지
        clip_param=0.2,

        # entropy 보상 비중
        # action 다양성/exploration 유지용
        entropy_coef=0.01,

        # 한 iteration에서 모은 데이터를 몇 번 반복 학습할지
        num_learning_epochs=5,

        # rollout 데이터를 몇 개 mini-batch로 나눌지
        num_mini_batches=4,

        # neural network learning rate
        learning_rate=3.0e-4,

        # KL 기준으로 learning rate를 자동 조절
        schedule="adaptive",

        # discount factor
        # 미래 reward를 얼마나 반영할지
        gamma=0.99,

        # GAE(lambda) 계수
        # advantage 계산 시 bias-variance 균형 조절
        lam=0.95,

        # policy update가 너무 커지지 않도록 보는 KL 기준값
        desired_kl=0.01,

        # gradient clipping
        # 학습 중 gradient 폭주 방지
        max_grad_norm=1.0,
    )