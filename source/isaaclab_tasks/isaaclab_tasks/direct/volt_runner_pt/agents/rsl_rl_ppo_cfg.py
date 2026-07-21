from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class VoltRunnerPtPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    기존 VoltRunner Pt / Align / Slow task에서 사용하는
    기본 PPO runner configuration.

    기존 task 호환성을 유지하기 위해 남겨둔다.
    """

    # 각 environment에서 PPO update 전에 수집할 step 수
    # num_envs=128이면 iteration당 128 * 24 transitions 수집
    num_steps_per_env = 24

    # 기본 최대 학습 iteration
    # 실행 명령의 --max_iterations 값으로 덮어쓸 수 있음
    max_iterations = 5000

    # checkpoint 저장 주기
    save_interval = 100

    # 기존 task 로그 폴더 이름
    experiment_name = "volt_runner_pt_align_slow"

    # observation 값 범위가 제한되어 있으므로 자동 정규화 미사용
    empirical_normalization = False

    # ---------------------------------------------------------
    # Actor-Critic network
    # observation 13차원 -> hidden layers -> action 3차원
    # ---------------------------------------------------------
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    # ---------------------------------------------------------
    # PPO algorithm
    # ---------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class VoltRunnerPtRasterTransferSlowPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Raster Search가 Pt를 검출한 이후 PPO 정렬로 전환되는 상황을
    학습하기 위한 별도 PPO runner configuration.

    연결 task:
    Isaac-VoltRunner-Pt-Raster-Transfer-Slow-Direct-v0
    """

    # 각 environment에서 수집할 rollout 길이
    num_steps_per_env = 24

    # 본 학습 목표
    # 실행 명령에서도 --max_iterations=9999를 지정할 예정
    max_iterations = 9999

    # checkpoint 저장 주기
    save_interval = 100

    # 기존 모델 로그와 구분되는 새 experiment 이름
    experiment_name = "volt_runner_pt_raster_transfer_slow"

    # 기존 ROS observation 구조와 동일하게 정규화하지 않음
    empirical_normalization = False

    # ---------------------------------------------------------
    # Actor-Critic network
    # observation 13차원
    #
    # 0     : current Pt
    # 1     : delta Pt
    # 2~6   : Pt history 5개
    # 7~9   : 적용 속도 vx, vy, wz
    # 10~12 : previous action ax, ay, aw
    #
    # action 3차원
    # 실제 환경에서는 wz scale을 0으로 설정
    # ---------------------------------------------------------
    policy = RslRlPpoActorCriticCfg(
        # 기존보다 exploration을 조금 줄여
        # 저속 정렬에서 과도한 action 변화를 완화
        init_noise_std=0.35,

        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    # ---------------------------------------------------------
    # PPO algorithm
    # ---------------------------------------------------------
    algorithm = RslRlPpoAlgorithmCfg(
        # value function loss 비중
        value_loss_coef=1.0,

        # critic update clipping
        use_clipped_value_loss=True,

        # PPO policy clipping 범위
        clip_param=0.2,

        # 기존 0.01보다 조금 낮춰
        # 학습 후반의 불필요한 action 변동을 줄임
        entropy_coef=0.008,

        # 수집 데이터 반복 학습 횟수
        num_learning_epochs=5,

        # rollout mini-batch 개수
        num_mini_batches=4,

        # learning rate
        learning_rate=3.0e-4,

        # KL 기반 learning rate 조절
        schedule="adaptive",

        # discount factor
        gamma=0.99,

        # GAE lambda
        lam=0.95,

        # 목표 KL divergence
        desired_kl=0.01,

        # gradient clipping
        max_grad_norm=1.0,
    )
