# mahjong_agent/config_multithread.py
"""
多线程/多环境高吞吐训练配置预设
"""

from .config import PPOConfig


def get_multithread_config() -> PPOConfig:
    """针对多核CPU与单GPU的高吞吐配置。

    默认参数适合 32~48 线程CPU、单张中高端GPU（>=12GB）。
    可按需调整 num_envs、num_threads、rollout_steps 与 mini_batch_size。
    """
    return PPOConfig(
        # 并行
        num_envs=8,  # 并行环境数（4~16按机器调）
        num_threads=32,  # PyTorch CPU线程（24~48按机器调）
        # 采样与优化
        rollout_steps=4096,  # 更大的rollout提升吞吐与稳定
        mini_batch_size=1024,  # 提高batch尺寸以提升GPU利用率
        num_epochs=4,
        # 学习率与稳定性
        learning_rate=1e-4,
        clip_range=0.1,
        entropy_coef=0.005,
        value_loss_coef=0.5,
        # 网络规模（可按显存上调hidden_dim）
        hidden_dim=512,
        num_hidden_layers=3,
        use_transformer=False,
        # 日志/评估频率
        log_interval=2,
        save_interval=20,
        eval_interval=10,
        # 其它
        device="cuda",
        seed=42,
    )
