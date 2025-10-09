# mahjong_agent/config_multithread.py
"""
多线程/多环境高吞吐训练配置预设
"""

from .config import PPOConfig


def get_multithread_config() -> PPOConfig:
    """针对多核CPU与单GPU的高吞吐配置。

    默认参数适合 32~72 线程CPU、单张中高端GPU（>=12GB）。
    对于 72 核 CPU，使用 32 个并行环境以充分利用所有核心。
    可按需调整 num_envs、num_threads、rollout_steps 与 mini_batch_size。
    """
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
    
    # 根据 CPU 核心数自动调整环境数
    # 每 2-3 个核心一个环境
    if cpu_count >= 64:
        num_envs = 32  # 64-72 核
    elif cpu_count >= 48:
        num_envs = 24  # 48-63 核
    elif cpu_count >= 32:
        num_envs = 16  # 32-47 核
    elif cpu_count >= 16:
        num_envs = 8   # 16-31 核
    else:
        num_envs = 4   # <16 核
    
    return PPOConfig(
        # 并行
        num_envs=num_envs,  # 根据CPU自动调整
        num_threads=cpu_count,  # 使用所有CPU核心
        cpu_core_limit=None,  # 外部可覆盖为固定核心数（如36）
        # 可在外部设置：config.pin_cpu_affinity=True 启用Windows/Linux下子进程CPU亲和度分散
        # 采样与优化
        rollout_steps=4096,  # 更大的rollout提升吞吐与稳定
        mini_batch_size=2048,  # 更大的batch以提升GPU利用率（32环境时）
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
