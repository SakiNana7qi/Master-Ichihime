# mahjong_agent/config.py
"""
PPO训练超参数配置
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class PPOConfig:
    """PPO算法超参数配置"""

    # ========================
    # 环境相关参数
    # ========================
    num_agents: int = 4  # 智能体数量
    observation_dim: int = (
        34 + 34 + 4 * 34 + 4 * 34 + 4 + 4 + 5 * 34 + 5 + 3
    )  # 观测维度
    action_dim: int = 112  # 动作空间大小

    # ========================
    # 网络架构参数
    # ========================
    # Encoder
    hidden_dim: int = 768  # 隐藏层维度（增大模型）
    num_hidden_layers: int = 4  # MLP隐藏层数量（增大深度）
    activation: Literal["relu", "gelu", "tanh"] = "gelu"  # 激活函数
    use_layer_norm: bool = True  # 是否使用LayerNorm
    dropout: float = 0.0  # Dropout率（训练时）

    # Transformer (可选，高级特征提取)
    use_transformer: bool = True  # 是否使用Transformer（默认开启）
    num_transformer_layers: int = 4  # Transformer层数（增大）
    num_attention_heads: int = 12  # 注意力头数（与768维匹配）

    # ========================
    # PPO算法参数
    # ========================
    # 学习率
    learning_rate: float = 3e-4  # 初始学习率
    lr_schedule: Literal["constant", "linear", "cosine"] = "linear"  # 学习率调度
    # 学习率下限（避免收敛到0）。若同时设置，以 min_learning_rate 优先；否则按比例 * learning_rate。
    min_lr_ratio: float = 0.1  # 最低为初始学习率的该比例（例如0.1=10%）
    min_learning_rate: Optional[float] = None  # 绝对下限（None表示不使用绝对值）

    # PPO核心参数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE的lambda参数
    clip_range: float = 0.2  # PPO裁剪范围
    clip_range_vf: float = 0.2  # 价值函数裁剪范围（None表示不裁剪）

    # 损失权重
    value_loss_coef: float = 0.5  # 价值损失系数
    entropy_coef: float = 0.01  # 熵正则化系数
    max_grad_norm: float = 0.5  # 梯度裁剪阈值

    # 训练批次参数
    num_epochs: int = 4  # 每次更新的epoch数
    mini_batch_size: int = 256  # 小批次大小

    # ========================
    # 训练流程参数
    # ========================
    # Rollout参数
    rollout_steps: int = 8192  # 每次收集的步数（增大以提升GPU批处理）
    num_envs: int = 1  # 并行环境数（暂时设为1，后续可扩展）

    # 训练总步数
    total_timesteps: int = 1_000_000_000  # 总训练步数（扩大10倍）
    log_interval: int = 10  # 日志记录间隔（多少次rollout）
    save_interval: int = 50  # 模型保存间隔（多少次rollout）
    eval_interval: int = 20  # 评估间隔（多少次rollout）

    # ========================
    # 多智能体相关
    # ========================
    shared_policy: bool = True  # 是否共享策略（MAPPO中推荐）
    centralized_critic: bool = False  # 是否使用集中式Critic（暂不实现）

    # ========================
    # 其他参数
    # ========================
    seed: int = 42  # 随机种子
    device: str = "cuda"  # 设备 ("cuda" 或 "cpu")
    num_threads: int = 4  # PyTorch线程数
    pin_cpu_affinity: bool = False  # 是否为子进程设置CPU亲和度
    cpu_core_limit: int | None = None  # 限制可用CPU核心数（None表示不限制）
    cores_per_proc: Optional[int] = None  # 每个子进程绑定的核心数（None则自动计算）
    # 运行期开关
    use_shared_memory: bool = False  # 是否为并行环境启用共享内存通道

    # 路径
    save_dir: str = "./checkpoints"  # 模型保存路径
    log_dir: str = "./logs"  # 日志保存路径

    # 调试
    verbose: bool = True  # 是否打印详细信息
    render_training: bool = False  # 训练时是否渲染（会很慢）
    profile_timing: bool = False  # 是否输出阶段耗时剖析

    def __post_init__(self):
        """配置验证"""
        assert self.gamma > 0 and self.gamma <= 1, "gamma必须在(0, 1]之间"
        assert (
            self.gae_lambda >= 0 and self.gae_lambda <= 1
        ), "gae_lambda必须在[0, 1]之间"
        assert self.clip_range > 0, "clip_range必须大于0"
        assert self.learning_rate > 0, "learning_rate必须大于0"
        assert self.rollout_steps > 0, "rollout_steps必须大于0"
        assert self.mini_batch_size > 0, "mini_batch_size必须大于0"
        # 学习率下限校验
        if self.min_learning_rate is not None:
            assert 0 < self.min_learning_rate <= self.learning_rate, "min_learning_rate 必须在 (0, learning_rate]"
        if self.min_lr_ratio is not None:
            assert 0.0 < self.min_lr_ratio <= 1.0, "min_lr_ratio 必须在 (0, 1]"
        # 在并行环境下，应确保 (rollout_steps * num_envs) 能被 mini_batch_size 整除
        effective_envs = max(1, getattr(self, "num_envs", 1))
        assert (
            (self.rollout_steps * effective_envs) % self.mini_batch_size == 0
        ), "(rollout_steps * num_envs) 应能被 mini_batch_size 整除"


# 预设配置


def get_default_config() -> PPOConfig:
    """获取默认配置（适合初期训练）"""
    return PPOConfig()


def get_fast_config() -> PPOConfig:
    """获取快速训练配置（用于调试和快速实验）"""
    return PPOConfig(
        rollout_steps=512,
        mini_batch_size=128,
        total_timesteps=1_000_000,
        hidden_dim=256,
        num_hidden_layers=2,
        log_interval=5,
        save_interval=20,
    )


def get_high_performance_config() -> PPOConfig:
    """获取高性能配置（用于长时间训练）"""
    return PPOConfig(
        rollout_steps=16384,
        mini_batch_size=4096,
        total_timesteps=500_000_000,
        hidden_dim=1024,
        num_hidden_layers=4,
        use_transformer=True,
        num_transformer_layers=3,
        learning_rate=2e-4,
        num_epochs=8,
    )
