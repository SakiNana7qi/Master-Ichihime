# mahjong_agent/__init__.py
"""
麻将AI Agent模块 - 基于PPO算法的强化学习实现

主要组件:
- MahjongActorCritic: Actor-Critic神经网络模型
- RolloutBuffer: 经验数据缓冲区
- PPOUpdater: PPO算法更新器
- PPOConfig: 超参数配置
- MahjongTrainer: 训练器
- MahjongEvaluator: 评估器
"""

from .model import MahjongActorCritic, convert_numpy_obs_to_torch
from .rollout_buffer import RolloutBuffer, MultiAgentRolloutBuffer
from .ppo_updater import PPOUpdater, MultiAgentPPOUpdater
from .config import (
    PPOConfig,
    get_default_config,
    get_fast_config,
    get_high_performance_config,
)
from .train import MahjongTrainer
from .evaluate import MahjongEvaluator

__version__ = "1.0.0"

__all__ = [
    # 模型
    "MahjongActorCritic",
    "convert_numpy_obs_to_torch",
    # 缓冲区
    "RolloutBuffer",
    "MultiAgentRolloutBuffer",
    # 更新器
    "PPOUpdater",
    "MultiAgentPPOUpdater",
    # 配置
    "PPOConfig",
    "get_default_config",
    "get_fast_config",
    "get_high_performance_config",
    # 训练与评估
    "MahjongTrainer",
    "MahjongEvaluator",
]
