# mahjong_agent/ppo_updater.py
"""
PPO算法更新器
实现PPO的核心更新逻辑，包括策略损失、价值损失和熵正则化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
import numpy as np

from .config import PPOConfig
from .model import MahjongActorCritic
from .rollout_buffer import RolloutBuffer


class PPOUpdater:
    """
    PPO算法更新器
    负责执行PPO的策略更新
    """

    def __init__(
        self,
        model: MahjongActorCritic,
        config: PPOConfig,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        """
        初始化PPO更新器

        Args:
            model: Actor-Critic模型
            config: PPO配置
            optimizer: 优化器（如果为None则创建默认的Adam优化器）
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # 优化器
        if optimizer is None:
            try:
                # 尝试使用 fused Adam（PyTorch自带/torch.optim中在CUDA可用）
                self.optimizer = optim.Adam(
                    model.parameters(), lr=config.learning_rate, eps=1e-5, fused=True
                )
                print("已启用 fused Adam 优化器")
            except TypeError:
                self.optimizer = optim.Adam(
                    model.parameters(), lr=config.learning_rate, eps=1e-5
                )
        else:
            self.optimizer = optimizer

        # 学习率调度器
        self.lr_scheduler = self._create_lr_scheduler()

        # 统计信息
        self.update_count = 0

    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        if self.config.lr_schedule == "constant":
            return None
        
        # 计算整个训练期间的优化步数（以小批次为步）
        # buffer_size = rollout_steps * num_envs（并行环境会放大每次update消耗的样本数）
        effective_num_envs = max(1, getattr(self.config, "num_envs", 1))
        buffer_size_per_update = self.config.rollout_steps * effective_num_envs
        # 每个epoch内的batch数
        num_batches_per_epoch = max(1, buffer_size_per_update // self.config.mini_batch_size)
        # 总的优化步数（按batch计）
        total_opt_steps = max(
            1,
            (self.config.total_timesteps // buffer_size_per_update)
            * self.config.num_epochs
            * num_batches_per_epoch,
        )

        # 计算学习率下限
        min_lr = None
        if getattr(self.config, "min_learning_rate", None) is not None:
            min_lr = float(self.config.min_learning_rate)
        else:
            ratio = float(getattr(self.config, "min_lr_ratio", 0.0) or 0.0)
            if ratio > 0.0:
                min_lr = float(self.config.learning_rate) * ratio
        if min_lr is None or min_lr <= 0.0:
            min_lr = 1e-8  # 保底

        base_lr = float(self.config.learning_rate)

        if self.config.lr_schedule == "linear":
            # 线性从1降到 min_lr/base_lr，并钳制为非负
            def lr_lambda(step: int) -> float:
                frac = max(0.0, 1.0 - step / total_opt_steps)
                lr = base_lr * frac
                return max(min_lr / base_lr, lr / base_lr)
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        elif self.config.lr_schedule == "cosine":
            # 余弦退火到底为 min_lr
            cos = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_opt_steps, eta_min=min_lr)
            return cos
        else:
            raise ValueError(f"未知的学习率调度: {self.config.lr_schedule}")

    def update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        """
        使用收集的数据更新策略

        Args:
            rollout_buffer: 包含轨迹数据的缓冲区

        Returns:
            包含训练统计信息的字典
        """
        # 统计信息
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        approx_kls = []

        # 进行多个epoch的训练
        for epoch in range(self.config.num_epochs):
            # 从缓冲区获取小批次数据
            for batch in rollout_buffer.get(self.config.mini_batch_size):
                # 提取批次数据
                obs = batch["observations"]
                action_masks = batch["action_masks"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                # 优势标准化（提高训练稳定性）
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # 前向传播：启用 AMP 以提升吞吐（bfloat16/TF32）
                if obs["hand"].is_cuda:
                    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                else:
                    class _Null:
                        def __enter__(self):
                            return None
                        def __exit__(self, *args):
                            return False
                    amp_ctx = _Null()
                with amp_ctx:
                    # 前向与损失一起做大批量，infer吞吐更好
                    _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                        obs, action_mask=action_masks, action=actions
                    )
                    # 训练时 loss 以 float32 计算更稳定
                    new_log_probs = new_log_probs.float()
                    entropy = entropy.float()
                    new_values = new_values.float()

                # ==================== 策略损失 ====================
                # 计算重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs)

                # PPO的裁剪目标
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ==================== 价值损失 ====================
                if self.config.clip_range_vf is not None:
                    # 价值函数裁剪（可选）
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -self.config.clip_range_vf,
                        self.config.clip_range_vf,
                    )
                    value_loss1 = (new_values - returns) ** 2
                    value_loss2 = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    # 标准MSE损失
                    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # ==================== 熵损失 ====================
                entropy_loss = entropy.mean()

                # ==================== 总损失 ====================
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()

                # 更新学习率
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 统计信息
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

                # 计算裁剪比例（用于监控训练进度）
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.config.clip_range)
                        .float()
                        .mean()
                        .item()
                    )
                    clip_fractions.append(clip_fraction)

                    # 近似KL散度
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    approx_kls.append(approx_kl)

        self.update_count += 1

        # 返回统计信息
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kls),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def save(self, path: str):
        """保存模型和优化器状态"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        """加载模型和优化器状态"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        print(f"已加载模型，更新次数: {self.update_count}")


class MultiAgentPPOUpdater:
    """
    多智能体PPO更新器
    支持多个独立的策略（非共享参数情况）
    """

    def __init__(
        self,
        models: List[MahjongActorCritic],
        config: PPOConfig,
        optimizers: Optional[List[optim.Optimizer]] = None,
    ):
        """
        初始化多智能体PPO更新器

        Args:
            models: Actor-Critic模型列表
            config: PPO配置
            optimizers: 优化器列表
        """
        self.config = config
        self.num_agents = len(models)

        # 为每个智能体创建独立的更新器
        if optimizers is None:
            optimizers = [None] * self.num_agents

        self.updaters = [
            PPOUpdater(model, config, optimizer)
            for model, optimizer in zip(models, optimizers)
        ]

    def update(self, rollout_buffers: List[RolloutBuffer]) -> List[Dict[str, float]]:
        """
        更新所有智能体的策略

        Args:
            rollout_buffers: 每个智能体的缓冲区列表

        Returns:
            每个智能体的训练统计信息列表
        """
        stats = []
        for updater, buffer in zip(self.updaters, rollout_buffers):
            agent_stats = updater.update(buffer)
            stats.append(agent_stats)
        return stats

    def save(self, path_template: str):
        """
        保存所有智能体的模型

        Args:
            path_template: 路径模板，应包含 {agent_id} 占位符
        """
        for i, updater in enumerate(self.updaters):
            path = path_template.format(agent_id=i)
            updater.save(path)

    def load(self, path_template: str):
        """
        加载所有智能体的模型

        Args:
            path_template: 路径模板，应包含 {agent_id} 占位符
        """
        for i, updater in enumerate(self.updaters):
            path = path_template.format(agent_id=i)
            updater.load(path)
