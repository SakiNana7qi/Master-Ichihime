# mahjong_agent/rollout_buffer.py
"""
Rollout缓冲区 - 用于存储轨迹数据并计算GAE优势
"""

import torch
import numpy as np
from typing import Dict, Generator, Optional, List
from .config import PPOConfig


class RolloutBuffer:
    """
    存储PPO训练过程中收集的轨迹数据
    支持GAE（Generalized Advantage Estimation）计算
    """

    def __init__(self, config: PPOConfig, device: torch.device):
        """
        初始化Rollout缓冲区

        Args:
            config: PPO配置
            device: 计算设备
        """
        self.config = config
        # 训练设备（用于取batch时搬到GPU）；存储一律放在CPU，减少频繁的小拷贝
        self.device = device
        self.store_device = torch.device("cpu")
        # 支持并行环境：按 env 数量扩展缓冲容量
        self.buffer_size = config.rollout_steps * max(1, getattr(config, "num_envs", 1))
        self.pos = 0
        self.full = False

        # 存储观测的各个部分
        self.observations = {
            "hand": torch.zeros(
                (self.buffer_size, 34), dtype=torch.float32, device=self.store_device
            ),
            "drawn_tile": torch.zeros(
                (self.buffer_size, 34), dtype=torch.float32, device=self.store_device
            ),
            "rivers": torch.zeros(
                (self.buffer_size, 4, 34), dtype=torch.float32, device=self.store_device
            ),
            "melds": torch.zeros(
                (self.buffer_size, 4, 34), dtype=torch.float32, device=self.store_device
            ),
            "riichi_status": torch.zeros(
                (self.buffer_size, 4), dtype=torch.float32, device=self.store_device
            ),
            "scores": torch.zeros(
                (self.buffer_size, 4), dtype=torch.float32, device=self.store_device
            ),
            "dora_indicators": torch.zeros(
                (self.buffer_size, 5, 34), dtype=torch.float32, device=self.store_device
            ),
            "game_info": torch.zeros(
                (self.buffer_size, 5), dtype=torch.float32, device=self.store_device
            ),
            "phase_info": torch.zeros(
                (self.buffer_size, 3), dtype=torch.float32, device=self.store_device
            ),
        }

        # 动作掩码
        self.action_masks = torch.zeros(
            (self.buffer_size, config.action_dim), dtype=torch.float32, device=self.store_device
        )

        # 动作、奖励、值等
        self.actions = torch.zeros((self.buffer_size,), dtype=torch.long, device=self.store_device)
        self.log_probs = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.store_device
        )
        self.rewards = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.store_device
        )
        self.values = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.store_device
        )
        self.dones = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.store_device
        )

        # GAE计算结果
        self.advantages = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.store_device
        )
        self.returns = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.store_device
        )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: np.ndarray,
    ):
        """
        添加一步经验到缓冲区

        Args:
            obs: 观测字典
            action: 采取的动作
            log_prob: 动作的对数概率
            reward: 获得的奖励
            value: Critic估计的价值
            done: 是否终止
            action_mask: 动作掩码
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError(
                "缓冲区已满！请先调用reset()或compute_returns_and_advantages()"
            )

        # 存储观测
        for key, value in obs.items():
            if key != "action_mask":  # action_mask单独存储
                self.observations[key][self.pos] = torch.from_numpy(value).to(self.store_device).float()

        # 存储其他数据（确保类型正确）
        self.action_masks[self.pos] = torch.from_numpy(action_mask).to(self.store_device).float()
        self.actions[self.pos] = int(action)
        self.log_probs[self.pos] = float(log_prob)
        self.rewards[self.pos] = float(reward)

        # 防御性处理value - 确保是标量
        if isinstance(value, np.ndarray):
            if value.size > 1:
                value = value.flatten()[0]
            value = float(value.item()) if hasattr(value, "item") else float(value)
        else:
            value = float(value)
        self.values[self.pos] = value

        self.dones[self.pos] = float(done)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def add_batch(
        self,
        obs_batch: Dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        action_masks: np.ndarray,
    ):
        """批量添加一整步来自并行环境的数据，形状第一维为 num_envs。"""
        num_envs = actions.shape[0]
        if self.pos + num_envs > self.buffer_size:
            raise RuntimeError("缓冲区已满！请先调用reset()或compute_returns_and_advantages()")

        sl = slice(self.pos, self.pos + num_envs)
        for key, arr in obs_batch.items():
            if key == "action_mask":
                continue
            self.observations[key][sl] = torch.from_numpy(arr).to(self.store_device).float()
        self.action_masks[sl] = torch.from_numpy(action_masks).to(self.store_device).float()
        self.actions[sl] = torch.from_numpy(actions).to(self.store_device).long()
        self.log_probs[sl] = torch.from_numpy(log_probs).to(self.store_device).float()
        self.rewards[sl] = torch.from_numpy(rewards).to(self.store_device).float()
        self.values[sl] = torch.from_numpy(values).to(self.store_device).float()
        self.dones[sl] = torch.from_numpy(dones).to(self.store_device).float()

        self.pos += num_envs
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: float, last_done: bool = False
    ) -> None:
        """
        使用GAE算法计算优势和回报

        注意：缓冲区按 [env0..envN-1, env0..envN-1, ...] 追加，
        因此在并行环境下需要按步长 num_envs 回溯，避免跨环境串联。

        Args:
            last_value: 为兼容旧接口保留（并行下我们不在rollout尾部bootstrap）
            last_done: 为兼容旧接口保留
        """
        num_envs = max(1, getattr(self.config, "num_envs", 1))
        T = self.buffer_size // num_envs  # 每个环境的时间长度

        advantages = torch.zeros_like(self.rewards)

        # 逐环境回溯计算GAE
        for env_id in range(num_envs):
            last_gae_lam = 0.0
            for t in reversed(range(T)):
                idx = env_id + t * num_envs
                if t == T - 1:
                    # rollout 末尾不做 bootstrap，作为终止处理
                    next_non_terminal = 0.0
                    next_value = 0.0
                else:
                    next_idx = env_id + (t + 1) * num_envs
                    next_non_terminal = 1.0 - float(self.dones[next_idx].item())
                    next_value = float(self.values[next_idx].item())

                delta = (
                    float(self.rewards[idx].item())
                    + self.config.gamma * next_value * next_non_terminal
                    - float(self.values[idx].item())
                )
                last_gae_lam = (
                    delta
                    + self.config.gamma
                    * self.config.gae_lambda
                    * next_non_terminal
                    * last_gae_lam
                )
                advantages[idx] = last_gae_lam

        # 回报 = 优势 + 价值
        self.advantages = advantages
        self.returns = advantages + self.values

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        以小批次返回缓冲区数据（用于训练）

        Args:
            batch_size: 小批次大小，如果为None则使用config中的值

        Yields:
            包含一个小批次数据的字典
        """
        if not self.full:
            raise RuntimeError("缓冲区未满！请先收集足够的数据")

        batch_size = batch_size or self.config.mini_batch_size
        indices = torch.randperm(self.buffer_size, device=self.store_device)

        # 分批返回
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]

            # 构建批次数据
            batch = {
                "observations": {
                    key: self.observations[key][batch_indices].to(self.device)
                    for key in self.observations
                },
                "action_masks": self.action_masks[batch_indices].to(self.device),
                "actions": self.actions[batch_indices].to(self.device),
                "old_log_probs": self.log_probs[batch_indices].to(self.device),
                "advantages": self.advantages[batch_indices].to(self.device),
                "returns": self.returns[batch_indices].to(self.device),
                "old_values": self.values[batch_indices].to(self.device),
            }

            yield batch

    def reset(self):
        """重置缓冲区"""
        self.pos = 0
        self.full = False

    def size(self) -> int:
        """返回当前缓冲区中的数据量"""
        return self.pos if not self.full else self.buffer_size

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        获取所有数据（不分批）

        Returns:
            包含所有数据的字典
        """
        if not self.full:
            actual_size = self.pos
        else:
            actual_size = self.buffer_size

        return {
            "observations": {
                key: self.observations[key][:actual_size] for key in self.observations
            },
            "action_masks": self.action_masks[:actual_size],
            "actions": self.actions[:actual_size],
            "old_log_probs": self.log_probs[:actual_size],
            "advantages": self.advantages[:actual_size],
            "returns": self.returns[:actual_size],
            "old_values": self.values[:actual_size],
            "rewards": self.rewards[:actual_size],
        }


class MultiAgentRolloutBuffer:
    """
    多智能体Rollout缓冲区
    为每个智能体维护独立的缓冲区（如果不共享策略）
    """

    def __init__(self, config: PPOConfig, device: torch.device, num_agents: int = 4):
        """
        初始化多智能体缓冲区

        Args:
            config: PPO配置
            device: 计算设备
            num_agents: 智能体数量
        """
        self.config = config
        self.device = device
        self.num_agents = num_agents

        # 为每个智能体创建独立的缓冲区
        self.buffers = [RolloutBuffer(config, device) for _ in range(num_agents)]

    def add(
        self,
        agent_id: int,
        obs: Dict[str, np.ndarray],
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: np.ndarray,
    ):
        """为指定智能体添加经验"""
        self.buffers[agent_id].add(
            obs, action, log_prob, reward, value, done, action_mask
        )

    def compute_returns_and_advantages(
        self, last_values: List[float], last_dones: List[bool]
    ):
        """为所有智能体计算优势和回报"""
        for i, buffer in enumerate(self.buffers):
            buffer.compute_returns_and_advantages(last_values[i], last_dones[i])

    def get(self, agent_id: int, batch_size: Optional[int] = None):
        """获取指定智能体的数据"""
        return self.buffers[agent_id].get(batch_size)

    def get_all(self, batch_size: Optional[int] = None):
        """
        获取所有智能体的数据（合并）
        用于共享策略的情况
        """
        all_batches = []
        for buffer in self.buffers:
            for batch in buffer.get(batch_size):
                all_batches.append(batch)

        # 合并所有批次
        # 这里简化处理，实际上可以更优雅地合并
        return all_batches

    def reset(self):
        """重置所有缓冲区"""
        for buffer in self.buffers:
            buffer.reset()

    def size(self) -> int:
        """返回总数据量"""
        return sum(buffer.size() for buffer in self.buffers)
