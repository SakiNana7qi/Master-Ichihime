# mahjong_agent/model.py
"""
Actor-Critic神经网络模型
包含特征提取器(Encoder)、策略头(Actor)和价值头(Critic)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional
import numpy as np

from .config import PPOConfig


class MahjongObservationEncoder(nn.Module):
    """
    麻将观测编码器
    将字典形式的观测转换为特征向量
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        # 各个部分的维度
        self.hand_dim = 34  # 手牌
        self.drawn_tile_dim = 34  # 摸牌
        self.rivers_dim = 4 * 34  # 牌河（4个玩家）
        self.melds_dim = 4 * 34  # 副露
        self.riichi_dim = 4  # 立直状态
        self.scores_dim = 4  # 分数
        self.dora_dim = 5 * 34  # 宝牌指示牌
        self.game_info_dim = 5  # 场况信息
        self.phase_info_dim = 3  # 阶段信息

        # 总输入维度
        self.total_dim = (
            self.hand_dim
            + self.drawn_tile_dim
            + self.rivers_dim
            + self.melds_dim
            + self.riichi_dim
            + self.scores_dim
            + self.dora_dim
            + self.game_info_dim
            + self.phase_info_dim
        )

        # 特征提取层
        # 我们为不同类型的输入设计不同的编码器
        self.hand_encoder = nn.Sequential(
            nn.Linear(self.hand_dim, 128), nn.GELU(), nn.Linear(128, 128)
        )

        self.drawn_tile_encoder = nn.Sequential(
            nn.Linear(self.drawn_tile_dim, 64), nn.GELU(), nn.Linear(64, 64)
        )

        self.rivers_encoder = nn.Sequential(
            nn.Linear(self.rivers_dim, 256), nn.GELU(), nn.Linear(256, 256)
        )

        self.melds_encoder = nn.Sequential(
            nn.Linear(self.melds_dim, 256), nn.GELU(), nn.Linear(256, 256)
        )

        self.game_info_encoder = nn.Sequential(
            nn.Linear(
                self.riichi_dim
                + self.scores_dim
                + self.game_info_dim
                + self.phase_info_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        self.dora_encoder = nn.Sequential(
            nn.Linear(self.dora_dim, 128), nn.GELU(), nn.Linear(128, 128)
        )

        # 合并后的特征维度
        self.combined_dim = 128 + 64 + 256 + 256 + 128 + 128  # = 960

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将观测字典编码为特征向量

        Args:
            obs: 观测字典，包含hand, drawn_tile, rivers等

        Returns:
            编码后的特征向量
        """
        # 编码各部分
        hand_feat = self.hand_encoder(obs["hand"].float())
        drawn_tile_feat = self.drawn_tile_encoder(obs["drawn_tile"].float())
        rivers_feat = self.rivers_encoder(obs["rivers"].flatten(start_dim=-2).float())
        melds_feat = self.melds_encoder(obs["melds"].flatten(start_dim=-2).float())
        dora_feat = self.dora_encoder(
            obs["dora_indicators"].flatten(start_dim=-2).float()
        )

        # 合并游戏信息
        game_info = torch.cat(
            [
                obs["riichi_status"].float(),
                obs["scores"].float(),
                obs["game_info"].float(),
                obs["phase_info"].float(),
            ],
            dim=-1,
        )
        game_info_feat = self.game_info_encoder(game_info)

        # 拼接所有特征
        combined = torch.cat(
            [
                hand_feat,
                drawn_tile_feat,
                rivers_feat,
                melds_feat,
                game_info_feat,
                dora_feat,
            ],
            dim=-1,
        )

        return combined


class TransformerEncoder(nn.Module):
    """
    Transformer编码器（可选）
    用于捕捉更复杂的序列和关系特征
    """

    def __init__(self, config: PPOConfig, input_dim: int):
        super().__init__()
        self.config = config

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=input_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_transformer_layers
        )

        # Layer Norm
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) 或 (batch, dim)
        Returns:
            编码后的特征
        """
        if x.dim() == 2:
            # (batch, dim) -> (batch, 1, dim)
            x = x.unsqueeze(1)

        # 预归一化以稳定Transformer数值
        x = self.layer_norm(x)
        x = self.transformer(x)

        # 取序列的平均或最后一个
        x = x.mean(dim=1)  # (batch, dim)

        # 后归一化进一步稳定输出
        return self.layer_norm(x)


class MahjongActorCritic(nn.Module):
    """
    麻将AI的Actor-Critic网络
    包含一个共享的特征提取器和两个独立的头部
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        # 观测编码器
        self.obs_encoder = MahjongObservationEncoder(config)

        # 共享特征提取器（Encoder主干）
        layers = []
        input_dim = self.obs_encoder.combined_dim

        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(self._get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = config.hidden_dim

        self.shared_encoder = nn.Sequential(*layers)

        # Transformer（可选）
        if config.use_transformer:
            self.transformer = TransformerEncoder(config, config.hidden_dim)
        else:
            self.transformer = None

        # Actor头（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        # Critic头（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            self._get_activation(config.activation),
            nn.Linear(config.hidden_dim, 1),
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"未知的激活函数: {activation}")

    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观测字典
            deterministic: 是否使用确定性策略（评估时使用）

        Returns:
            action_logits: 动作的logits (batch, action_dim)
            value: 状态价值 (batch, 1)
        """
        # 编码观测
        features = self.obs_encoder(obs)

        # 共享特征提取
        features = self.shared_encoder(features)

        # Transformer（可选）
        if self.transformer is not None:
            features = self.transformer(features)

        # Actor和Critic
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、log概率、熵和价值

        Args:
            obs: 观测字典
            action_mask: 动作掩码 (batch, action_dim)，1表示合法，0表示非法
            action: 指定的动作（用于计算log_prob），如果为None则采样新动作
            deterministic: 是否使用确定性策略

        Returns:
            action: 采样的动作 (batch,)
            log_prob: 动作的对数概率 (batch,)
            entropy: 策略熵 (batch,)
            value: 状态价值 (batch,)
        """
        # 前向传播
        action_logits, value = self.forward(obs, deterministic)

        # 数值稳定性处理：将NaN/Inf替换为有限数，并裁剪范围
        action_logits = torch.nan_to_num(
            action_logits, nan=0.0, posinf=1e9, neginf=-1e9
        )
        action_logits = action_logits.clamp(min=-50.0, max=50.0)

        # 应用动作掩码
        if action_mask is not None:
            # 将非法动作的logits设置为极小值
            masked_logits = action_logits.masked_fill(action_mask == 0, -1e9)

            # 处理极端情况：如果没有任何合法动作，回退到选择索引0
            if action_mask.dim() == 2:
                invalid_rows = action_mask.sum(dim=-1) == 0
                if invalid_rows.any():
                    masked_logits[invalid_rows, 0] = 0.0
            else:
                if action_mask.sum() == 0:
                    masked_logits[..., 0] = 0.0

            action_logits = masked_logits

        # 创建分类分布
        dist = Categorical(logits=action_logits)

        # 采样或使用给定的动作
        if action is None:
            if deterministic:
                # 确定性策略：选择概率最大的动作
                action = action_logits.argmax(dim=-1)
            else:
                # 随机策略：从分布中采样
                action = dist.sample()

        # 计算log概率和熵
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # 保持批量维度：(batch,)
        value = value.squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        仅获取价值估计（用于计算优势）

        Args:
            obs: 观测字典

        Returns:
            value: 状态价值 (batch,)
        """
        _, value = self.forward(obs)
        # 返回形状：(batch,)
        return value.squeeze(-1)


def convert_numpy_obs_to_torch(
    obs: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    将numpy观测转换为torch张量

    Args:
        obs: numpy格式的观测字典
        device: 目标设备

    Returns:
        torch格式的观测字典
    """
    return {key: torch.from_numpy(value).to(device) for key, value in obs.items()}
