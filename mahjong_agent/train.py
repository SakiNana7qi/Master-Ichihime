# mahjong_agent/train.py
"""
主训练脚本 - 实现完整的PPO训练循环
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mahjong_environment import MahjongEnv
from mahjong_agent.model import MahjongActorCritic
from mahjong_agent.rollout_buffer import RolloutBuffer
from mahjong_agent.ppo_updater import PPOUpdater
from mahjong_agent.config import PPOConfig, get_default_config


class MahjongTrainer:
    """
    麻将AI训练器
    实现完整的PPO训练流程
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        初始化训练器

        Args:
            config: PPO配置，如果为None则使用默认配置
            checkpoint_path: 检查点路径，用于恢复训练
        """
        self.config = config or get_default_config()

        # 设置设备
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"使用设备: {self.device}")

        # 设置随机种子
        self._set_seed(self.config.seed)

        # 创建环境
        self.env = MahjongEnv(
            render_mode="human" if self.config.render_training else None,
            seed=self.config.seed,
        )

        # 创建模型
        self.model = MahjongActorCritic(self.config).to(self.device)
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 创建PPO更新器
        self.ppo_updater = PPOUpdater(self.model, self.config)

        # 创建Rollout缓冲区
        self.rollout_buffer = RolloutBuffer(self.config, self.device)

        # 训练统计
        self.global_step = 0
        self.rollout_count = 0
        self.episode_count = 0

        # 创建保存目录
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        # 加载检查点
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print("训练器初始化完成！")

    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _numpy_obs_to_torch(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """将numpy观测转换为torch张量"""
        torch_obs = {}
        for key, value in obs.items():
            if key != "action_mask":  # action_mask在后面单独处理
                torch_obs[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
        return torch_obs

    def collect_rollouts(self) -> Dict[str, float]:
        """
        收集一轮经验数据

        Returns:
            统计信息字典
        """
        self.model.eval()  # 设为评估模式
        self.rollout_buffer.reset()

        # 统计信息
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        # 重置环境
        obs, info = self.env.reset(seed=self.config.seed + self.global_step)

        # 收集rollout_steps步的数据
        for step in range(self.config.rollout_steps):
            # 检查游戏是否结束
            if self.env.agent_selection is None:
                # 保存episode统计
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)

                # 重置环境
                obs, info = self.env.reset()
                current_episode_reward = 0
                current_episode_length = 0
                self.episode_count += 1

            # 获取当前玩家
            current_agent = self.env.agent_selection

            # 获取观测和动作掩码
            action_mask = obs["action_mask"]

            # 转换观测为torch格式
            torch_obs = self._numpy_obs_to_torch(obs)
            torch_action_mask = (
                torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            )

            # 选择动作
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(
                    torch_obs, action_mask=torch_action_mask
                )

            # 转换为Python标量（确保类型正确）
            action_np = int(action.cpu().item())
            log_prob_np = float(log_prob.cpu().item())

            # 确保value是标量 - 彻底修复
            if value.numel() > 1:
                value = value.flatten()[0]  # 取第一个元素
            elif value.dim() > 0:
                value = value.squeeze()
            value_np = float(value.cpu().item())

            # 执行动作
            self.env.step(action_np)

            # 获取奖励和终止状态
            reward = self.env.rewards.get(current_agent, 0.0)
            done = self.env.terminations.get(current_agent, False)

            # 存储到缓冲区
            self.rollout_buffer.add(
                obs=obs,
                action=action_np,
                log_prob=log_prob_np,
                reward=reward,
                value=value_np,
                done=done,
                action_mask=action_mask,
            )

            # 更新统计
            current_episode_reward += reward
            current_episode_length += 1
            self.global_step += 1

            # 获取下一个观测
            if self.env.agent_selection is not None:
                obs = self.env.observe(self.env.agent_selection)

            # 渲染（如果需要）
            if self.config.render_training and step % 50 == 0:
                self.env.render()

        # 计算最后一个状态的价值（用于bootstrap）
        if self.env.agent_selection is not None:
            obs = self.env.observe(self.env.agent_selection)
            torch_obs = self._numpy_obs_to_torch(obs)
            with torch.no_grad():
                last_value = self.model.get_value(torch_obs).cpu().item()
            last_done = self.env.terminations.get(self.env.agent_selection, False)
        else:
            last_value = 0.0
            last_done = True

        # 计算优势和回报
        self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

        # 返回统计信息
        stats = {
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

        return stats

    def train_step(self) -> Dict[str, float]:
        """
        执行一次训练步骤（收集数据 + 更新策略）

        Returns:
            训练统计信息
        """
        # 收集数据
        rollout_stats = self.collect_rollouts()

        # 更新策略
        self.model.train()  # 设为训练模式
        update_stats = self.ppo_updater.update(self.rollout_buffer)

        # 合并统计信息
        stats = {**rollout_stats, **update_stats}

        return stats

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 80)
        print(" " * 30 + "开始训练")
        print("=" * 80)
        print(f"总步数: {self.config.total_timesteps:,}")
        print(f"Rollout步数: {self.config.rollout_steps}")
        print(
            f"预计迭代次数: {self.config.total_timesteps // self.config.rollout_steps}"
        )
        print("=" * 80 + "\n")

        start_time = time.time()

        # 主训练循环
        while self.global_step < self.config.total_timesteps:
            self.rollout_count += 1

            # 执行一次训练步骤
            stats = self.train_step()

            # 记录日志
            if self.rollout_count % self.config.log_interval == 0:
                self._log_stats(stats, start_time)

            # 保存模型
            if self.rollout_count % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.rollout_count}.pt")
                print(f"[OK] Checkpoint saved: checkpoint_{self.rollout_count}.pt")

            # 评估
            if self.rollout_count % self.config.eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=5)
                self._log_eval_stats(eval_stats)

        # 训练结束
        print("\n" + "=" * 80)
        print(" " * 30 + "训练完成")
        print("=" * 80)
        print(f"总步数: {self.global_step:,}")
        print(f"总时间: {(time.time() - start_time) / 3600:.2f} 小时")
        print("=" * 80 + "\n")

        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        print("[OK] Final model saved")

        self.writer.close()

    def _log_stats(self, stats: Dict[str, float], start_time: float):
        """记录训练统计信息"""
        elapsed_time = time.time() - start_time
        fps = self.global_step / elapsed_time

        # 控制台输出
        if self.config.verbose:
            print(f"\n【Rollout {self.rollout_count}】 步数: {self.global_step:,}")
            print(f"  平均回报: {stats.get('mean_episode_reward', 0):.3f}")
            print(f"  平均长度: {stats.get('mean_episode_length', 0):.1f}")
            print(f"  策略损失: {stats.get('policy_loss', 0):.4f}")
            print(f"  价值损失: {stats.get('value_loss', 0):.4f}")
            print(f"  熵: {stats.get('entropy', 0):.4f}")
            print(f"  裁剪比例: {stats.get('clip_fraction', 0):.3f}")
            print(f"  近似KL: {stats.get('approx_kl', 0):.4f}")
            print(f"  学习率: {stats.get('learning_rate', 0):.6f}")
            print(f"  FPS: {fps:.1f}")

        # TensorBoard
        for key, value in stats.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)
        self.writer.add_scalar("train/fps", fps, self.global_step)

    def _log_eval_stats(self, stats: Dict[str, float]):
        """记录评估统计信息"""
        if self.config.verbose:
            print(f"\n【评估】")
            print(f"  平均回报: {stats.get('mean_reward', 0):.3f}")
            print(f"  平均长度: {stats.get('mean_length', 0):.1f}")
            print(f"  胜率: {stats.get('win_rate', 0):.2%}")

        for key, value in stats.items():
            self.writer.add_scalar(f"eval/{key}", value, self.global_step)

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估当前策略

        Args:
            num_episodes: 评估的局数

        Returns:
            评估统计信息
        """
        self.model.eval()

        episode_rewards = []
        episode_lengths = []
        wins = 0

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                if self.env.agent_selection is None:
                    break

                current_agent = self.env.agent_selection
                action_mask = obs["action_mask"]

                # 使用确定性策略
                torch_obs = self._numpy_obs_to_torch(obs)
                torch_action_mask = (
                    torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    action, _, _, _ = self.model.get_action_and_value(
                        torch_obs, action_mask=torch_action_mask, deterministic=True
                    )

                self.env.step(action.cpu().item())

                reward = self.env.rewards.get(current_agent, 0.0)
                done = self.env.terminations.get(current_agent, False)

                episode_reward += reward
                episode_length += 1

                if self.env.agent_selection is not None:
                    obs = self.env.observe(self.env.agent_selection)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 检查是否获胜（player_0的奖励最高）
            if self.env.rewards.get("player_0", 0) > 0:
                wins += 1

        return {
            "mean_reward": np.mean(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "win_rate": wins / num_episodes,
        }

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = os.path.join(self.config.save_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.ppo_updater.optimizer.state_dict(),
                "global_step": self.global_step,
                "rollout_count": self.rollout_count,
                "episode_count": self.episode_count,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ppo_updater.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.rollout_count = checkpoint["rollout_count"]
        self.episode_count = checkpoint["episode_count"]
        print(f"已加载检查点: {path}")
        print(f"  全局步数: {self.global_step}")
        print(f"  Rollout次数: {self.rollout_count}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="训练麻将AI")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast", "high_performance"],
        help="配置类型",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="检查点路径（恢复训练）"
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 加载配置
    if args.config == "default":
        from mahjong_agent.config import get_default_config

        config = get_default_config()
    elif args.config == "fast":
        from mahjong_agent.config import get_fast_config

        config = get_fast_config()
    elif args.config == "high_performance":
        from mahjong_agent.config import get_high_performance_config

        config = get_high_performance_config()

    # 覆盖配置
    config.device = args.device
    config.seed = args.seed

    # 创建训练器并开始训练
    trainer = MahjongTrainer(config=config, checkpoint_path=args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
