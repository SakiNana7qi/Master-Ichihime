# mahjong_agent/evaluate.py
"""
评估脚本 - 评估训练好的麻将AI
支持与随机策略、固定策略对战，并提供详细的性能分析
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mahjong_environment import MahjongEnv
from mahjong_agent.model import MahjongActorCritic
from mahjong_agent.config import PPOConfig


class MahjongEvaluator:
    """麻将AI评估器"""

    def __init__(
        self, model_path: str, config: Optional[PPOConfig] = None, device: str = "cuda"
    ):
        """
        初始化评估器

        Args:
            model_path: 模型权重路径
            config: PPO配置
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)

        if config is None:
            # 从检查点加载配置
            config = checkpoint.get("config", PPOConfig())

        self.config = config
        self.model = MahjongActorCritic(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"已加载模型: {model_path}")
        print(f"训练步数: {checkpoint.get('global_step', 'N/A')}")

        # 创建环境
        self.env = MahjongEnv(render_mode=None)

    def _numpy_obs_to_torch(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """将numpy观测转换为torch张量"""
        torch_obs = {}
        for key, value in obs.items():
            if key != "action_mask":
                torch_obs[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
        return torch_obs

    def evaluate(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            num_episodes: 评估局数
            deterministic: 是否使用确定性策略
            verbose: 是否显示进度

        Returns:
            评估统计信息
        """
        # 统计信息
        episode_rewards = {f"player_{i}": [] for i in range(4)}
        episode_lengths = []
        final_scores = {f"player_{i}": [] for i in range(4)}

        wins = {f"player_{i}": 0 for i in range(4)}
        win_types = {"ron": 0, "tsumo": 0, "draw": 0}

        iterator = (
            tqdm(range(num_episodes), desc="评估中") if verbose else range(num_episodes)
        )

        for episode in iterator:
            obs, info = self.env.reset(seed=episode)
            episode_length = 0
            episode_reward_acc = {f"player_{i}": 0.0 for i in range(4)}

            while True:
                if self.env.agent_selection is None:
                    break

                current_agent = self.env.agent_selection
                action_mask = obs["action_mask"]

                # 获取动作
                torch_obs = self._numpy_obs_to_torch(obs)
                torch_action_mask = (
                    torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    action, _, _, _ = self.model.get_action_and_value(
                        torch_obs,
                        action_mask=torch_action_mask,
                        deterministic=deterministic,
                    )

                # 执行动作
                self.env.step(action.cpu().item())

                # 累计奖励
                for agent in self.env.possible_agents:
                    episode_reward_acc[agent] += self.env.rewards.get(agent, 0.0)

                episode_length += 1

                # 获取下一个观测
                if self.env.agent_selection is not None:
                    obs = self.env.observe(self.env.agent_selection)

            # 记录统计信息
            episode_lengths.append(episode_length)

            for agent in self.env.possible_agents:
                episode_rewards[agent].append(episode_reward_acc[agent])
                player_idx = int(agent.split("_")[1])
                final_scores[agent].append(
                    self.env.game_state.players[player_idx].score
                )

            # 记录胜者
            if self.env.game_state.round_result:
                result = self.env.game_state.round_result
                if result.winner is not None:
                    winner_agent = f"player_{result.winner}"
                    wins[winner_agent] += 1

                    # 记录胜利类型
                    if result.result_type == "ron":
                        win_types["ron"] += 1
                    elif result.result_type == "tsumo":
                        win_types["tsumo"] += 1
                else:
                    win_types["draw"] += 1

        # 计算统计结果
        results = {
            "num_episodes": num_episodes,
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
        }

        # 每个玩家的统计
        for agent in self.env.possible_agents:
            prefix = f"{agent}_"
            results[prefix + "mean_reward"] = np.mean(episode_rewards[agent])
            results[prefix + "std_reward"] = np.std(episode_rewards[agent])
            results[prefix + "mean_score"] = np.mean(final_scores[agent])
            results[prefix + "std_score"] = np.std(final_scores[agent])
            results[prefix + "win_rate"] = wins[agent] / num_episodes

        # 胜利类型统计
        results["ron_rate"] = win_types["ron"] / num_episodes
        results["tsumo_rate"] = win_types["tsumo"] / num_episodes
        results["draw_rate"] = win_types["draw"] / num_episodes

        # player_0（主要评估对象）的性能
        results["player_0_performance"] = (
            results["player_0_mean_reward"] + results["player_0_win_rate"] * 10
        )

        return results

    def play_interactive(self, render: bool = True):
        """
        交互式游戏（人类可以观察AI如何决策）

        Args:
            render: 是否渲染游戏状态
        """
        print("\n" + "=" * 80)
        print(" " * 30 + "交互式演示")
        print("=" * 80 + "\n")

        obs, info = self.env.reset()
        step = 0

        while True:
            if self.env.agent_selection is None:
                break

            current_agent = self.env.agent_selection
            action_mask = obs["action_mask"]

            # 获取AI动作
            torch_obs = self._numpy_obs_to_torch(obs)
            torch_action_mask = (
                torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(
                    torch_obs, action_mask=torch_action_mask, deterministic=False
                )

            # 解码动作
            from mahjong_environment.utils.action_encoder import ActionEncoder

            action_type, params = ActionEncoder.decode_action(action.cpu().item())

            # 显示决策信息
            print(f"\n【步骤 {step}】 当前玩家: {current_agent}")
            print(f"  动作: {action_type} {params}")
            print(f"  价值估计: {value.cpu().item():.3f}")
            print(f"  动作概率: {torch.exp(log_prob).cpu().item():.3f}")

            # 执行动作
            self.env.step(action.cpu().item())
            step += 1

            # 渲染
            if render and step % 10 == 0:
                self.env.render()

            # 获取下一个观测
            if self.env.agent_selection is not None:
                obs = self.env.observe(self.env.agent_selection)

        # 显示最终结果
        print("\n" + "=" * 80)
        print("游戏结束！")
        print("=" * 80)
        self.env.render()

        print("\n最终分数:")
        for i, agent in enumerate(self.env.possible_agents):
            player = self.env.game_state.players[i]
            reward = self.env.rewards[agent]
            print(f"  {agent}: {player.score} 点 (奖励: {reward:+.2f})")

    def benchmark_vs_random(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        与随机策略对战的基准测试

        Args:
            num_episodes: 测试局数

        Returns:
            基准测试结果
        """
        print(f"\n对战随机策略 ({num_episodes}局)...")
        results = self.evaluate(num_episodes=num_episodes, verbose=True)

        print("\n基准测试结果:")
        print(f"  player_0 平均奖励: {results['player_0_mean_reward']:.3f}")
        print(f"  player_0 胜率: {results['player_0_win_rate']:.2%}")
        print(f"  player_0 平均分数: {results['player_0_mean_score']:.1f}")
        print(f"  荣和率: {results['ron_rate']:.2%}")
        print(f"  自摸率: {results['tsumo_rate']:.2%}")
        print(f"  流局率: {results['draw_rate']:.2%}")

        return results

    def save_evaluation_report(self, results: Dict[str, float], output_path: str):
        """
        保存评估报告

        Args:
            results: 评估结果
            output_path: 输出路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 30 + "麻将AI评估报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"评估局数: {results['num_episodes']}\n")
            f.write(
                f"平均局长: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}\n\n"
            )

            f.write("=" * 80 + "\n")
            f.write("各玩家性能:\n")
            f.write("=" * 80 + "\n\n")

            for i in range(4):
                agent = f"player_{i}"
                f.write(f"{agent}:\n")
                f.write(
                    f"  平均奖励: {results[f'{agent}_mean_reward']:.3f} ± {results[f'{agent}_std_reward']:.3f}\n"
                )
                f.write(
                    f"  平均分数: {results[f'{agent}_mean_score']:.1f} ± {results[f'{agent}_std_score']:.1f}\n"
                )
                f.write(f"  胜率: {results[f'{agent}_win_rate']:.2%}\n\n")

            f.write("=" * 80 + "\n")
            f.write("胜利类型统计:\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"荣和率: {results['ron_rate']:.2%}\n")
            f.write(f"自摸率: {results['tsumo_rate']:.2%}\n")
            f.write(f"流局率: {results['draw_rate']:.2%}\n")

        print(f"\n评估报告已保存至: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估麻将AI")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--interactive", action="store_true", help="交互式演示")
    parser.add_argument(
        "--output", type=str, default="evaluation_report.txt", help="报告输出路径"
    )

    args = parser.parse_args()

    # 创建评估器
    evaluator = MahjongEvaluator(model_path=args.model, device=args.device)

    if args.interactive:
        # 交互式演示
        evaluator.play_interactive(render=True)
    else:
        # 标准评估
        results = evaluator.benchmark_vs_random(num_episodes=args.episodes)

        # 保存报告
        evaluator.save_evaluation_report(results, args.output)


if __name__ == "__main__":
    main()
