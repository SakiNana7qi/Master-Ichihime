# mahjong_environment/example_random_agent.py
"""
随机智能体示例
演示如何使用麻将环境进行完整的游戏模拟
"""

import sys
import os
import random
import numpy as np

# 添加父目录到路径，使得可以导入mahjong_environment包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正确导入
from mahjong_environment import MahjongEnv


def random_agent_demo():
    """随机智能体演示"""
    print("=" * 80)
    print(" " * 25 + "麻将环境 - 随机智能体演示")
    print("=" * 80 + "\n")

    # 创建环境
    env = MahjongEnv(render_mode="human", seed=42)
    print("[OK] 环境创建成功\n")

    # 重置环境
    obs, info = env.reset(seed=42)
    print("[OK] 环境重置成功")
    print(f"初始玩家: {env.agent_selection}\n")

    # 初始状态渲染
    env.render()

    # 游戏主循环
    step_count = 0
    max_steps = 200  # 防止无限循环

    print("\n" + "=" * 80)
    print("开始游戏...")
    print("=" * 80 + "\n")

    while step_count < max_steps:
        # 检查游戏是否结束
        if env.agent_selection is None or env.terminations[env.agent_selection]:
            break

        current_agent = env.agent_selection

        # 获取合法动作
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask == 1)[0]

        if len(legal_actions) == 0:
            print(f"警告: {current_agent} 没有合法动作！")
            break

        # 随机选择一个合法动作
        action = random.choice(legal_actions)

        # 解码动作（用于显示）
        try:
            from mahjong_environment.utils.action_encoder import ActionEncoder
        except ImportError:
            from utils.action_encoder import ActionEncoder

        action_type, params = ActionEncoder.decode_action(action)

        # 执行动作
        env.step(action)
        step_count += 1

        # 每20步显示一次详细信息
        if step_count % 20 == 0:
            print(f"\n【第 {step_count} 步】")
            print(f"玩家: {current_agent}")
            print(f"动作: {action_type} {params}")
            env.render()

        # 获取新观测
        if env.agent_selection is not None:
            obs = env.observe(env.agent_selection)

    # 游戏结束
    print("\n" + "=" * 80)
    print("游戏结束！")
    print("=" * 80 + "\n")

    # 显示最终结果
    env.render()

    print("\n" + "=" * 80)
    print("最终结果:")
    print("=" * 80)

    # 显示每个玩家的信息
    for i, agent in enumerate(env.possible_agents):
        player = env.game_state.players[i]
        reward = env.rewards[agent]

        status = "[+]" if reward > 0 else "[-]" if reward < 0 else "[=]"

        print(f"\n{status} {agent}:")
        print(f"   分数: {player.score}")
        print(f"   奖励: {reward:+.2f}")

        if player.is_riichi:
            print(f"   状态: 立直")

    # 显示游戏统计
    print(f"\n总步数: {step_count}")

    if env.game_state.round_result:
        result = env.game_state.round_result
        print(f"结果类型: {result.result_type}")

        if result.winner is not None:
            winner_agent = f"player_{result.winner}"
            print(f"和牌者: {winner_agent}")
            print(f"番数: {result.han} 番")
            print(f"符数: {result.fu} 符")
            print(f"得点: {result.points} 点")

            if result.loser is not None:
                loser_agent = f"player_{result.loser}"
                print(f"放铳者: {loser_agent}")

    print("\n" + "=" * 80 + "\n")


def interactive_demo():
    """交互式演示（简化版，主要展示API用法）"""
    print("=" * 80)
    print(" " * 20 + "麻将环境 - 交互式演示（API用法）")
    print("=" * 80 + "\n")

    env = MahjongEnv(render_mode="human", seed=None)

    print("演示环境API的基本用法:\n")

    # 1. 重置
    print("1. 重置环境")
    obs, info = env.reset()
    print(f"   返回: observation (dict), info (dict)")
    print(f"   当前玩家: {env.agent_selection}")
    print(f"   观测键: {list(obs.keys())}\n")

    # 2. 观测
    print("2. 获取观测")
    print(f"   手牌形状: {obs['hand'].shape}")
    print(f"   手牌总数: {np.sum(obs['hand'])}")
    print(f"   动作掩码形状: {obs['action_mask'].shape}")

    legal_count = np.sum(obs["action_mask"])
    print(f"   合法动作数: {legal_count}\n")

    # 3. 执行动作
    print("3. 执行动作")
    legal_actions = np.where(obs["action_mask"] == 1)[0]
    action = legal_actions[0]

    try:
        from mahjong_environment.utils.action_encoder import ActionEncoder
    except ImportError:
        from utils.action_encoder import ActionEncoder

    action_type, params = ActionEncoder.decode_action(action)

    print(f"   选择动作: {action}")
    print(f"   动作类型: {action_type}")
    print(f"   动作参数: {params}")

    env.step(action)
    print(f"   执行后当前玩家: {env.agent_selection}\n")

    # 4. 渲染
    print("4. 渲染游戏状态")
    env.render()

    # 5. 检查状态
    print("\n5. 检查游戏状态")
    print(f"   终止状态: {env.terminations}")
    print(f"   奖励: {env.rewards}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # 运行随机智能体演示
    random_agent_demo()

    # 如果想查看交互式API演示，取消下面的注释
    # interactive_demo()
