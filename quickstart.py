#!/usr/bin/env python3
# quickstart.py
"""
快速开始脚本
展示麻将环境的基本用法
"""

import sys
import random
import numpy as np

try:
    from mahjong_environment import MahjongEnv
except ImportError:
    print("错误：无法导入mahjong_environment模块")
    print("请确保您在正确的目录中运行此脚本")
    sys.exit(1)


def quickstart():
    """快速开始演示"""
    print("\n" + "=" * 80)
    print(" " * 20 + "🀄 雀魂立直麻将环境 - 快速开始 🀄")
    print("=" * 80 + "\n")

    print("欢迎使用立直麻将强化学习环境！\n")
    print("本脚本将演示环境的基本使用方法...\n")

    # 步骤1: 创建环境
    print("=" * 80)
    print("步骤 1/5: 创建环境")
    print("=" * 80)
    print("代码: env = MahjongEnv(render_mode='human', seed=42)")

    env = MahjongEnv(render_mode="human", seed=42)
    print("✓ 环境创建成功！")
    print(f"  - 玩家数量: {len(env.possible_agents)}")
    print(f"  - 动作空间大小: {env.action_spaces['player_0'].n}")
    input("\n按Enter继续...\n")

    # 步骤2: 重置环境
    print("=" * 80)
    print("步骤 2/5: 重置环境")
    print("=" * 80)
    print("代码: obs, info = env.reset()")

    obs, info = env.reset(seed=42)
    print("✓ 环境重置成功！")
    print(f"  - 当前玩家: {env.agent_selection}")
    print(f"  - 观测空间包含: {len(obs)} 个键")
    print(f"  - 手牌数量: {np.sum(obs['hand'])} 张")

    # 显示初始状态
    print("\n初始游戏状态:")
    env.render()
    input("\n按Enter继续...\n")

    # 步骤3: 查看合法动作
    print("=" * 80)
    print("步骤 3/5: 获取合法动作")
    print("=" * 80)
    print("代码: legal_actions = np.where(obs['action_mask'] == 1)[0]")

    legal_actions = np.where(obs["action_mask"] == 1)[0]
    print(f"✓ 当前有 {len(legal_actions)} 个合法动作")
    print(f"  - 前5个合法动作: {legal_actions[:5]}")

    # 解码几个动作看看
    from mahjong_environment.utils.action_encoder import ActionEncoder

    print("\n  动作解码示例:")
    for i in range(min(3, len(legal_actions))):
        action_id = legal_actions[i]
        action_type, params = ActionEncoder.decode_action(action_id)
        print(f"    动作{action_id}: {action_type} {params}")

    input("\n按Enter继续...\n")

    # 步骤4: 执行动作
    print("=" * 80)
    print("步骤 4/5: 执行动作")
    print("=" * 80)
    print("代码: env.step(action)")

    action = random.choice(legal_actions)
    action_type, params = ActionEncoder.decode_action(action)

    print(f"选择动作: {action} ({action_type} {params})")
    env.step(action)

    print("✓ 动作执行成功！")
    print(f"  - 新的当前玩家: {env.agent_selection}")

    env.render()
    input("\n按Enter继续...\n")

    # 步骤5: 运行完整游戏
    print("=" * 80)
    print("步骤 5/5: 运行完整游戏（前20步）")
    print("=" * 80)
    print("现在将模拟20步随机游戏...\n")

    step_count = 1  # 已经执行了1步
    max_steps = 20

    while step_count < max_steps and not env.terminations[env.agent_selection]:
        # 获取新观测
        obs = env.observe(env.agent_selection)

        # 选择动作
        legal_actions = np.where(obs["action_mask"] == 1)[0]
        if len(legal_actions) == 0:
            break

        action = random.choice(legal_actions)

        # 执行
        env.step(action)
        step_count += 1

        # 每5步显示一次
        if step_count % 5 == 0:
            print(f"\n--- 第 {step_count} 步 ---")
            action_type, params = ActionEncoder.decode_action(action)
            print(f"动作: {action_type} {params}")

    print(f"\n✓ 完成 {step_count} 步模拟")

    # 最终状态
    print("\n" + "=" * 80)
    print("最终游戏状态:")
    print("=" * 80)
    env.render()

    # 总结
    print("\n" + "=" * 80)
    print("🎉 快速开始完成！")
    print("=" * 80 + "\n")

    print("您已经学会了环境的基本用法：")
    print("  1. ✓ 创建环境: MahjongEnv()")
    print("  2. ✓ 重置环境: env.reset()")
    print("  3. ✓ 获取观测: env.observe(agent)")
    print("  4. ✓ 执行动作: env.step(action)")
    print("  5. ✓ 渲染状态: env.render()")

    print("\n接下来的步骤:")
    print("  • 查看 mahjong_environment/README.md 了解详细文档")
    print("  • 运行 python mahjong_environment/test_env.py 进行完整测试")
    print("  • 运行 python mahjong_environment/example_random_agent.py 查看完整示例")
    print("  • 开始实现你自己的麻将AI！")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        quickstart()
    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
