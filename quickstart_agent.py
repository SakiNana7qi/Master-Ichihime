# quickstart_agent.py
"""
快速入门 - 麻将AI Agent
演示如何使用PPO训练和评估麻将AI
"""

import sys
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def demo_training():
    """演示训练流程（快速版本，用于测试）"""
    print("=" * 80)
    print(" " * 25 + "麻将AI训练快速演示")
    print("=" * 80 + "\n")

    from mahjong_agent import MahjongTrainer, get_fast_config

    # 使用快速配置（适合测试）
    config = get_fast_config()

    # 调整参数以便快速演示
    config.rollout_steps = 256  # 减少rollout步数
    config.total_timesteps = 10000  # 只训练10k步（演示用）
    config.log_interval = 2  # 更频繁地记录日志
    config.save_interval = 5  # 更频繁地保存
    config.verbose = True

    print("配置信息:")
    print(f"  设备: {config.device}")
    print(f"  总步数: {config.total_timesteps:,}")
    print(f"  Rollout步数: {config.rollout_steps}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  隐藏层维度: {config.hidden_dim}")
    print()

    # 创建训练器
    print("初始化训练器...")
    trainer = MahjongTrainer(config=config)

    print("\n开始训练...\n")

    # 开始训练
    try:
        trainer.train()
        print("\n✓ 训练完成！")
        print(f"模型已保存至: {config.save_dir}")
    except KeyboardInterrupt:
        print("\n训练被中断")
        print("保存当前模型...")
        trainer.save_checkpoint("interrupted.pt")
        print("✓ 模型已保存")


def demo_model_architecture():
    """演示模型架构"""
    print("=" * 80)
    print(" " * 25 + "麻将AI模型架构展示")
    print("=" * 80 + "\n")

    import torch
    from mahjong_agent import MahjongActorCritic, get_default_config

    config = get_default_config()

    # 创建模型
    model = MahjongActorCritic(config)

    print("模型结构:")
    print("-" * 80)
    print(model)
    print("-" * 80)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # 测试前向传播
    print("\n测试前向传播...")

    # 创建假数据
    batch_size = 4
    obs = {
        "hand": torch.zeros(batch_size, 34),
        "drawn_tile": torch.zeros(batch_size, 34),
        "rivers": torch.zeros(batch_size, 4, 34),
        "melds": torch.zeros(batch_size, 4, 34),
        "riichi_status": torch.zeros(batch_size, 4),
        "scores": torch.zeros(batch_size, 4),
        "dora_indicators": torch.zeros(batch_size, 5, 34),
        "game_info": torch.zeros(batch_size, 5),
        "phase_info": torch.zeros(batch_size, 3),
    }
    action_mask = torch.ones(batch_size, 112)

    with torch.no_grad():
        action, log_prob, entropy, value = model.get_action_and_value(
            obs, action_mask=action_mask
        )

    print(f"  输入批次大小: {batch_size}")
    print(f"  输出动作形状: {action.shape}")
    print(f"  价值估计形状: {value.shape}")
    print(f"  示例价值: {value[0].item():.3f}")
    print("\n✓ 模型测试通过！")


def demo_environment_interaction():
    """演示与环境的交互"""
    print("=" * 80)
    print(" " * 25 + "AI与环境交互演示")
    print("=" * 80 + "\n")

    import torch
    import numpy as np
    from mahjong_environment import MahjongEnv
    from mahjong_agent import MahjongActorCritic, get_default_config

    # 创建环境和模型
    env = MahjongEnv(render_mode="human", seed=42)
    config = get_default_config()
    model = MahjongActorCritic(config)
    model.eval()

    device = torch.device("cpu")
    model = model.to(device)

    print("环境和模型已创建\n")

    # 重置环境
    obs, info = env.reset(seed=42)
    print("环境已重置")
    print(f"当前玩家: {env.agent_selection}\n")

    # 执行几步
    steps = 10
    print(f"执行 {steps} 步...")

    for step in range(steps):
        if env.agent_selection is None:
            print("游戏结束")
            break

        current_agent = env.agent_selection
        action_mask = obs["action_mask"]

        # 准备观测
        torch_obs = {}
        for key, value in obs.items():
            if key != "action_mask":
                torch_obs[key] = torch.from_numpy(value).unsqueeze(0).to(device)

        torch_action_mask = torch.from_numpy(action_mask).unsqueeze(0).to(device)

        # 选择动作
        with torch.no_grad():
            action, log_prob, entropy, value = model.get_action_and_value(
                torch_obs, action_mask=torch_action_mask
            )

        action_np = action.cpu().item()

        # 解码动作
        from mahjong_environment.utils.action_encoder import ActionEncoder

        action_type, params = ActionEncoder.decode_action(action_np)

        print(f"\n步骤 {step + 1}: {current_agent}")
        print(f"  动作: {action_type} {params}")
        print(f"  价值: {value.cpu().item():.3f}")

        # 执行动作
        env.step(action_np)

        # 获取下一个观测
        if env.agent_selection is not None:
            obs = env.observe(env.agent_selection)

    print("\n" + "=" * 80)
    env.render()
    print("\n✓ 交互演示完成！")


def show_menu():
    """显示菜单"""
    print("\n" + "=" * 80)
    print(" " * 20 + "麻将AI Agent 快速入门菜单")
    print("=" * 80)
    print("\n请选择要执行的演示:")
    print("  1. 模型架构展示")
    print("  2. 环境交互演示")
    print("  3. 训练演示（快速版本）")
    print("  0. 退出")
    print()


def main():
    """主函数"""
    import sys

    print("\n🀄 欢迎使用麻将AI Agent系统！\n")

    while True:
        show_menu()

        try:
            choice = input("请输入选项 (0-3): ").strip()

            if choice == "0":
                print("\n再见！")
                break
            elif choice == "1":
                demo_model_architecture()
                input("\n按回车键继续...")
            elif choice == "2":
                demo_environment_interaction()
                input("\n按回车键继续...")
            elif choice == "3":
                confirm = (
                    input("\n训练将需要几分钟时间，确定继续? (y/n): ").strip().lower()
                )
                if confirm == "y":
                    demo_training()
                    input("\n按回车键继续...")
                else:
                    print("已取消")
            else:
                print("无效的选项，请重新选择")
        except KeyboardInterrupt:
            print("\n\n操作已取消")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback

            traceback.print_exc()
            input("\n按回车键继续...")


if __name__ == "__main__":
    main()
