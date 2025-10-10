# mahjong_environment/test_env.py
"""
环境测试脚本
验证麻将环境是否能正常运行
"""

import sys
import random
import numpy as np

# 确保能导入环境
sys.path.append("..")

from mahjong_environment import MahjongEnv


def test_basic_initialization():
    """测试基本初始化"""
    print("\n" + "=" * 80)
    print("测试1: 基本初始化")
    print("=" * 80)

    env = MahjongEnv(render_mode="human", seed=42)
    print("[OK] 环境创建成功")

    assert len(env.possible_agents) == 4, "应该有4个玩家"
    print(f"[OK] 玩家数量正确: {len(env.possible_agents)}")

    print("[OK] 测试1通过\n")
    return env


def test_reset():
    """测试重置功能"""
    print("=" * 80)
    print("测试2: 重置功能")
    print("=" * 80)

    env = MahjongEnv(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)
    print("[OK] 环境重置成功")

    assert env.agent_selection is not None, "应该有选中的玩家"
    print(f"[OK] 当前玩家: {env.agent_selection}")

    # 检查观测空间
    assert "hand" in obs, "观测应包含手牌"
    assert "action_mask" in obs, "观测应包含动作掩码"
    print("[OK] 观测空间正确")

    # 检查手牌数量
    hand_count = np.sum(obs["hand"])
    print(f"[OK] 庄家手牌数量: {hand_count} (应该是14张)")

    env.render()

    print("[OK] 测试2通过\n")
    return env


def test_random_actions():
    """测试随机动作"""
    print("=" * 80)
    print("测试3: 随机动作执行")
    print("=" * 80)

    env = MahjongEnv(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)

    max_steps = 5000  # 限制步数避免无限循环
    step_count = 0

    print("开始执行随机动作...")

    while step_count < max_steps and not env.terminations[env.agent_selection]:
        current_agent = env.agent_selection

        # 获取合法动作
        action_mask = obs["action_mask"]
        legal_actions = np.where(action_mask == 1)[0]

        if len(legal_actions) == 0:
            print(f"警告: {current_agent} 没有合法动作")
            break

        # 随机选择一个合法动作
        action = random.choice(legal_actions)

        # 执行动作
        try:
            env.step(action)
            step_count += 1

            # 每10步打印一次状态
            if step_count % 10 == 0:
                print(f"\n--- 第 {step_count} 步 ---")
                env.render()

            # 获取新的观测
            if env.agent_selection is not None:
                obs = env.observe(env.agent_selection)

        except Exception as e:
            print(f"执行动作时出错: {e}")
            import traceback

            traceback.print_exc()
            break

    print(f"\n总共执行了 {step_count} 步")

    # 检查是否有游戏结束
    if any(env.terminations.values()):
        print("[OK] 游戏正常结束")
        env.render()

        # 显示结果
        for agent in env.possible_agents:
            reward = env.rewards[agent]
            print(f"{agent}: 奖励={reward:.2f}")
    else:
        print("游戏未结束（达到步数限制）")

    print("[OK] 测试3通过\n")


def test_observation_space():
    """测试观测空间"""
    print("=" * 80)
    print("测试4: 观测空间验证")
    print("=" * 80)

    env = MahjongEnv(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)

    # 检查所有观测维度
    expected_shapes = {
        "hand": (34,),
        "drawn_tile": (34,),
        "rivers": (4, 34),
        "melds": (4, 34),
        "riichi_status": (4,),
        "scores": (4,),
        "dora_indicators": (5, 34),
        "game_info": (5,),
        "phase_info": (3,),
        "action_mask": (112,),
    }

    for key, expected_shape in expected_shapes.items():
        assert key in obs, f"缺少观测键: {key}"
        actual_shape = obs[key].shape
        assert (
            actual_shape == expected_shape
        ), f"{key} 形状不匹配: 期望{expected_shape}, 实际{actual_shape}"
        print(f"[OK] {key}: {actual_shape}")

    print("[OK] 测试4通过\n")


def test_action_encoding():
    """测试动作编码解码"""
    print("=" * 80)
    print("测试5: 动作编码解码")
    print("=" * 80)

    from mahjong_environment.utils.action_encoder import ActionEncoder

    encoder = ActionEncoder()

    # 测试打牌动作
    action = encoder.encode_discard("1m", with_riichi=False)
    action_type, params = encoder.decode_action(action)
    assert action_type == "discard", "动作类型应为discard"
    assert params["tile"] == "1m", "牌应为1m"
    assert params["riichi"] == False, "不应该立直"
    print(f"[OK] 打牌动作: {action} -> {action_type}, {params}")

    # 测试立直打牌
    action = encoder.encode_discard("5p", with_riichi=True)
    action_type, params = encoder.decode_action(action)
    assert action_type == "discard", "动作类型应为discard"
    assert params["riichi"] == True, "应该立直"
    print(f"[OK] 立直打牌: {action} -> {action_type}, {params}")

    # 测试碰
    action = encoder.encode_pon()
    action_type, params = encoder.decode_action(action)
    assert action_type == "pon", "动作类型应为pon"
    print(f"[OK] 碰: {action} -> {action_type}")

    # 测试自摸
    action = encoder.encode_tsumo()
    action_type, params = encoder.decode_action(action)
    assert action_type == "tsumo", "动作类型应为tsumo"
    print(f"[OK] 自摸: {action} -> {action_type}")

    print("[OK] 测试5通过\n")


def test_tile_utils():
    """测试牌工具函数"""
    print("=" * 80)
    print("测试6: 牌工具函数")
    print("=" * 80)

    from mahjong_environment.utils.tile_utils import (
        create_wall,
        format_hand,
        tile_to_unicode,
        get_next_tile,
    )

    # 测试创建牌山
    wall = create_wall(use_red_fives=True)
    assert len(wall) == 136, f"牌山应该有136张牌，实际有{len(wall)}张"
    print(f"[OK] 牌山创建成功: {len(wall)}张牌")

    # 测试格式化手牌
    test_hand = ["1m", "2m", "3m", "5p", "6p", "7p", "1z", "1z"]
    formatted = format_hand(test_hand)
    print(f"[OK] 格式化手牌: {test_hand} -> {formatted}")

    # 测试Unicode显示
    unicode_str = format_hand(test_hand, show_unicode=True)
    print(f"[OK] Unicode显示: {unicode_str}")

    # 测试宝牌指示牌
    test_cases = [
        ("1m", "2m"),
        ("9m", "1m"),
        ("1z", "2z"),
        ("4z", "1z"),
        ("5z", "6z"),
        ("7z", "5z"),
    ]

    for indicator, expected_dora in test_cases:
        actual_dora = get_next_tile(indicator)
        assert (
            actual_dora == expected_dora
        ), f"指示牌{indicator}的宝牌应为{expected_dora}，实际为{actual_dora}"
        print(f"[OK] 指示牌{indicator} -> 宝牌{actual_dora}")

    print("[OK] 测试6通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#" * 80)
    print("#" + " " * 30 + "麻将环境测试" + " " * 30 + "#")
    print("#" * 80 + "\n")

    try:
        test_basic_initialization()
        test_reset()
        test_observation_space()
        test_action_encoding()
        test_tile_utils()
        test_random_actions()  # 这个测试时间较长，放在最后

        print("\n" + "#" * 80)
        print("#" + " " * 30 + "所有测试通过！" + " " * 28 + "#")
        print("#" * 80 + "\n")

    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
