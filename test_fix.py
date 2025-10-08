#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的训练代码
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_model_output():
    """测试模型输出类型"""
    print("测试模型输出类型...")

    import torch
    from mahjong_agent import MahjongActorCritic, get_fast_config

    config = get_fast_config()
    model = MahjongActorCritic(config)
    model.eval()

    # 创建假数据
    batch_size = 1
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

    print(f"action shape: {action.shape}, type: {type(action)}")
    print(f"log_prob shape: {log_prob.shape}, type: {type(log_prob)}")
    print(f"entropy shape: {entropy.shape}, type: {type(entropy)}")
    print(f"value shape: {value.shape}, type: {type(value)}")

    # 测试转换
    try:
        action_np = int(action.cpu().item())
        log_prob_np = float(log_prob.cpu().item())
        entropy_np = float(entropy.cpu().item())

        if value.dim() > 0:
            value = value.squeeze()
        value_np = float(value.cpu().item())

        print("✓ 类型转换成功!")
        print(f"action_np: {action_np} (type: {type(action_np)})")
        print(f"log_prob_np: {log_prob_np} (type: {type(log_prob_np)})")
        print(f"entropy_np: {entropy_np} (type: {type(entropy_np)})")
        print(f"value_np: {value_np} (type: {type(value_np)})")

    except Exception as e:
        print(f"✗ 类型转换失败: {e}")
        return False

    return True


def test_rollout_buffer():
    """测试rollout buffer"""
    print("\n测试rollout buffer...")

    import torch
    import numpy as np
    from mahjong_agent import RolloutBuffer, get_fast_config

    config = get_fast_config()
    device = torch.device("cpu")
    buffer = RolloutBuffer(config, device)

    # 创建假数据
    obs = {
        "hand": np.zeros(34, dtype=np.int8),
        "drawn_tile": np.zeros(34, dtype=np.int8),
        "rivers": np.zeros((4, 34), dtype=np.int8),
        "melds": np.zeros((4, 34), dtype=np.int8),
        "riichi_status": np.zeros(4, dtype=np.int8),
        "scores": np.zeros(4, dtype=np.float32),
        "dora_indicators": np.zeros((5, 34), dtype=np.int8),
        "game_info": np.zeros(5, dtype=np.float32),
        "phase_info": np.zeros(3, dtype=np.int8),
    }
    action_mask = np.ones(112, dtype=np.int8)

    try:
        buffer.add(
            obs=obs,
            action=1,
            log_prob=0.5,
            reward=1.0,
            value=0.8,
            done=False,
            action_mask=action_mask,
        )
        print("✓ Rollout buffer添加成功!")

    except Exception as e:
        print(f"✗ Rollout buffer添加失败: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("测试修复后的代码")
    print("=" * 60)

    success1 = test_model_output()
    success2 = test_rollout_buffer()

    if success1 and success2:
        print("\n✓ 所有测试通过! 可以开始训练了!")
    else:
        print("\n✗ 还有问题需要修复")
