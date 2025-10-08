#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试修复
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_value_conversion():
    """测试value转换"""
    print("Testing value conversion...")

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

    print(f"value shape: {value.shape}")
    print(f"value numel: {value.numel()}")
    print(f"value dim: {value.dim()}")

    # 测试转换
    try:
        if value.numel() > 1:
            value = value.flatten()[0]
        elif value.dim() > 0:
            value = value.squeeze()
        value_np = float(value.cpu().item())

        print(f"SUCCESS! value_np: {value_np} (type: {type(value_np)})")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_value_conversion()
    if success:
        print("\nReady to train!")
    else:
        print("\nStill has issues")
