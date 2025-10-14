#!/usr/bin/env python
"""
何切推断小程序

用法示例：
python -m mahjong_agent.infer_discard --model checkpoints/final_model.pt \
  --hand "1366m 349p 024578s" --drawn "9p" --seat E --round E --device cuda

仅基于当前手牌与摸牌，输出可打牌(非立直)的概率分布与Top-K建议。
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from .config import PPOConfig
from .model import MahjongActorCritic
from mahjong_environment.utils.action_encoder import ActionEncoder


def parse_tiles_grouped(grouped: str) -> List[str]:
    """解析类似 "1366m 349p 024578s 77z" 为具体牌列表。
    其中数字可包含0（赤5）。
    """
    tiles: List[str] = []
    if not grouped:
        return tiles
    for match in re.finditer(r"([0-9]+)([mpsz])", grouped.replace(" ", "")):
        digits, suit = match.group(1), match.group(2)
        for ch in digits:
            if ch.isdigit():
                tiles.append(f"{ch}{suit}")
    return tiles


def tile_to_index(tile: str) -> int:
    """与 ActionEncoder.tile_to_tile_type 一致（0-33）。"""
    return ActionEncoder.tile_to_tile_type(tile)


def build_obs_from_hand(
    hand_grouped: str,
    drawn_tile: Optional[str] = None,
    seat_wind: str = "E",
    round_wind: str = "E",
) -> Dict[str, np.ndarray]:
    hand_tiles = parse_tiles_grouped(hand_grouped)
    hand_counts = np.zeros(34, dtype=np.int8)
    for t in hand_tiles:
        idx = tile_to_index(t)
        if idx >= 0:
            hand_counts[idx] = min(4, hand_counts[idx] + 1)

    drawn_vec = np.zeros(34, dtype=np.int8)
    if drawn_tile:
        di = tile_to_index(drawn_tile)
        if di >= 0:
            # 从手牌计数中“挪出”一张到drawn向量（若存在）
            if hand_counts[di] > 0:
                hand_counts[di] = int(hand_counts[di]) - 1
            drawn_vec[di] = 1

    rivers = np.zeros((4, 34), dtype=np.int8)
    melds = np.zeros((4, 34), dtype=np.int8)
    riichi_status = np.zeros(4, dtype=np.int8)
    scores = np.zeros(4, dtype=np.float32)
    dora_indicators = np.zeros((5, 34), dtype=np.int8)

    wind_map = {"E": 0, "S": 1, "W": 2, "N": 3, "东": 0, "南": 1, "西": 2, "北": 3}
    gi_round = wind_map.get(round_wind, 0) / 3.0
    gi_seat = wind_map.get(seat_wind, 0) / 3.0
    game_info = np.array([gi_round, gi_seat, 0.0, 0.0, 1.0], dtype=np.float32)
    phase_info = np.array([1, 1, 0], dtype=np.int8)  # 轮到自己、打牌阶段

    # 仅允许“非立直打出”的可打牌（何切场景）
    mask = np.zeros(ActionEncoder.NUM_ACTIONS, dtype=np.int8)
    for tile_type in range(34):
        if hand_counts[tile_type] > 0 or (drawn_tile and drawn_vec[tile_type] == 1):
            mask[ActionEncoder.DISCARD_START + tile_type] = 1

    return {
        "hand": hand_counts,
        "drawn_tile": drawn_vec,
        "rivers": rivers,
        "melds": melds,
        "riichi_status": riichi_status,
        "scores": scores,
        "dora_indicators": dora_indicators,
        "game_info": game_info,
        "phase_info": phase_info,
        "action_mask": mask,
    }


def load_model(model_path: str, device: str = "cuda") -> Tuple[MahjongActorCritic, PPOConfig]:
    # 1) 安全加载 checkpoint/state_dict
    try:
        from torch.serialization import add_safe_globals, safe_globals  # type: ignore
        try:
            add_safe_globals([PPOConfig])
        except Exception:
            pass
        try:
            with safe_globals([PPOConfig]):
                state = torch.load(model_path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(model_path, map_location=device, weights_only=False)
    except Exception:
        state = torch.load(model_path, map_location=device)

    # 2) 提取 config 与权重（兼容多种保存格式）
    cfg_from_ckpt: Optional[PPOConfig] = None
    state_dict = None
    if isinstance(state, dict):
        if isinstance(state.get("model_state_dict"), dict):
            state_dict = state["model_state_dict"]
        elif isinstance(state.get("state_dict"), dict):
            state_dict = state["state_dict"]
        elif all(isinstance(k, str) for k in state.keys()):
            state_dict = state
        # 读取保存的配置
        ckpt_cfg = state.get("config")
        if ckpt_cfg is not None:
            try:
                if isinstance(ckpt_cfg, PPOConfig):
                    cfg_from_ckpt = ckpt_cfg
                elif isinstance(ckpt_cfg, dict):
                    base = PPOConfig()
                    for k, v in ckpt_cfg.items():
                        if hasattr(base, k):
                            setattr(base, k, v)
                    cfg_from_ckpt = base
            except Exception:
                cfg_from_ckpt = None
    else:
        state_dict = state

    if state_dict is None:
        state_dict = {}

    # 3) 用 checkpoint 配置构建模型，避免权重不匹配
    config = cfg_from_ckpt if cfg_from_ckpt is not None else PPOConfig()
    model = MahjongActorCritic(config).to(device)

    # 4) 兼容 torch.compile 的 _orig_mod. 前缀
    if isinstance(state_dict, dict) and any(isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys()):
        stripped = {}
        for k, v in state_dict.items():
            if isinstance(k, str) and k.startswith("_orig_mod."):
                stripped[k[len("_orig_mod."):]] = v
            else:
                stripped[k] = v
        state_dict = stripped

    # 5) 加载权重（非严格，以最大化兼容性）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(unexpected) > 0:
        print(f"[提示] 存在未使用权重键: {unexpected}")
    if len(missing) > 0:
        print(f"[提示] 存在缺失权重键: {missing}")

    model.eval()
    return model, config


def softmax_masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask: bool，非法位置置 -inf
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs


def infer_discard(
    model_path: str,
    hand: str,
    drawn: Optional[str],
    seat: str,
    round_wind: str,
    device: str = "cuda",
    topk: int = 10,
):
    model, _ = load_model(model_path, device=device)
    obs = build_obs_from_hand(hand, drawn, seat, round_wind)

    # 转换为 torch
    torch_obs: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        if k == "action_mask":
            continue
        torch_obs[k] = torch.from_numpy(v).unsqueeze(0).to(device)
    action_mask = torch.from_numpy(obs["action_mask"]).unsqueeze(0).to(device=device, dtype=torch.bool)

    with torch.no_grad(), torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"), dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits, _ = model.forward(torch_obs)
        # logits: (1, action_dim)
        probs = softmax_masked(logits.to(dtype=torch.float32), action_mask)[0].cpu().numpy()

    # 仅保留非立直打牌(0..33)
    discard_probs: List[Tuple[str, float]] = []
    for tile_type in range(34):
        idx = ActionEncoder.DISCARD_START + tile_type
        p = float(probs[idx])
        if p > 0:
            discard_probs.append((ActionEncoder.tile_type_to_tile(tile_type), p))

    discard_probs.sort(key=lambda x: x[1], reverse=True)
    print("\n建议何切 (Top-{}):".format(topk))
    for tile, p in discard_probs[:topk]:
        print(f"  {tile}: {p:.3f}")

    if len(discard_probs) > 0:
        best_tile, best_p = discard_probs[0]
        print(f"\n推荐：{best_tile}  概率={best_p:.3f}")
    else:
        print("\n无可打出牌（请检查输入手牌/摸牌）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--hand", type=str, required=True, help="当前手牌，形如 '1366m 349p 024578s 77z'")
    parser.add_argument("--drawn", type=str, default=None, help="刚摸到的牌，如 '9p'，可选")
    parser.add_argument("--seat", type=str, default="E", help="自风：E/S/W/N 或 东南西北")
    parser.add_argument("--round", dest="round_wind", type=str, default="E", help="场风：E/S/W/N 或 东南西北")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    infer_discard(
        model_path=args.model,
        hand=args.hand,
        drawn=args.drawn,
        seat=args.seat,
        round_wind=args.round_wind,
        device=args.device,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()


