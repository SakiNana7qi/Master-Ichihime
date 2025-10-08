# mahjong_environment/utils/tile_utils.py
"""
麻将牌相关的工具函数
"""

from typing import List
import random
from collections import deque


def create_wall(use_red_fives: bool = True) -> List[str]:
    """
    创建完整的牌山（136张牌）

    Args:
        use_red_fives: 是否使用赤宝牌（赤5）

    Returns:
        List[str]: 牌山列表
    """
    wall = []

    # 数牌：万子、筒子、索子 (1-9)
    for suit in ["m", "p", "s"]:
        for num in range(1, 10):
            tile = f"{num}{suit}"
            # 每种牌4张
            count = 4

            # 如果使用赤宝牌，每种花色的5有一张是赤5
            if use_red_fives and num == 5:
                wall.append(f"0{suit}")  # 赤5
                count = 3  # 普通5只有3张

            wall.extend([tile] * count)

    # 字牌 (1z-7z: 东南西北白发中)
    for num in range(1, 8):
        tile = f"{num}z"
        wall.extend([tile] * 4)

    return wall


def shuffle_wall(wall: List[str], seed: int = None) -> deque:
    """
    洗牌并返回一个deque

    Args:
        wall: 牌山列表
        seed: 随机种子

    Returns:
        deque: 洗好的牌山
    """
    wall_copy = wall.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(wall_copy)
    return deque(wall_copy)


def tile_to_unicode(tile: str) -> str:
    """
    将牌的字符串表示转换为Unicode字符（用于美化显示）

    Args:
        tile: 牌的字符串表示 (如 "1m", "5z")

    Returns:
        str: Unicode字符
    """
    if len(tile) != 2:
        return tile

    num = tile[0]
    suit = tile[1]

    # Unicode麻将牌范围: U+1F000 - U+1F02F
    # 但这些字符在很多终端中显示不正确，暂时使用文字表示

    suit_names = {"m": "万", "p": "筒", "s": "索", "z": ""}

    if suit == "z":
        honor_names = {
            "1": "东",
            "2": "南",
            "3": "西",
            "4": "北",
            "5": "白",
            "6": "发",
            "7": "中",
        }
        return honor_names.get(num, tile)
    elif num == "0":
        return f"赤5{suit_names[suit]}"
    else:
        return f"{num}{suit_names[suit]}"


def format_hand(tiles: List[str], show_unicode: bool = False) -> str:
    """
    格式化显示手牌

    Args:
        tiles: 牌的列表
        show_unicode: 是否使用Unicode显示

    Returns:
        str: 格式化后的字符串
    """
    if not tiles:
        return "[]"

    if show_unicode:
        return " ".join([tile_to_unicode(t) for t in tiles])
    else:
        # 按花色分组显示
        manzu = [t for t in tiles if t.endswith("m")]
        pinzu = [t for t in tiles if t.endswith("p")]
        souzu = [t for t in tiles if t.endswith("s")]
        honors = [t for t in tiles if t.endswith("z")]

        result = []
        if manzu:
            result.append("".join([t[0] for t in sorted(manzu)]) + "m")
        if pinzu:
            result.append("".join([t[0] for t in sorted(pinzu)]) + "p")
        if souzu:
            result.append("".join([t[0] for t in sorted(souzu)]) + "s")
        if honors:
            result.append("".join([t[0] for t in sorted(honors)]) + "z")

        return " ".join(result)


def is_terminal(tile: str) -> bool:
    """判断是否为老头牌（1或9）"""
    if len(tile) != 2:
        return False
    return tile[0] in ["1", "9"] and tile[1] in ["m", "p", "s"]


def is_honor(tile: str) -> bool:
    """判断是否为字牌"""
    return len(tile) == 2 and tile[1] == "z"


def is_terminal_or_honor(tile: str) -> bool:
    """判断是否为幺九牌"""
    return is_terminal(tile) or is_honor(tile)


def is_simple(tile: str) -> bool:
    """判断是否为中张牌（2-8）"""
    if len(tile) != 2 or tile[1] not in ["m", "p", "s"]:
        return False
    num = tile[0]
    if num == "0":  # 赤5也是中张
        return True
    return num in ["2", "3", "4", "5", "6", "7", "8"]


def get_next_tile(tile: str) -> str:
    """
    获取下一张牌（用于宝牌指示牌）

    Args:
        tile: 指示牌

    Returns:
        str: 宝牌
    """
    if len(tile) != 2:
        return tile

    num = tile[0]
    suit = tile[1]

    # 赤5当作5处理
    if num == "0":
        num = "5"

    try:
        num_val = int(num)
    except ValueError:
        return tile

    if suit in ["m", "p", "s"]:
        # 数牌：9的下一张是1
        next_num = (num_val % 9) + 1
        return f"{next_num}{suit}"
    elif suit == "z":
        # 字牌：东南西北循环，白发中循环
        if num_val <= 4:  # 风牌 (1z-4z: 东南西北)
            next_num = (num_val % 4) + 1
        else:  # 三元牌 (5z-7z: 白发中)
            # 5z -> 6z, 6z -> 7z, 7z -> 5z
            if num_val == 5:
                next_num = 6
            elif num_val == 6:
                next_num = 7
            else:  # num_val == 7
                next_num = 5
        return f"{next_num}{suit}"

    return tile


def normalize_tile(tile: str) -> str:
    """
    规范化牌的表示（将赤5转换为普通5，用于某些判断）

    Args:
        tile: 牌

    Returns:
        str: 规范化后的牌
    """
    if len(tile) == 2 and tile[0] == "0":
        return f"5{tile[1]}"
    return tile


def tiles_are_same(tile1: str, tile2: str, ignore_red: bool = True) -> bool:
    """
    判断两张牌是否相同

    Args:
        tile1: 牌1
        tile2: 牌2
        ignore_red: 是否忽略赤宝牌的区别（赤5和普通5视为相同）

    Returns:
        bool: 是否相同
    """
    if ignore_red:
        return normalize_tile(tile1) == normalize_tile(tile2)
    return tile1 == tile2
