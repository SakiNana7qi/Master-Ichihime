# mahjong_scorer/utils/tile_converter.py

# 字符串到整数的映射
STR_TO_INT = {
    **{f"{i}m": i - 1 for i in range(1, 10)},
    **{f"{i}p": i + 8 for i in range(1, 10)},
    **{f"{i}s": i + 17 for i in range(1, 10)},
    **{f"{i}z": i + 26 for i in range(1, 8)},
    "0m": 4,
    "0p": 13,
    "0s": 22,  # 赤宝牌映射到对应的5
}

# 整数到字符串的映射
INT_TO_STR = {v: k for k, v in STR_TO_INT.items() if not k.startswith("0")}
# 特殊处理赤宝牌，让5m/5p/5s能被正常转换
INT_TO_STR[4] = "5m"
INT_TO_STR[13] = "5p"
INT_TO_STR[22] = "5s"


def to_string(tile_int: int) -> str:
    """将整数牌转换为字符串"""
    return INT_TO_STR[tile_int]


def to_integer(tile_str: str) -> int:
    """将字符串牌转换为整数"""
    return STR_TO_INT[tile_str]


def hand_to_34_array(hand_tiles: list[str]) -> list[int]:
    """将字符串手牌列表转换为34维数组 (计数)"""
    array = [0] * 34
    for tile_str in hand_tiles:
        tile_int = to_integer(tile_str)
        # 即使是赤宝牌，也只在对应的普通牌位置上计数
        normal_tile_int = tile_int % 9 + (tile_int // 9) * 9
        if tile_str.startswith("0"):
            normal_tile_int = STR_TO_INT[f"5{tile_str[1]}"] - 1
        else:
            normal_tile_int = to_integer(f"{tile_str[0]}{tile_str[1]}")
        array[normal_tile_int] += 1
    return array


def get_dora_tile(indicator_str: str) -> str:
    """根据宝牌指示牌计算宝牌"""
    if indicator_str in ["0m", "0p", "0s"]:
        indicator_str = f"5{indicator_str[1]}"

    num, suit = int(indicator_str[0]), indicator_str[1]

    if suit == "z":  # 字牌
        if 1 <= num <= 4:  # 东南西北
            return f"{(num % 4) + 1}z"
        else:  # 白发中
            return f"{((num - 5) % 3) + 5}z"
    else:  # 数牌
        if num == 9:
            return f"1{suit}"
        else:
            return f"{num + 1}{suit}"
