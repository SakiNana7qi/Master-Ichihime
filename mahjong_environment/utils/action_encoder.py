# mahjong_environment/utils/action_encoder.py
"""
动作编码解码器
将麻将的复杂动作映射到整数空间，便于强化学习使用
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class Action:
    """动作类"""

    action_type: str  # 'discard', 'chi', 'pon', 'kan', 'riichi', 'tsumo', 'ron', 'pass'
    tile: str = ""  # 相关的牌
    tiles: List[str] = None  # 用于吃（需要指定哪两张）
    kan_type: str = ""  # 'ankan', 'minkan', 'kakan'

    def __post_init__(self):
        if self.tiles is None:
            self.tiles = []


class ActionEncoder:
    """
    动作编码器

    动作空间设计：
    - 0-33: 打出手牌中的第i种牌（34种基本牌型）
    - 34-67: 打出手牌中的第i种牌并立直
    - 68-70: 吃（需要进一步指定吃的组合，简化为左/中/右吃）
    - 71: 碰
    - 72: 明杠
    - 73: 暗杠（需要进一步指定杠哪种牌）
    - 74-107: 加杠（34种牌）
    - 108: 自摸和
    - 109: 荣和
    - 110: 跳过/不响应
    - 111: 九种九牌流局

    总计：112个动作
    """

    NUM_ACTIONS = 112

    # 动作类型的起始索引
    DISCARD_START = 0
    DISCARD_RIICHI_START = 34
    CHI_START = 68
    PON = 71
    MINKAN = 72
    ANKAN = 73
    KAKAN_START = 74
    TSUMO = 108
    RON = 109
    PASS = 110
    KYUUSHU = 111

    def __init__(self):
        """初始化动作编码器"""
        pass

    @staticmethod
    def tile_to_tile_type(tile: str) -> int:
        """
        将牌转换为牌型索引 (0-33)
        赤5和普通5属于同一类型
        """
        if len(tile) != 2:
            return -1

        num = tile[0]
        suit = tile[1]

        # 赤宝牌当作5处理
        if num == "0":
            num = "5"

        try:
            num_val = int(num)
        except ValueError:
            return -1

        if suit == "m":  # 万子 0-8
            return num_val - 1
        elif suit == "p":  # 筒子 9-17
            return 9 + num_val - 1
        elif suit == "s":  # 索子 18-26
            return 18 + num_val - 1
        elif suit == "z":  # 字牌 27-33
            return 27 + num_val - 1

        return -1

    @staticmethod
    def tile_type_to_tile(tile_type: int) -> str:
        """将牌型索引转换为标准牌表示"""
        if tile_type < 0 or tile_type >= 34:
            return ""

        if tile_type < 9:  # 万子
            return f"{tile_type + 1}m"
        elif tile_type < 18:  # 筒子
            return f"{tile_type - 9 + 1}p"
        elif tile_type < 27:  # 索子
            return f"{tile_type - 18 + 1}s"
        else:  # 字牌
            return f"{tile_type - 27 + 1}z"

    @classmethod
    def encode_discard(cls, tile: str, with_riichi: bool = False) -> int:
        """
        编码打牌动作

        Args:
            tile: 要打出的牌
            with_riichi: 是否同时立直

        Returns:
            int: 动作编码
        """
        tile_type = cls.tile_to_tile_type(tile)
        if tile_type == -1:
            return -1

        if with_riichi:
            return cls.DISCARD_RIICHI_START + tile_type
        else:
            return cls.DISCARD_START + tile_type

    @classmethod
    def encode_chi(cls, chi_type: int = 0) -> int:
        """
        编码吃牌动作

        Args:
            chi_type: 吃的类型 (0=左吃, 1=中吃, 2=右吃)

        Returns:
            int: 动作编码
        """
        if chi_type < 0 or chi_type > 2:
            return -1
        return cls.CHI_START + chi_type

    @classmethod
    def encode_pon(cls) -> int:
        """编码碰牌动作"""
        return cls.PON

    @classmethod
    def encode_minkan(cls) -> int:
        """编码明杠动作"""
        return cls.MINKAN

    @classmethod
    def encode_ankan(cls) -> int:
        """编码暗杠动作"""
        return cls.ANKAN

    @classmethod
    def encode_kakan(cls, tile: str) -> int:
        """
        编码加杠动作

        Args:
            tile: 要加杠的牌

        Returns:
            int: 动作编码
        """
        tile_type = cls.tile_to_tile_type(tile)
        if tile_type == -1:
            return -1
        return cls.KAKAN_START + tile_type

    @classmethod
    def encode_tsumo(cls) -> int:
        """编码自摸和动作"""
        return cls.TSUMO

    @classmethod
    def encode_ron(cls) -> int:
        """编码荣和动作"""
        return cls.RON

    @classmethod
    def encode_pass(cls) -> int:
        """编码跳过动作"""
        return cls.PASS

    @classmethod
    def encode_kyuushu(cls) -> int:
        """编码九种九牌流局动作"""
        return cls.KYUUSHU

    @classmethod
    def decode_action(cls, action_id: int) -> Tuple[str, Dict]:
        """
        解码动作

        Args:
            action_id: 动作编码

        Returns:
            Tuple[str, Dict]: (动作类型, 动作参数)
        """
        if action_id < 0 or action_id >= cls.NUM_ACTIONS:
            return ("invalid", {})

        # 普通打牌
        if cls.DISCARD_START <= action_id < cls.DISCARD_RIICHI_START:
            tile_type = action_id - cls.DISCARD_START
            tile = cls.tile_type_to_tile(tile_type)
            return ("discard", {"tile": tile, "riichi": False})

        # 立直打牌
        elif cls.DISCARD_RIICHI_START <= action_id < cls.CHI_START:
            tile_type = action_id - cls.DISCARD_RIICHI_START
            tile = cls.tile_type_to_tile(tile_type)
            return ("discard", {"tile": tile, "riichi": True})

        # 吃
        elif cls.CHI_START <= action_id < cls.PON:
            chi_type = action_id - cls.CHI_START
            chi_names = ["left", "middle", "right"]
            return ("chi", {"chi_type": chi_names[chi_type]})

        # 碰
        elif action_id == cls.PON:
            return ("pon", {})

        # 明杠
        elif action_id == cls.MINKAN:
            return ("minkan", {})

        # 暗杠
        elif action_id == cls.ANKAN:
            return ("ankan", {})

        # 加杠
        elif cls.KAKAN_START <= action_id < cls.TSUMO:
            tile_type = action_id - cls.KAKAN_START
            tile = cls.tile_type_to_tile(tile_type)
            return ("kakan", {"tile": tile})

        # 自摸
        elif action_id == cls.TSUMO:
            return ("tsumo", {})

        # 荣和
        elif action_id == cls.RON:
            return ("ron", {})

        # 跳过
        elif action_id == cls.PASS:
            return ("pass", {})

        # 九种九牌
        elif action_id == cls.KYUUSHU:
            return ("kyuushu", {})

        return ("invalid", {})
