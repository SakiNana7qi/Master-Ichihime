# mahjong_scorer/utils/__init__.py
"""
麻将算点器工具模块

包含常量定义、数据结构和牌的转换函数
"""

from .constants import *
from .structures import *
from .tile_converter import *

__all__ = [
    # 从 constants 导出
    "RIICHI",
    "PINFU",
    "TANYAO",
    "MANZU",
    "PINZU",
    "SOUZU",
    "HONORS",
    "WINDS",
    "DRAGONS",
    "TERMINALS",
    "TERMINALS_AND_HONORS",
    "SIMPLES",
    "GREEN_TILES",
    "WIND_MAP",
    "WIND_NAMES",
    "YAKU_HAN",
    "MUTUALLY_EXCLUSIVE_YAKU",
    "KUISAGARI_YAKU",
    # 从 structures 导出
    "HandInfo",
    "GameState",
    "ScoreResult",
    "HandAnalysis",
    "YakuResult",
    "Meld",
    "Tile",
    # 从 tile_converter 导出
    "to_string",
    "to_integer",
    "hand_to_34_array",
    "get_dora_tile",
]
