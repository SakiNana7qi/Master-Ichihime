# mahjong_scorer/__init__.py
"""
雀魂立直麻将算点器

这个包提供了完整的日本麻将（立直麻将）算点功能，包括：
- 手牌分析和面子分解
- 役种判断（包括所有常见役和役满）
- 符数计算
- 点数分配
- 宝牌计算

主要使用方式：
    from mahjong_scorer import MainScorer, HandInfo, GameState

    scorer = MainScorer()
    result = scorer.calculate_score(hand_info, game_state)
"""

from .main_scorer import MainScorer
from .hand_analyzer import HandAnalyzer
from .yaku_checker import YakuChecker
from .fu_calculator import FuCalculator
from .point_distributor import PointDistributor

from .utils.structures import (
    HandInfo,
    GameState,
    ScoreResult,
    HandAnalysis,
    YakuResult,
    Meld,
    Tile,
)

from .utils.tile_converter import (
    to_string,
    to_integer,
    hand_to_34_array,
    get_dora_tile,
)

__version__ = "1.0.0"
__author__ = "Master-Ichihime"

__all__ = [
    # 主要类
    "MainScorer",
    "HandAnalyzer",
    "YakuChecker",
    "FuCalculator",
    "PointDistributor",
    # 数据结构
    "HandInfo",
    "GameState",
    "ScoreResult",
    "HandAnalysis",
    "YakuResult",
    "Meld",
    "Tile",
    # 工具函数
    "to_string",
    "to_integer",
    "hand_to_34_array",
    "get_dora_tile",
]
