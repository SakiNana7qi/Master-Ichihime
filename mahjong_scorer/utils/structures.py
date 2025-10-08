# mahjong_scorer/utils/structures.py
from dataclasses import dataclass, field
from typing import List, Literal, Dict

# 1m~9m 表示 1~9万
# 1p~9p 表示 1~9筒
# 1s~9s 表示 1~9索
# 1z~7z 表示东南西北白发中
# 0m0p0s 表示赤宝牌

Tile = str


@dataclass
class Meld:
    """记录副露"""

    meld_type: Literal["chi", "pon", "minkan", "ankan", "kakan"]
    tiles: List[Tile]
    from_who: int


@dataclass
class HandAnalysis:
    """手牌分析结果"""

    is_complete: bool = False
    decomposed_melds: List[List[Tile]] = field(default_factory=list)
    pair: List[Tile] = field(default_factory=list)
    wait_type: Literal[
        "ryammen",
        "kanchan",
        "penchan",
        "tanki",
        "shanpon",
        "",
    ] = ""  # 两面 坎张 边张 单骑 双碰
    is_menzenchin: bool = True
    fu_details: Dict[str, int] = field(default_factory=dict)  # 用于存储符数细节


@dataclass
class HandInfo:
    """和牌时手牌信息"""

    hand_tiles: List[Tile]
    open_melds: List[Meld] = field(default_factory=list)
    winning_tile: Tile = ""
    win_type: Literal["RON", "TSUMO"] = "RON"


@dataclass
class GameState:
    """描述牌局的全局信息"""

    player_wind: Literal["east", "south", "west", "north"] = "east"  # 自风
    prevalent_wind: Literal["east", "south", "west", "north"] = "east"  # 场风
    honba: int = 0
    kyotaku_sticks: int = 0

    dora_indicators: List[Tile] = field(default_factory=list)
    ura_dora_indicators: List[Tile] = field(default_factory=list)

    # 特殊状态
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_tenhou: bool = False
    is_chihou: bool = False

    @property
    def player_wind_tile(self) -> str:
        """返回自风对应的牌"""
        wind_map = {"east": "1z", "south": "2z", "west": "3z", "north": "4z"}
        return wind_map[self.player_wind]

    @property
    def prevalent_wind_tile(self) -> str:
        """返回场风对应的牌"""
        wind_map = {"east": "1z", "south": "2z", "west": "3z", "north": "4z"}
        return wind_map[self.prevalent_wind]

    @property
    def is_dealer(self) -> bool:
        """判断是否为庄家（东家）"""
        return self.player_wind == "east"


@dataclass
class YakuResult:
    name: str
    han: int


@dataclass
class ScoreResult:
    han: int = 0
    fu: int = 0
    base_points: int = 0
    winner_gain: int = 0
    payments: List[dict] = field(default_factory=list)
    yaku_list: List[YakuResult] = field(default_factory=list)
    error: str = ""
