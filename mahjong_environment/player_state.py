# mahjong_environment/player_state.py
"""
玩家状态管理类
负责管理单个玩家的所有状态信息
"""

from dataclasses import dataclass, field
from typing import List, Literal
from mahjong_scorer.utils.structures import Meld


@dataclass
class PlayerState:
    """玩家状态类，管理单个玩家的所有信息"""

    # 基本信息
    player_id: int  # 玩家ID (0-3)
    seat_wind: Literal["east", "south", "west", "north"]  # 自风

    # 手牌信息（私密）
    hand: List[str] = field(default_factory=list)  # 手牌（暗牌）
    drawn_tile: str = ""  # 刚摸到的牌（用于区分）

    # 公开信息
    river: List[str] = field(default_factory=list)  # 牌河（打出的牌）
    river_tsumogiri: List[bool] = field(default_factory=list)  # 标记哪些是摸切
    open_melds: List[Meld] = field(default_factory=list)  # 副露

    # 立直状态
    is_riichi: bool = False
    is_double_riichi: bool = False
    riichi_turn: int = -1  # 立直时的巡目
    riichi_stick_placed: bool = False  # 是否已放置立直棒

    # 状态标记
    is_ippatsu: bool = False  # 一发状态（立直后未被中断）
    is_furiten: bool = False  # 振听状态
    is_menzen: bool = True  # 门前状态（没有吃碰）

    # 分数
    score: int = 25000  # 点数（初始25000点）

    # 行动历史
    last_discard: str = ""  # 最后打出的牌
    can_act: bool = False  # 当前是否可以行动

    def add_tile(self, tile: str, is_drawn: bool = False):
        """添加牌到手牌"""
        if is_drawn:
            self.drawn_tile = tile
        else:
            self.hand.append(tile)
            self.hand.sort()

    def remove_tile(self, tile: str) -> bool:
        """从手牌中移除指定的牌"""
        if tile == self.drawn_tile:
            self.drawn_tile = ""
            return True
        elif tile in self.hand:
            self.hand.remove(tile)
            return True
        return False

    def discard_tile(self, tile: str, is_tsumogiri: bool = False):
        """打出一张牌到牌河"""
        if self.remove_tile(tile):
            # 若不是摸切，则表示从手牌打出；根据麻将规则，本巡摸到的牌应并入手牌
            # 以保持每巡结束后“手牌始终为13张”。
            if not is_tsumogiri and self.drawn_tile:
                self.hand.append(self.drawn_tile)
                self.hand.sort()
                self.drawn_tile = ""
            self.river.append(tile)
            self.river_tsumogiri.append(is_tsumogiri)
            self.last_discard = tile
            return True
        return False

    def add_meld(self, meld: Meld):
        """添加副露"""
        self.open_melds.append(meld)
        if meld.meld_type in ["chi", "pon"]:
            self.is_menzen = False

    def get_all_tiles(self) -> List[str]:
        """获取所有手牌（包括刚摸到的牌）"""
        tiles = self.hand.copy()
        if self.drawn_tile:
            tiles.append(self.drawn_tile)
        return sorted(tiles)

    def get_tile_count_34(self) -> List[int]:
        """
        将手牌转换为34维向量
        返回: [1m数量, 2m数量, ..., 9m, 1p, ..., 9p, 1s, ..., 9s, 1z, ..., 7z]
        """
        tile_count = [0] * 34
        all_tiles = self.get_all_tiles()

        for tile in all_tiles:
            idx = self._tile_to_index(tile)
            if idx != -1:
                tile_count[idx] += 1

        return tile_count

    def _tile_to_index(self, tile: str) -> int:
        """将牌转换为索引 (0-33)"""
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

    def declare_riichi(self, turn: int):
        """宣告立直"""
        self.is_riichi = True
        self.riichi_turn = turn
        self.is_ippatsu = True  # 进入一发状态

    def break_ippatsu(self):
        """打破一发状态"""
        self.is_ippatsu = False

    def reset(self):
        """重置玩家状态（新的一局）"""
        self.hand = []
        self.drawn_tile = ""
        self.river = []
        self.river_tsumogiri = []
        self.open_melds = []
        self.is_riichi = False
        self.is_double_riichi = False
        self.riichi_turn = -1
        self.riichi_stick_placed = False
        self.is_ippatsu = False
        self.is_furiten = False
        self.is_menzen = True
        self.last_discard = ""
        self.can_act = False
