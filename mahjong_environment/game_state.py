# mahjong_environment/game_state.py
"""
游戏状态管理器
负责管理整个麻将游戏的核心状态和逻辑
"""

from typing import List, Optional, Tuple, Dict, Literal
from collections import deque
from dataclasses import dataclass, field

from .player_state import PlayerState
from .utils.tile_utils import create_wall, shuffle_wall, get_next_tile
from mahjong_scorer.utils.structures import Meld


@dataclass
class RoundResult:
    """一局的结果"""

    result_type: Literal["ron", "tsumo", "draw", "abort"]  # 和牌/自摸/流局/中途流局
    winner: Optional[int] = None  # 和牌者ID（可能有多个，这里简化为单个）
    loser: Optional[int] = None  # 放铳者ID（仅荣和时有效）
    score_deltas: List[int] = field(default_factory=list)  # 各玩家的分数变化
    han: int = 0
    fu: int = 0
    points: int = 0
    # 役种名称列表（用于评估/交互输出）
    yaku_names: List[str] = field(default_factory=list)


class GameState:
    """
    游戏状态管理器

    负责管理：
    - 牌山和王牌
    - 宝牌指示牌
    - 四位玩家的状态
    - 当前回合和游戏阶段
    - 场况信息（东几局、本场数等）
    """

    def __init__(self, seed: Optional[int] = None):
        """
        初始化游戏状态

        Args:
            seed: 随机种子
        """
        self.seed = seed

        # 牌山相关
        self.wall: deque = deque()  # 牌山
        self.dead_wall: List[str] = []  # 王牌（14张）
        self.dora_indicators: List[str] = []  # 宝牌指示牌
        self.ura_dora_indicators: List[str] = []  # 里宝牌指示牌

        # 玩家状态
        self.players: List[PlayerState] = []
        for i in range(4):
            wind = ["east", "south", "west", "north"][i]
            self.players.append(PlayerState(player_id=i, seat_wind=wind))

        # 场况信息
        self.round_wind: Literal["east", "south", "west", "north"] = "east"  # 场风
        self.round_number: int = 0  # 局数（0=东一局, 1=东二局, ..., 4=南一局, ...）
        self.honba: int = 0  # 本场数
        self.riichi_sticks: int = 0  # 立直棒数量

        # 游戏进行状态
        self.current_player: int = 0  # 当前行动的玩家
        self.dealer: int = 0  # 庄家
        self.turn_count: int = 0  # 巡目（从0开始）
        self.tiles_remaining: int = 0  # 剩余可摸牌数

        # 游戏阶段状态机
        self.phase: Literal[
            "deal",  # 发牌阶段
            "draw",  # 摸牌阶段
            "discard",  # 打牌阶段
            "response",  # 等待其他玩家响应（吃碰杠和）
            "end",  # 一局结束
        ] = "deal"

        # 最近的打牌信息（用于响应）
        self.last_discard: str = ""
        self.last_discard_player: int = -1
        self.last_discard_can_be_claimed: bool = False

        # 响应队列（记录哪些玩家可以响应）
        self.pending_responses: List[int] = []  # 待响应的玩家列表
        self.response_actions: Dict[int, str] = {}  # 玩家ID -> 响应动作类型

        # 一局结束信息
        self.round_result: Optional[RoundResult] = None

    def init_round(self):
        """初始化新的一局游戏"""
        # 重置玩家状态
        for player in self.players:
            player.reset()

        # 更新玩家的自风（根据庄家位置）
        winds = ["east", "south", "west", "north"]
        for i in range(4):
            self.players[i].seat_wind = winds[(i - self.dealer) % 4]

        # 创建并洗牌
        wall = create_wall(use_red_fives=True)
        self.wall = shuffle_wall(wall, self.seed)

        # 分离王牌（最后14张）
        self.dead_wall = [self.wall.pop() for _ in range(14)]
        self.dead_wall.reverse()  # 翻转使其顺序正确

        # 设置宝牌指示牌（王牌的第3张，后续可能翻开更多）
        self.dora_indicators = [self.dead_wall[4]]  # 初始宝牌指示牌
        self.ura_dora_indicators = [self.dead_wall[9]]  # 里宝牌（立直后才能看）

        # 发牌：每人13张
        for _ in range(13):
            for player_id in range(4):
                if self.wall:
                    tile = self.wall.popleft()
                    self.players[player_id].add_tile(tile)

        # 庄家多摸一张（14张）
        if self.wall:
            tile = self.wall.popleft()
            self.players[self.dealer].add_tile(tile, is_drawn=True)

        # 初始化游戏状态
        self.current_player = self.dealer
        self.turn_count = 0
        self.tiles_remaining = len(self.wall)
        self.phase = "discard"  # 庄家首先打牌
        self.last_discard = ""
        self.last_discard_player = -1
        self.last_discard_can_be_claimed = False
        self.pending_responses = []
        self.response_actions = {}
        self.round_result = None

    def draw_tile(self, player_id: int) -> Optional[str]:
        """
        玩家从牌山摸牌

        Args:
            player_id: 玩家ID

        Returns:
            Optional[str]: 摸到的牌，如果牌山为空则返回None
        """
        if not self.wall:
            return None

        tile = self.wall.popleft()
        self.players[player_id].add_tile(tile, is_drawn=True)
        self.tiles_remaining = len(self.wall)
        return tile

    def draw_tile_from_dead_wall(self, player_id: int) -> Optional[str]:
        """
        从王牌摸牌（岭上牌，用于杠后摸牌）

        Args:
            player_id: 玩家ID

        Returns:
            Optional[str]: 摸到的牌
        """
        # 从王牌末尾摸牌（岭上牌）
        if len(self.dead_wall) > 5:  # 保留宝牌指示牌区域
            tile = self.dead_wall.pop()
            self.players[player_id].add_tile(tile, is_drawn=True)
            # 杠后需要补充王牌（从牌山末尾）
            if self.wall:
                self.dead_wall.insert(0, self.wall.pop())
            return tile
        return None

    def discard_tile(
        self, player_id: int, tile: str, is_tsumogiri: bool = False
    ) -> bool:
        """
        玩家打出一张牌

        Args:
            player_id: 玩家ID
            tile: 要打出的牌
            is_tsumogiri: 是否为摸切（打出刚摸到的牌）

        Returns:
            bool: 是否成功打出
        """
        player = self.players[player_id]
        if player.discard_tile(tile, is_tsumogiri):
            self.last_discard = tile
            self.last_discard_player = player_id
            self.last_discard_can_be_claimed = True
            return True
        return False

    def add_meld(self, player_id: int, meld: Meld):
        """
        添加副露

        Args:
            player_id: 玩家ID
            meld: 副露信息
        """
        self.players[player_id].add_meld(meld)

    def reveal_dora(self):
        """翻开新的宝牌指示牌（杠后）"""
        # 每次杠，翻开下一个宝牌指示牌
        dora_count = len(self.dora_indicators)
        if dora_count < 5:  # 最多5个宝牌指示牌
            new_dora_idx = 4 + dora_count
            if new_dora_idx < len(self.dead_wall):
                self.dora_indicators.append(self.dead_wall[new_dora_idx])
                # 同时确定对应的里宝牌
                ura_dora_idx = 9 + dora_count
                if ura_dora_idx < len(self.dead_wall):
                    self.ura_dora_indicators.append(self.dead_wall[ura_dora_idx])

    def next_player(self, current: int) -> int:
        """获取下一个玩家的ID"""
        return (current + 1) % 4

    def is_draw(self) -> bool:
        """判断是否流局（牌山为空且剩余牌数达到流局标准）"""
        # 通常当剩余牌数为0时（王牌14张保留）
        return self.tiles_remaining == 0

    def can_riichi(self, player_id: int) -> bool:
        """
        判断玩家是否可以立直

        Args:
            player_id: 玩家ID

        Returns:
            bool: 是否可以立直
        """
        player = self.players[player_id]

        # 已经立直了
        if player.is_riichi:
            return False

        # 不是门前清
        if not player.is_menzen:
            return False

        # 点数不足1000点（需要放立直棒）
        if player.score < 1000:
            return False

        # 剩余牌数不足4张（不能立直）
        if self.tiles_remaining < 4:
            return False

        return True

    def apply_riichi(self, player_id: int):
        """
        应用立直

        Args:
            player_id: 玩家ID
        """
        player = self.players[player_id]

        # 判断是否为两立直（第一巡且无人鸣牌）
        if self.turn_count == 0 and all(len(p.open_melds) == 0 for p in self.players):
            player.is_double_riichi = True

        player.declare_riichi(self.turn_count)
        player.score -= 1000  # 放置立直棒
        self.riichi_sticks += 1

    def get_public_state(self) -> Dict:
        """
        获取公开信息（所有玩家都能看到的信息）

        Returns:
            Dict: 公开状态信息
        """
        return {
            "round_wind": self.round_wind,
            "round_number": self.round_number,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "dealer": self.dealer,
            "current_player": self.current_player,
            "turn_count": self.turn_count,
            "tiles_remaining": self.tiles_remaining,
            "dora_indicators": self.dora_indicators.copy(),
            "last_discard": self.last_discard,
            "last_discard_player": self.last_discard_player,
            "phase": self.phase,
            "players_scores": [p.score for p in self.players],
            "players_riichi": [p.is_riichi for p in self.players],
            "players_rivers": [p.river.copy() for p in self.players],
            "players_melds": [
                [
                    {
                        "type": m.meld_type,
                        "tiles": m.tiles.copy(),
                        "from_who": m.from_who,
                    }
                    for m in p.open_melds
                ]
                for p in self.players
            ],
        }

    def end_round(self, result: RoundResult):
        """
        结束当前局

        Args:
            result: 一局的结果
        """
        self.round_result = result
        self.phase = "end"

        # 应用分数变化
        for i, delta in enumerate(result.score_deltas):
            self.players[i].score += delta

        # 更新庄家和本场数（基本规则）：
        # 和了：庄家和则连庄(+1本场)；闲家和则庄移转，本场清零
        # 流局：按听牌连庄；此处暂保守为连庄(+1)
        if result.result_type in ["ron", "tsumo"]:
            if result.winner == self.dealer:
                self.honba += 1
            else:
                self.dealer = self.next_player(self.dealer)
                self.honba = 0
        elif result.result_type == "draw":
            self.honba += 1

    def advance_round(self):
        """进入下一局（东二局、南一局等）"""
        self.round_number += 1

        # 更新场风
        if self.round_number < 4:
            self.round_wind = "east"
        elif self.round_number < 8:
            self.round_wind = "south"
        else:
            # 南场结束，游戏结束
            pass
