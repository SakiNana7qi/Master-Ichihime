# mahjong_environment/mahjong_env.py
"""
立直麻将环境 - PettingZoo风格的多智能体环境
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .game_state import GameState, RoundResult
from .player_state import PlayerState
from .utils.action_encoder import ActionEncoder
from .utils.legal_actions_helper import LegalActionsHelper
from .utils.tile_utils import format_hand, tile_to_unicode
from .utils.meld_helper import MeldHelper
from mahjong_scorer.main_scorer import MainScorer
from mahjong_scorer.utils.structures import HandInfo, GameState as ScorerGameState


class MahjongEnv:
    """
    立直麻将环境

    遵循PettingZoo风格的多智能体环境API
    支持4个玩家的完整麻将游戏模拟
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "mahjong_v0"}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        """
        初始化环境

        Args:
            render_mode: 渲染模式 ("human" 或 "ansi")
            seed: 随机种子
        """
        self.render_mode = render_mode
        self.seed = seed

        # 智能体列表
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agents = self.possible_agents.copy()

        # 当前需要行动的智能体
        self.agent_selection: Optional[str] = None

        # 游戏状态
        self.game_state: Optional[GameState] = None

        # 工具类
        self.action_encoder = ActionEncoder()
        self.legal_actions_helper = LegalActionsHelper()
        self.scorer = MainScorer()
        self.meld_helper = MeldHelper()

        # 定义动作空间（所有玩家共享）
        self.action_spaces = {
            agent: spaces.Discrete(ActionEncoder.NUM_ACTIONS)
            for agent in self.possible_agents
        }

        # 定义观测空间
        self._define_observation_space()

        # 累计奖励、终止、截断、信息
        self.rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self.terminations: Dict[str, bool] = {
            agent: False for agent in self.possible_agents
        }
        self.truncations: Dict[str, bool] = {
            agent: False for agent in self.possible_agents
        }
        self.infos: Dict[str, Dict] = {agent: {} for agent in self.possible_agents}

        # 上一次观测的缓存
        self._last_observations: Dict[str, Any] = {}

    def _define_observation_space(self):
        """定义观测空间"""
        # 观测空间包含：
        # 1. 手牌 (34维，每个元素0-4)
        # 2. 公开信息（其他玩家的牌河、副露、分数等）
        # 3. 动作掩码 (112维，0或1)

        obs_space = spaces.Dict(
            {
                # 自己的手牌 (34种牌，每种0-4张)
                "hand": spaces.Box(low=0, high=4, shape=(34,), dtype=np.int8),
                # 刚摸到的牌 (34维one-hot，或全0表示没有)
                "drawn_tile": spaces.Box(low=0, high=1, shape=(34,), dtype=np.int8),
                # 4个玩家的牌河 (每个玩家最多打24张牌，34维计数)
                "rivers": spaces.Box(low=0, high=24, shape=(4, 34), dtype=np.int8),
                # 4个玩家的副露信息 (简化表示：每种牌的副露数量)
                "melds": spaces.Box(low=0, high=4, shape=(4, 34), dtype=np.int8),
                # 4个玩家的立直状态 (0或1)
                "riichi_status": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
                # 4个玩家的分数 (归一化到0-1)
                "scores": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
                # 宝牌指示牌 (最多5个，34维one-hot的5个向量)
                "dora_indicators": spaces.Box(
                    low=0, high=1, shape=(5, 34), dtype=np.int8
                ),
                # 场况信息：[场风(0-3), 自风(0-3), 本场数/10, 立直棒数/4, 剩余牌数/70]
                "game_info": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
                # 当前阶段：[是否轮到自己, 是否打牌阶段, 是否响应阶段]
                "phase_info": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
                # 动作掩码 (112个动作的合法性)
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(ActionEncoder.NUM_ACTIONS,), dtype=np.int8
                ),
            }
        )

        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        重置环境到初始状态

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            Tuple[Dict, Dict]: (初始观测, 信息字典)
        """
        if seed is not None:
            self.seed = seed

        # 重置智能体列表
        self.agents = self.possible_agents.copy()

        # 创建新的游戏状态
        self.game_state = GameState(seed=self.seed)
        self.game_state.init_round()

        # 重置累计信息
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        # 设置第一个行动的玩家（庄家）
        self.agent_selection = f"player_{self.game_state.dealer}"

        # 获取初始观测
        obs = self.observe(self.agent_selection)
        self._last_observations[self.agent_selection] = obs

        return obs, self.infos[self.agent_selection]

    def step(self, action: int):
        """
        执行一个动作，推进游戏状态

        Args:
            action: 动作ID
        """
        if self.game_state is None:
            raise RuntimeError("游戏未初始化，请先调用reset()")

        # 获取当前玩家
        current_agent = self.agent_selection
        player_id = int(current_agent.split("_")[1])

        # 清空上一轮的奖励
        for agent in self.agents:
            self.rewards[agent] = 0.0

        # 解码动作
        action_type, action_params = self.action_encoder.decode_action(action)

        # 执行动作并推进状态
        self._execute_action(player_id, action_type, action_params)

        # 更新下一个需要行动的玩家
        self._update_agent_selection()

        # 检查游戏是否结束
        if self.game_state.phase == "end":
            self._handle_round_end()

    def observe(self, agent: str) -> Dict:
        """
        获取指定智能体的观测

        Args:
            agent: 智能体ID

        Returns:
            Dict: 观测字典
        """
        if self.game_state is None:
            # 返回空观测
            return self._get_empty_observation()

        player_id = int(agent.split("_")[1])
        player = self.game_state.players[player_id]

        # 构建观测
        obs = {}

        # 1. 手牌 (34维)
        obs["hand"] = np.array(player.get_tile_count_34(), dtype=np.int8)

        # 2. 刚摸到的牌 (34维one-hot)
        drawn_tile_vec = np.zeros(34, dtype=np.int8)
        if player.drawn_tile:
            idx = player._tile_to_index(player.drawn_tile)
            if idx != -1:
                drawn_tile_vec[idx] = 1
        obs["drawn_tile"] = drawn_tile_vec

        # 3. 牌河信息 (4x34)
        rivers = np.zeros((4, 34), dtype=np.int8)
        for i, p in enumerate(self.game_state.players):
            for tile in p.river:
                idx = p._tile_to_index(tile)
                if idx != -1:
                    rivers[i][idx] += 1
        obs["rivers"] = rivers

        # 4. 副露信息 (4x34)
        melds = np.zeros((4, 34), dtype=np.int8)
        for i, p in enumerate(self.game_state.players):
            for meld in p.open_melds:
                for tile in meld.tiles:
                    idx = p._tile_to_index(tile)
                    if idx != -1:
                        melds[i][idx] += 1
        obs["melds"] = melds

        # 5. 立直状态 (4维)
        riichi_status = np.array(
            [int(p.is_riichi) for p in self.game_state.players], dtype=np.int8
        )
        obs["riichi_status"] = riichi_status

        # 6. 分数 (4维，归一化)
        scores = np.array(
            [p.score / 100000.0 for p in self.game_state.players], dtype=np.float32
        )
        obs["scores"] = scores

        # 7. 宝牌指示牌 (5x34)
        dora_indicators = np.zeros((5, 34), dtype=np.int8)
        for i, dora in enumerate(self.game_state.dora_indicators[:5]):
            idx = player._tile_to_index(dora)
            if idx != -1:
                dora_indicators[i][idx] = 1
        obs["dora_indicators"] = dora_indicators

        # 8. 场况信息
        wind_map = {"east": 0, "south": 1, "west": 2, "north": 3}
        game_info = np.array(
            [
                wind_map[self.game_state.round_wind] / 3.0,  # 场风
                wind_map[player.seat_wind] / 3.0,  # 自风
                min(self.game_state.honba / 10.0, 1.0),  # 本场数
                min(self.game_state.riichi_sticks / 4.0, 1.0),  # 立直棒
                self.game_state.tiles_remaining / 70.0,  # 剩余牌数
            ],
            dtype=np.float32,
        )
        obs["game_info"] = game_info

        # 9. 阶段信息
        phase_info = np.array(
            [
                int(self.agent_selection == agent),  # 是否轮到自己
                int(self.game_state.phase == "discard"),  # 是否打牌阶段
                int(self.game_state.phase == "response"),  # 是否响应阶段
            ],
            dtype=np.int8,
        )
        obs["phase_info"] = phase_info

        # 10. 动作掩码
        _, action_mask = self.legal_actions_helper.get_legal_actions(
            player_id, self.game_state
        )
        obs["action_mask"] = np.array(action_mask, dtype=np.int8)

        return obs

    def render(self):
        """渲染当前游戏状态"""
        if self.game_state is None:
            print("游戏未开始")
            return

        if self.render_mode == "human" or self.render_mode == "ansi":
            self._render_text()

    def _render_text(self):
        """文本模式渲染"""
        print("\n" + "=" * 80)
        print(f"【{self._wind_name(self.game_state.round_wind)}场】")
        print(
            f"本场数: {self.game_state.honba}  立直棒: {self.game_state.riichi_sticks}  "
            f"剩余牌数: {self.game_state.tiles_remaining}"
        )
        print(
            f"阶段: {self.game_state.phase}  当前玩家: player_{self.game_state.current_player}"
        )
        print("=" * 80)

        for i, player in enumerate(self.game_state.players):
            is_current = i == self.game_state.current_player
            is_dealer = i == self.game_state.dealer
            marker = ">>>" if is_current else "   "

            print(f"\n{marker} 玩家{i} [{self._wind_name(player.seat_wind)}]", end="")
            if is_dealer:
                print(" [庄家]", end="")
            if player.is_riichi:
                print(" [立直]", end="")
            print(f"  分数: {player.score}")

            # 手牌（仅调试时显示，实际AI训练时不应该看到其他玩家手牌）
            if self.render_mode == "human":
                hand_str = format_hand(player.get_all_tiles())
                print(f"    手牌: {hand_str}")

            # 牌河
            if player.river:
                river_str = " ".join(player.river[-10:])  # 只显示最近10张
                if len(player.river) > 10:
                    river_str = "... " + river_str
                print(f"    牌河: {river_str}")

            # 副露
            if player.open_melds:
                melds_str = " | ".join(
                    [f"{m.meld_type}:{format_hand(m.tiles)}" for m in player.open_melds]
                )
                print(f"    副露: {melds_str}")

        print("\n" + "=" * 80)

        # 宝牌指示牌
        if self.game_state.dora_indicators:
            dora_str = " ".join(self.game_state.dora_indicators)
            print(f"宝牌指示牌: {dora_str}")

        print()

    def _execute_action(self, player_id: int, action_type: str, action_params: Dict):
        """
        执行玩家的动作

        Args:
            player_id: 玩家ID
            action_type: 动作类型
            action_params: 动作参数
        """
        player = self.game_state.players[player_id]

        if action_type == "discard":
            self._handle_discard(player_id, action_params)

        elif action_type == "chi":
            self._handle_chi(player_id, action_params)

        elif action_type == "pon":
            self._handle_pon(player_id)

        elif action_type == "minkan":
            self._handle_minkan(player_id)

        elif action_type == "ankan":
            self._handle_ankan(player_id)

        elif action_type == "kakan":
            self._handle_kakan(player_id, action_params)

        elif action_type == "tsumo":
            self._handle_tsumo(player_id)

        elif action_type == "ron":
            self._handle_ron(player_id)

        elif action_type == "pass":
            self._handle_pass(player_id)

        elif action_type == "kyuushu":
            self._handle_kyuushu(player_id)

    def _handle_discard(self, player_id: int, params: Dict):
        """处理打牌动作"""
        tile = params["tile"]
        with_riichi = params.get("riichi", False)

        player = self.game_state.players[player_id]

        # 如果是立直打牌
        if with_riichi:
            self.game_state.apply_riichi(player_id)

        # 判断是否为摸切
        is_tsumogiri = tile == player.drawn_tile

        # 打出牌
        self.game_state.discard_tile(player_id, tile, is_tsumogiri)

        # 进入响应阶段，询问其他玩家是否要鸣牌/和牌
        self.game_state.phase = "response"
        self.game_state.pending_responses = [i for i in range(4) if i != player_id]

    def _handle_chi(self, player_id: int, params: Dict):
        """处理吃牌动作"""
        chi_type = params.get("chi_type", "left")
        player = self.game_state.players[player_id]

        result = self.meld_helper.create_chi_meld(
            chi_type,
            player.get_all_tiles(),
            self.game_state.last_discard,
            self.game_state.last_discard_player,
        )

        if result is None:
            return

        meld, tiles_to_remove = result

        # 从手牌移除相应的牌
        for tile in tiles_to_remove:
            player.remove_tile(tile)

        # 添加副露
        player.add_meld(meld)

        # 吃牌后，轮到该玩家
        self.game_state.current_player = player_id
        self.game_state.phase = "discard"
        self.game_state.last_discard_can_be_claimed = False

        # 打断其他玩家的一发
        for p in self.game_state.players:
            if p.player_id != player_id:
                p.break_ippatsu()

    def _handle_pon(self, player_id: int):
        """处理碰牌动作"""
        player = self.game_state.players[player_id]

        result = self.meld_helper.create_pon_meld(
            player.get_all_tiles(),
            self.game_state.last_discard,
            self.game_state.last_discard_player,
        )

        if result is None:
            return

        meld, tiles_to_remove = result

        # 从手牌移除相应的牌
        for tile in tiles_to_remove:
            player.remove_tile(tile)

        # 添加副露
        player.add_meld(meld)

        # 碰牌后，轮到该玩家
        self.game_state.current_player = player_id
        self.game_state.phase = "discard"
        self.game_state.last_discard_can_be_claimed = False

        # 打断所有玩家的一发
        for p in self.game_state.players:
            p.break_ippatsu()

    def _handle_minkan(self, player_id: int):
        """处理明杠动作"""
        player = self.game_state.players[player_id]

        result = self.meld_helper.create_minkan_meld(
            player.get_all_tiles(),
            self.game_state.last_discard,
            self.game_state.last_discard_player,
        )

        if result is None:
            return

        meld, tiles_to_remove = result

        # 从手牌移除相应的牌
        for tile in tiles_to_remove:
            player.remove_tile(tile)

        # 添加副露
        player.add_meld(meld)

        # 明杠后，翻开新的宝牌指示牌
        self.game_state.reveal_dora()

        # 从王牌摸一张岭上牌
        tile = self.game_state.draw_tile_from_dead_wall(player_id)

        # 轮到该玩家
        self.game_state.current_player = player_id
        self.game_state.phase = "discard"
        self.game_state.last_discard_can_be_claimed = False

        # 打断所有玩家的一发
        for p in self.game_state.players:
            p.break_ippatsu()

    def _handle_ankan(self, player_id: int):
        """处理暗杠动作"""
        player = self.game_state.players[player_id]

        # 找到可以暗杠的牌
        ankan_tiles = self.meld_helper.can_ankan(player.get_all_tiles())

        if not ankan_tiles:
            return

        # 简化：杠第一种可以杠的牌
        target_tile = ankan_tiles[0]

        result = self.meld_helper.create_ankan_meld(player.get_all_tiles(), target_tile)

        if result is None:
            return

        meld, tiles_to_remove = result

        # 从手牌移除相应的牌
        for tile in tiles_to_remove:
            player.remove_tile(tile)

        # 添加副露
        player.add_meld(meld)

        # 暗杠后，翻开新的宝牌指示牌
        self.game_state.reveal_dora()

        # 从王牌摸一张岭上牌
        tile = self.game_state.draw_tile_from_dead_wall(player_id)

        # 继续该玩家的回合
        self.game_state.phase = "discard"

    def _handle_kakan(self, player_id: int, params: Dict):
        """处理加杠动作"""
        tile = params.get("tile", "")
        player = self.game_state.players[player_id]

        result = self.meld_helper.create_kakan_meld(
            player.get_all_tiles(), player.open_melds, tile
        )

        if result is None:
            return

        kakan_meld, original_pon, tile_to_remove = result

        # 从手牌移除这张牌
        player.remove_tile(tile_to_remove)

        # 替换副露（移除碰，添加杠）
        player.open_melds.remove(original_pon)
        player.add_meld(kakan_meld)

        # 加杠后，翻开新的宝牌指示牌
        self.game_state.reveal_dora()

        # 从王牌摸一张岭上牌
        tile = self.game_state.draw_tile_from_dead_wall(player_id)

        # 继续该玩家的回合
        self.game_state.phase = "discard"

        # 加杠可以被抢杠，需要询问其他玩家
        # 简化处理：暂不实现抢杠

    def _handle_tsumo(self, player_id: int):
        """处理自摸和动作"""
        player = self.game_state.players[player_id]

        # 构建HandInfo
        hand_info = HandInfo(
            hand_tiles=player.hand.copy(),
            open_melds=player.open_melds.copy(),
            winning_tile=player.drawn_tile,
            win_type="TSUMO",
        )

        # 构建GameState（算点器用）
        scorer_state = self._build_scorer_game_state(player_id)

        # 计算得分
        result = self.scorer.calculate_score(hand_info, scorer_state)

        if result.error:
            # 自摸失败（理论上不应该发生，因为已经检查过了）
            return

        # 创建结算结果
        score_deltas = [0, 0, 0, 0]
        score_deltas[player_id] = result.winner_gain

        for payment in result.payments:
            payer_id = self._get_player_id_from_wind(payment["from"])
            score_deltas[payer_id] = -payment["amount"]

        round_result = RoundResult(
            result_type="tsumo",
            winner=player_id,
            score_deltas=score_deltas,
            han=result.han,
            fu=result.fu,
            points=result.winner_gain,
        )

        self.game_state.end_round(round_result)

    def _handle_ron(self, player_id: int):
        """处理荣和动作"""
        player = self.game_state.players[player_id]

        # 构建HandInfo
        hand_info = HandInfo(
            hand_tiles=player.get_all_tiles(),
            open_melds=player.open_melds.copy(),
            winning_tile=self.game_state.last_discard,
            win_type="RON",
        )

        # 构建GameState（算点器用）
        scorer_state = self._build_scorer_game_state(player_id)

        # 计算得分
        result = self.scorer.calculate_score(hand_info, scorer_state)

        if result.error:
            return

        # 创建结算结果
        score_deltas = [0, 0, 0, 0]
        loser_id = self.game_state.last_discard_player
        score_deltas[player_id] = result.winner_gain
        score_deltas[loser_id] = -result.winner_gain

        round_result = RoundResult(
            result_type="ron",
            winner=player_id,
            loser=loser_id,
            score_deltas=score_deltas,
            han=result.han,
            fu=result.fu,
            points=result.winner_gain,
        )

        self.game_state.end_round(round_result)

    def _handle_pass(self, player_id: int):
        """处理跳过动作"""
        # 从待响应列表中移除
        if player_id in self.game_state.pending_responses:
            self.game_state.pending_responses.remove(player_id)

    def _handle_kyuushu(self, player_id: int):
        """处理九种九牌流局"""
        # 创建流局结果
        round_result = RoundResult(result_type="abort", score_deltas=[0, 0, 0, 0])
        self.game_state.end_round(round_result)

    def _update_agent_selection(self):
        """更新下一个需要行动的玩家"""
        if self.game_state.phase == "end":
            self.agent_selection = None
            return

        if self.game_state.phase == "response":
            # 如果还有待响应的玩家，选择第一个
            if self.game_state.pending_responses:
                next_player = self.game_state.pending_responses[0]
                self.agent_selection = f"player_{next_player}"
            else:
                # 无人响应，进入摸牌阶段
                next_player = self.game_state.next_player(
                    self.game_state.last_discard_player
                )
                self.game_state.current_player = next_player
                self.game_state.phase = "draw"

                # 摸牌
                tile = self.game_state.draw_tile(next_player)
                if tile is None:
                    # 流局
                    self._handle_draw()
                else:
                    # 进入打牌阶段
                    self.game_state.phase = "discard"
                    self.agent_selection = f"player_{next_player}"
        else:
            # 打牌阶段，当前玩家继续
            self.agent_selection = f"player_{self.game_state.current_player}"

    def _handle_draw(self):
        """处理流局"""
        # TODO: 检查听牌，计算听牌费
        round_result = RoundResult(result_type="draw", score_deltas=[0, 0, 0, 0])
        self.game_state.end_round(round_result)

    def _handle_round_end(self):
        """处理一局结束"""
        # 分配奖励
        for i, delta in enumerate(self.game_state.round_result.score_deltas):
            agent = f"player_{i}"
            self.rewards[agent] = delta / 1000.0  # 归一化奖励

        # 标记所有智能体终止
        for agent in self.agents:
            self.terminations[agent] = True

        # 更新信息
        for agent in self.agents:
            self.infos[agent] = {
                "round_result": self.game_state.round_result,
                "final_scores": [p.score for p in self.game_state.players],
            }

    def _build_scorer_game_state(self, player_id: int) -> ScorerGameState:
        """构建算点器使用的GameState"""
        player = self.game_state.players[player_id]

        return ScorerGameState(
            player_wind=player.seat_wind,
            prevalent_wind=self.game_state.round_wind,
            honba=self.game_state.honba,
            kyotaku_sticks=self.game_state.riichi_sticks,
            dora_indicators=self.game_state.dora_indicators.copy(),
            ura_dora_indicators=self.game_state.ura_dora_indicators.copy(),
            is_riichi=player.is_riichi,
            is_double_riichi=player.is_double_riichi,
            is_ippatsu=player.is_ippatsu,
        )

    def _get_player_id_from_wind(self, wind: str) -> int:
        """从风位获取玩家ID"""
        for i, player in enumerate(self.game_state.players):
            if player.seat_wind == wind:
                return i
        return 0

    def _wind_name(self, wind: str) -> str:
        """风位名称"""
        names = {"east": "东", "south": "南", "west": "西", "north": "北"}
        return names.get(wind, wind)

    def _get_empty_observation(self) -> Dict:
        """获取空观测（用于初始化）"""
        return {
            "hand": np.zeros(34, dtype=np.int8),
            "drawn_tile": np.zeros(34, dtype=np.int8),
            "rivers": np.zeros((4, 34), dtype=np.int8),
            "melds": np.zeros((4, 34), dtype=np.int8),
            "riichi_status": np.zeros(4, dtype=np.int8),
            "scores": np.zeros(4, dtype=np.float32),
            "dora_indicators": np.zeros((5, 34), dtype=np.int8),
            "game_info": np.zeros(5, dtype=np.float32),
            "phase_info": np.zeros(3, dtype=np.int8),
            "action_mask": np.zeros(ActionEncoder.NUM_ACTIONS, dtype=np.int8),
        }

    def last(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        返回上一个智能体的观测、奖励、终止、截断和信息

        Returns:
            Tuple: (observation, reward, termination, truncation, info)
        """
        agent = self.agent_selection
        if agent is None:
            agent = self.possible_agents[0]

        obs = self._last_observations.get(agent, self._get_empty_observation())
        return (
            obs,
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def close(self):
        """清理环境资源"""
        pass
