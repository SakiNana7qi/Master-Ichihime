# mahjong_environment/utils/legal_actions_helper.py
"""
合法动作检测器
判断在当前状态下，玩家可以执行哪些动作
"""

from typing import List, Set, Tuple
import numpy as np
import sys
import os

# 添加父目录到路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mahjong_environment.player_state import PlayerState
from mahjong_environment.game_state import GameState
from mahjong_environment.utils.action_encoder import ActionEncoder
from mahjong_environment.utils.tile_utils import normalize_tile, tiles_are_same
from mahjong_scorer.main_scorer import MainScorer
try:
    from mahjong_environment.fastops import FASTOPS_AVAILABLE, compute_fast_actions
except Exception:
    FASTOPS_AVAILABLE = False


class LegalActionsHelper:
    """合法动作检测器"""

    def __init__(self):
        self.encoder = ActionEncoder()
        self.scorer = MainScorer()

    def get_legal_actions(
        self, player_id: int, game_state: GameState
    ) -> Tuple[List[int], List[bool]]:
        """
        获取玩家的所有合法动作

        Args:
            player_id: 玩家ID
            game_state: 游戏状态

        Returns:
            Tuple[List[int], List[bool]]: (合法动作列表, 动作掩码数组)
        """
        legal_actions = []
        action_mask = [False] * ActionEncoder.NUM_ACTIONS

        player = game_state.players[player_id]
        phase = game_state.phase

        # 快速路径：训练期可用Cython近似掩码（显著加速）；评估/交互应将 game_state.fast_mask 设为 False
        if FASTOPS_AVAILABLE and getattr(game_state, "fast_mask", True):
            # phase_info: [is_turn, is_discard, is_response]
            is_discard = int(phase == "discard")
            is_response = int(phase == "response")
            phase_code = 1 if is_discard else (2 if is_response else 0)
            # 准备 hand34 / drawn34 / riichi4
            hand34 = player.get_tile_count_34()
            drawn34 = [0]*34
            if player.drawn_tile:
                try:
                    idx = player._tile_to_index(player.drawn_tile)
                    if idx >= 0:
                        drawn34[idx] = 1
                except Exception:
                    pass
            riichi4 = [int(p.is_riichi) for p in game_state.players]
            last_idx = -1
            if getattr(game_state, "last_discard", None):
                try:
                    last_idx = player._tile_to_index(game_state.last_discard)
                except Exception:
                    last_idx = -1
            last_player = getattr(game_state, "last_discard_player", -1)

            try:
                mask_np = compute_fast_actions(
                    np.asarray(hand34, dtype=np.int8),
                    np.asarray(drawn34, dtype=np.int8),
                    np.asarray(riichi4, dtype=np.int8),
                    phase_code, int(player_id), int(last_idx), int(last_player)
                )
                # 回填为布尔列表
                action_mask = [bool(int(x)) for x in np.asarray(mask_np, dtype=np.int8)]
                # 训练时仅依赖掩码，直接返回以避免后续昂贵的精确判定
                return [], action_mask
            except Exception:
                pass

        if phase == "discard":
            # 打牌阶段：可以打出手中的任何一张牌
            legal_actions.extend(self._get_discard_actions(player, game_state))

            # 可以自摸和
            if self._can_tsumo(player, game_state):
                legal_actions.append(ActionEncoder.TSUMO)

            # 可以暗杠
            legal_actions.extend(self._get_ankan_actions(player))

            # 可以加杠
            legal_actions.extend(self._get_kakan_actions(player))

            # 可以九种九牌流局（第一巡，手中有9种幺九牌）
            if self._can_kyuushu(player, game_state):
                legal_actions.append(ActionEncoder.KYUUSHU)

        elif phase == "response":
            # 响应阶段：可以响应他人的打牌
            if player_id in game_state.pending_responses:
                # 可以荣和
                if self._can_ron(player, game_state):
                    legal_actions.append(ActionEncoder.RON)

                # 可以碰
                if self._can_pon(player, game_state):
                    legal_actions.append(ActionEncoder.PON)

                # 可以明杠
                if self._can_minkan(player, game_state):
                    legal_actions.append(ActionEncoder.MINKAN)

                # 可以吃（只有下家）
                if self._is_next_player(player_id, game_state.last_discard_player):
                    chi_actions = self._get_chi_actions(player, game_state)
                    legal_actions.extend(chi_actions)

                # 总是可以跳过
                legal_actions.append(ActionEncoder.PASS)

        # 将合法动作转换为掩码（精确路径覆盖快速路径）
        for action in legal_actions:
            if 0 <= action < ActionEncoder.NUM_ACTIONS:
                action_mask[action] = True

        return legal_actions, action_mask

    def _get_discard_actions(
        self, player: PlayerState, game_state: GameState
    ) -> List[int]:
        """获取所有可以打出的牌的动作"""
        actions = []
        all_tiles = player.get_all_tiles()

        # 去重：同种牌只需要一个动作
        unique_tiles = set()
        for tile in all_tiles:
            tile_type = ActionEncoder.tile_to_tile_type(tile)
            if tile_type not in unique_tiles:
                unique_tiles.add(tile_type)
                action = ActionEncoder.encode_discard(tile, with_riichi=False)
                if action != -1:
                    actions.append(action)

                # 如果可以立直，添加立直打牌动作
                if (not player.is_riichi) and game_state.can_riichi(
                    player.player_id
                ) and self._is_tenpai_after_discard(player, tile):
                    action_riichi = ActionEncoder.encode_discard(tile, with_riichi=True)
                    if action_riichi != -1:
                        actions.append(action_riichi)

        return actions

    def _can_tsumo(self, player: PlayerState, game_state: GameState) -> bool:
        """判断是否可以自摸和：必须形完且有役。"""
        if not player.drawn_tile:
            return False
        from mahjong_scorer.utils.structures import HandInfo
        hand_info = HandInfo(
            hand_tiles=player.hand.copy(),  # 不包含摸牌
            open_melds=player.open_melds.copy(),
            winning_tile=player.drawn_tile,
            win_type="TSUMO",
        )
        # 严格用算点器判定（包含役种判定），无役则不可和
        score = self.scorer.calculate_score(hand_info, self._to_scorer_gamestate(player, game_state))
        return not bool(getattr(score, "error", None))

    def _can_ron(self, player: PlayerState, game_state: GameState) -> bool:
        """判断是否可以荣和：必须形完且有役；振听不可荣和。"""
        if not game_state.last_discard:
            return False
        if player.is_furiten:
            return False
        from mahjong_scorer.utils.structures import HandInfo
        hand_info = HandInfo(
            hand_tiles=player.get_all_tiles(),
            open_melds=player.open_melds.copy(),
            winning_tile=game_state.last_discard,
            win_type="RON",
        )
        score = self.scorer.calculate_score(hand_info, self._to_scorer_gamestate(player, game_state))
        return not bool(getattr(score, "error", None))

    def _to_scorer_gamestate(self, player: PlayerState, game_state: GameState):
        from mahjong_scorer.utils.structures import GameState as ScGS
        return ScGS(
            player_wind=player.seat_wind,
            prevalent_wind=game_state.round_wind,
            honba=game_state.honba,
            kyotaku_sticks=game_state.riichi_sticks,
            dora_indicators=game_state.dora_indicators.copy(),
            ura_dora_indicators=game_state.ura_dora_indicators.copy(),
            is_riichi=player.is_riichi,
            is_double_riichi=player.is_double_riichi,
            is_ippatsu=player.is_ippatsu,
        )

    def _can_pon(self, player: PlayerState, game_state: GameState) -> bool:
        """判断是否可以碰"""
        if not game_state.last_discard:
            return False

        # 已经立直了不能碰
        if player.is_riichi:
            return False

        # 手中至少有2张相同的牌
        all_tiles = player.get_all_tiles()
        discard_normalized = normalize_tile(game_state.last_discard)
        count = sum(1 for t in all_tiles if normalize_tile(t) == discard_normalized)

        return count >= 2

    def _can_minkan(self, player: PlayerState, game_state: GameState) -> bool:
        """判断是否可以明杠（大明杠，杠别人打出的牌）"""
        if not game_state.last_discard:
            return False

        # 已经立直了不能明杠
        if player.is_riichi:
            return False

        # 手中有3张相同的牌
        all_tiles = player.get_all_tiles()
        discard_normalized = normalize_tile(game_state.last_discard)
        count = sum(1 for t in all_tiles if normalize_tile(t) == discard_normalized)

        return count >= 3

    def _get_chi_actions(self, player: PlayerState, game_state: GameState) -> List[int]:
        """获取所有可能的吃牌动作"""
        actions = []

        if not game_state.last_discard:
            return actions

        # 已经立直了不能吃
        if player.is_riichi:
            return actions

        # 只能吃数牌
        discard = game_state.last_discard
        if len(discard) != 2 or discard[1] not in ["m", "p", "s"]:
            return actions

        all_tiles = player.get_all_tiles()
        discard_normalized = normalize_tile(discard)
        suit = discard[1]

        try:
            discard_num = int(discard_normalized[0])
        except ValueError:
            return actions

        # 左吃：discarded + 1, + 2 (例如：吃3，需要4和5)
        if discard_num <= 7:
            needed1 = f"{discard_num + 1}{suit}"
            needed2 = f"{discard_num + 2}{suit}"
            if needed1 in all_tiles and needed2 in all_tiles:
                actions.append(ActionEncoder.encode_chi(0))

        # 中吃：discarded - 1, + 1 (例如：吃4，需要3和5)
        if 2 <= discard_num <= 8:
            needed1 = f"{discard_num - 1}{suit}"
            needed2 = f"{discard_num + 1}{suit}"
            if needed1 in all_tiles and needed2 in all_tiles:
                actions.append(ActionEncoder.encode_chi(1))

        # 右吃：discarded - 2, - 1 (例如：吃5，需要3和4)
        if discard_num >= 3:
            needed1 = f"{discard_num - 2}{suit}"
            needed2 = f"{discard_num - 1}{suit}"
            if needed1 in all_tiles and needed2 in all_tiles:
                actions.append(ActionEncoder.encode_chi(2))

        return actions

    def _get_ankan_actions(self, player: PlayerState) -> List[int]:
        """获取所有可能的暗杠动作"""
        actions = []

        # 立直后只能在摸到第4张牌时立即暗杠，且不改变听牌
        if player.is_riichi:
            # 简化处理：立直后不暗杠
            return actions

        all_tiles = player.get_all_tiles()

        # 统计每种牌的数量
        tile_counts = {}
        for tile in all_tiles:
            normalized = normalize_tile(tile)
            tile_counts[normalized] = tile_counts.get(normalized, 0) + 1

        # 如果有4张相同的牌，可以暗杠
        for tile, count in tile_counts.items():
            if count == 4:
                # 暗杠动作用固定的编码（需要进一步指定是哪种牌）
                # 这里简化：使用ANKAN动作
                actions.append(ActionEncoder.ANKAN)
                break  # 一次只能杠一种

        return actions

    def _get_kakan_actions(self, player: PlayerState) -> List[int]:
        """获取所有可能的加杠动作"""
        actions = []

        # 必须有碰过的牌才能加杠
        if not player.open_melds:
            return actions

        all_tiles = player.get_all_tiles()

        # 找到所有碰过的牌
        for meld in player.open_melds:
            if meld.meld_type == "pon":
                # 碰过的牌的第一张
                pon_tile = normalize_tile(meld.tiles[0])
                # 检查手中是否有第4张
                for tile in all_tiles:
                    if normalize_tile(tile) == pon_tile:
                        action = ActionEncoder.encode_kakan(tile)
                        if action != -1 and action not in actions:
                            actions.append(action)
                        break

        return actions

    def _can_kyuushu(self, player: PlayerState, game_state: GameState) -> bool:
        """判断是否可以九种九牌流局"""
        # 必须是第一巡
        if game_state.turn_count != 0:
            return False

        # 无人鸣牌
        if any(len(p.open_melds) > 0 for p in game_state.players):
            return False

        # 统计手中的幺九牌种类
        all_tiles = player.get_all_tiles()
        terminal_honors = set()

        for tile in all_tiles:
            normalized = normalize_tile(tile)
            if self._is_terminal_or_honor(normalized):
                terminal_honors.add(normalized)

        # 至少9种不同的幺九牌
        return len(terminal_honors) >= 9

    def _is_terminal_or_honor(self, tile: str) -> bool:
        """判断是否为幺九牌"""
        if len(tile) != 2:
            return False

        num = tile[0]
        suit = tile[1]

        # 字牌
        if suit == "z":
            return True

        # 老头牌（1和9）
        if num in ["1", "9"]:
            return True

        return False

    def _is_next_player(self, player_id: int, discard_player: int) -> bool:
        """判断是否为下家（只有下家可以吃）"""
        return (discard_player + 1) % 4 == player_id

    def _is_tenpai_after_discard(self, player: PlayerState, discard_tile: str) -> bool:
        """判断打出某张牌后是否听牌"""
        # 临时移除要打出的牌
        temp_hand = player.get_all_tiles()
        if discard_tile in temp_hand:
            temp_hand.remove(discard_tile)
        else:
            return False

        # 使用算点器的听牌判断
        return self.scorer.is_tenpai(temp_hand, player.open_melds)
