# mahjong_scorer/fu_calculator.py
from typing import List
import math
from .utils.structures import HandInfo, GameState, YakuResult, HandAnalysis
from .utils.constants import *


class FuCalculator:
    def __init__(
        self,
        hand_info: HandInfo,
        game_state: GameState,
        analysis: HandAnalysis,
        yaku_list: List[YakuResult],
    ):
        self.hand_info = hand_info
        self.game_state = game_state
        self.analysis = analysis
        self.yaku_list = yaku_list
        self.yaku_names = [y.name for y in yaku_list]

    def calculate_fu(self) -> int:
        """计算符数"""
        # 役满不计算符数
        total_han = sum(y.han for y in self.yaku_list)
        if total_han >= 13:
            return 0

        # 特殊牌型
        if "七对子" in self.yaku_names:
            return 25

        # 平和自摸固定30符
        if "平和" in self.yaku_names and self.hand_info.win_type == "TSUMO":
            return 30

        # 平和荣和固定30符
        if "平和" in self.yaku_names and self.hand_info.win_type == "RON":
            return 30

        # 常规计算
        fu = 20  # 底符

        # 和牌方式加符
        if self.hand_info.win_type == "TSUMO":
            fu += 2  # 自摸加2符
        elif self.analysis.is_menzenchin and self.hand_info.win_type == "RON":
            fu += 10  # 门前清荣和加10符

        # 听牌形态加符
        if self.analysis.wait_type in ["kanchan", "penchan", "tanki"]:
            fu += 2

        # 雀头加符
        fu += self._calculate_pair_fu()

        # 面子加符
        fu += self._calculate_melds_fu()

        # 副露的面子加符
        fu += self._calculate_open_melds_fu()

        # 向上取整到10
        fu = math.ceil(fu / 10) * 10

        # 特殊情况：如果只有20符（没有任何加符），则视为30符
        if fu == 20 and self.hand_info.win_type == "RON":
            fu = 30

        return fu

    def _calculate_pair_fu(self) -> int:
        """计算雀头的符数"""
        if len(self.analysis.pair) == 0:
            return 0

        pair_tile = self.analysis.pair[0]
        fu = 0

        # 三元牌做雀头：2符
        if pair_tile in DRAGONS:
            fu += 2

        # 自风做雀头：2符
        if pair_tile == self.game_state.player_wind_tile:
            fu += 2

        # 场风做雀头：2符
        if pair_tile == self.game_state.prevalent_wind_tile:
            fu += 2

        return fu

    def _calculate_melds_fu(self) -> int:
        """计算手牌中面子的符数"""
        fu = 0

        for meld in self.analysis.decomposed_melds:
            if len(meld) != 3:
                continue

            # 判断是刻子还是顺子
            if meld[0] == meld[1]:  # 刻子
                tile = meld[0]
                is_terminal_or_honor = tile in TERMINALS_AND_HONORS

                # 判断是否是暗刻
                is_ankou = True
                if self.hand_info.win_type == "RON":
                    # 荣和时，如果和牌张在这个刻子中，则是明刻
                    if self.hand_info.winning_tile in meld:
                        is_ankou = False

                # 计算符数
                if is_terminal_or_honor:
                    # 幺九刻
                    if is_ankou:
                        fu += 8  # 暗刻幺九：8符
                    else:
                        fu += 4  # 明刻幺九：4符
                else:
                    # 中张刻
                    if is_ankou:
                        fu += 4  # 暗刻中张：4符
                    else:
                        fu += 2  # 明刻中张：2符

            # 顺子不加符

        return fu

    def _calculate_open_melds_fu(self) -> int:
        """计算副露面子的符数"""
        fu = 0

        for meld in self.hand_info.open_melds:
            if meld.meld_type == "chi":
                # 吃（顺子）不加符
                continue

            tile = meld.tiles[0]
            is_terminal_or_honor = tile in TERMINALS_AND_HONORS

            if meld.meld_type == "pon":
                # 碰（明刻）
                if is_terminal_or_honor:
                    fu += 4  # 明刻幺九：4符
                else:
                    fu += 2  # 明刻中张：2符

            elif meld.meld_type == "ankan":
                # 暗杠
                if is_terminal_or_honor:
                    fu += 32  # 暗杠幺九：32符
                else:
                    fu += 16  # 暗杠中张：16符

            elif meld.meld_type in ["minkan", "kakan"]:
                # 明杠
                if is_terminal_or_honor:
                    fu += 16  # 明杠幺九：16符
                else:
                    fu += 8  # 明杠中张：8符

        return fu