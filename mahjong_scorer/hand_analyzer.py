# mahjong_scorer/hand_analyzer.py
from collections import Counter
from typing import List, Optional

from .utils.structures import HandInfo, HandAnalysis, Tile
from .utils.tile_converter import to_integer, to_string


class HandAnalyzer:
    """
    手牌分析器，用于判断和牌、分解面子和雀头、确定听牌类型。
    内部统一使用34维数组。
    """

    def __init__(self):
        pass

    def analyze_hand(self, hand_info: HandInfo) -> HandAnalysis:
        """
        分析完整和牌手牌的主入口。
        """
        full_hand_str = sorted(hand_info.hand_tiles + [hand_info.winning_tile])

        # 1. 转换为34维数组进行计算
        hand_array = self._to_34_array(full_hand_str)

        # 2. 检查特殊牌型：七对子和国士无双
        if self._is_chitoitsu(hand_array):
            # 七对子比较特殊，没有传统面子
            decomposed_melds = [
                [tile, tile] for tile, count in enumerate(hand_array) if count == 2
            ]
            pair = []  # 七对子没有雀头
            # 七对子一定是单骑听
            wait_type = "tanki"
            return HandAnalysis(
                is_complete=True,
                decomposed_melds=[
                    [to_string(m[0]), to_string(m[1])] for m in decomposed_melds
                ],
                pair=[],
                wait_type=wait_type,
                is_menzenchin=not hand_info.open_melds,
            )

        if self._is_kokushi(hand_array):
            # 国士无双也没有传统面子
            return HandAnalysis(
                is_complete=True,
                decomposed_melds=[],  # 特殊处理
                pair=[],  # 特殊处理
                wait_type="kokushi",  # 特殊听牌
                is_menzenchin=not hand_info.open_melds,
            )

        # 3. 常规牌型分析
        analysis = self._find_regular_hand_composition(
            hand_array, hand_info.winning_tile
        )
        if analysis:
            analysis.is_menzenchin = not hand_info.open_melds
            return analysis

        return HandAnalysis(is_complete=False)

    def _to_34_array(self, hand_str: List[str]) -> List[int]:
        """将字符串手牌列表转换为34维计数数组"""
        array = [0] * 34
        for tile_str in hand_str:
            array[to_integer(tile_str)] += 1
        return array

    def _is_chitoitsu(self, hand_array: List[int]) -> bool:
        """判断是否为七对子"""
        pairs = 0
        for count in hand_array:
            if count == 2:
                pairs += 1
        return pairs == 7

    def _is_kokushi(self, hand_array: List[int]) -> bool:
        """判断是否为国士无双"""
        terminals_and_honors_indices = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]

        has_pair = False
        for i in terminals_and_honors_indices:
            if hand_array[i] == 0:
                return False
            if hand_array[i] == 2:
                has_pair = True

        return has_pair

    def _find_regular_hand_composition(
        self, hand_array: List[int], winning_tile_str: str
    ) -> Optional[HandAnalysis]:
        """
        使用递归寻找常规牌型（4面子1雀头）的组合。
        """
        winning_tile_int = to_integer(winning_tile_str)

        for i in range(34):
            # 尝试将每一种牌作为雀头
            if hand_array[i] >= 2:
                temp_hand = list(hand_array)
                temp_hand[i] -= 2

                # 递归地从剩余的牌中移除面子
                decomposed_melds = self._find_melds(temp_hand)

                if decomposed_melds is not None:
                    # 找到了一个合法的组合
                    pair = [i, i]

                    # 判断听牌类型
                    wait_type = self._determine_wait_type(
                        decomposed_melds, pair, winning_tile_int
                    )

                    return HandAnalysis(
                        is_complete=True,
                        decomposed_melds=[
                            [to_string(t) for t in meld] for meld in decomposed_melds
                        ],
                        pair=[to_string(t) for t in pair],
                        wait_type=wait_type,
                    )
        return None

    def _find_melds(self, hand_array: List[int]) -> Optional[List[List[int]]]:
        """
        递归函数，用于从手牌中分解出所有面子。
        """
        try:
            # 找到第一张有牌的地方开始分解
            first_tile_index = next(
                i for i, count in enumerate(hand_array) if count > 0
            )
        except StopIteration:
            # 如果手牌为空，说明所有牌都成功组成了面子
            return []

        i = first_tile_index

        # 优先尝试移除刻子
        if hand_array[i] >= 3:
            temp_hand = list(hand_array)
            temp_hand[i] -= 3
            result = self._find_melds(temp_hand)
            if result is not None:
                return [[i, i, i]] + result

        # 尝试移除顺子 (字牌不能组成顺子)
        if i <= 26 and (i % 9) <= 6:  # 确保是1-7的数牌
            if hand_array[i] > 0 and hand_array[i + 1] > 0 and hand_array[i + 2] > 0:
                temp_hand = list(hand_array)
                temp_hand[i] -= 1
                temp_hand[i + 1] -= 1
                temp_hand[i + 2] -= 1
                result = self._find_melds(temp_hand)
                if result is not None:
                    return [[i, i + 1, i + 2]] + result

        # 如果以上都失败，说明这个分解路径是死路
        return None

    def _determine_wait_type(
        self, decomposed_melds: List[List[int]], pair: List[int], winning_tile: int
    ) -> str:
        """
        根据分解出的面子和和牌张，判断听牌类型。
        这是一个简化的版本，实际情况可能更复杂。
        """
        # 检查是否是单骑听 (和牌张是雀头)
        if winning_tile == pair[0]:
            return "tanki"

        for meld in decomposed_melds:
            if winning_tile in meld:
                # 刻子听牌 -> 双碰听
                if meld[0] == meld[1]:
                    return "shanpon"

                # 顺子听牌
                else:
                    # 坎张听 (和的是中间的牌)
                    if winning_tile == meld[1]:
                        return "kanchan"
                    # 边张听 (和的是3或7)
                    elif (winning_tile == meld[0] and meld[0] % 9 == 7) or (
                        winning_tile == meld[2] and meld[0] % 9 == 0
                    ):
                        return "penchan"
                    # 两面听
                    else:
                        return "ryammen"
        return ""
