# mahjong_scorer/main_scorer.py
from typing import List
from .utils.structures import HandInfo, GameState, ScoreResult, HandAnalysis
from .hand_analyzer import HandAnalyzer
from .yaku_checker import YakuChecker
from .fu_calculator import FuCalculator
from .point_distributor import PointDistributor
from .utils.tile_converter import get_dora_tile


class MainScorer:
    """麻将算点器主类"""

    def __init__(self):
        self.hand_analyzer = HandAnalyzer()

    def calculate_score(
        self, hand_info: HandInfo, game_state: GameState
    ) -> ScoreResult:
        """
        计算和牌得分

        Args:
            hand_info: 手牌信息
            game_state: 游戏状态信息

        Returns:
            ScoreResult: 包含番数、符数、点数等完整信息的结果
        """
        # 步骤 0: 手牌分析
        analysis_result = self.hand_analyzer.analyze_hand(hand_info)

        if not analysis_result.is_complete:
            return ScoreResult(error="手牌不是完整和牌形态")

        # 步骤 1: 役种判断
        yaku_checker = YakuChecker(hand_info, game_state, analysis_result)
        yaku_list = yaku_checker.check_all_yaku()

        if not yaku_list:
            return ScoreResult(error="无役")

        # 计算役的番数
        han = sum(y.han for y in yaku_list)

        # 步骤 2: 宝牌计算
        dora_han = self._calculate_dora(hand_info, game_state)
        total_han = han + dora_han

        # 步骤 3: 符数计算
        fu_calculator = FuCalculator(hand_info, game_state, analysis_result, yaku_list)
        fu = fu_calculator.calculate_fu()

        # 步骤 4 & 5: 基本点和最终结算
        distributor = PointDistributor(total_han, fu, hand_info, game_state)
        final_points = distributor.calculate_final_score()

        return ScoreResult(
            han=total_han,
            fu=fu,
            yaku_list=yaku_list,
            base_points=final_points["base_points"],
            winner_gain=final_points["winner_gain"],
            payments=final_points["payments"],
        )

    def _calculate_dora(self, hand_info: HandInfo, game_state: GameState) -> int:
        """
        计算宝牌数量

        Args:
            hand_info: 手牌信息
            game_state: 游戏状态信息

        Returns:
            int: 宝牌总数（表宝牌 + 里宝牌 + 赤宝牌）
        """
        dora_count = 0

        # 收集所有牌（手牌 + 和牌张 + 副露牌）
        all_tiles = hand_info.hand_tiles + [hand_info.winning_tile]
        for meld in hand_info.open_melds:
            all_tiles.extend(meld.tiles)

        # 表宝牌
        for indicator in game_state.dora_indicators:
            dora_tile = get_dora_tile(indicator)
            dora_count += all_tiles.count(dora_tile)

        # 里宝牌（只有立直才能计算）
        if game_state.is_riichi or game_state.is_double_riichi:
            for indicator in game_state.ura_dora_indicators:
                dora_tile = get_dora_tile(indicator)
                dora_count += all_tiles.count(dora_tile)

        # 赤宝牌（赤5）
        dora_count += all_tiles.count("0m")  # 赤5万
        dora_count += all_tiles.count("0p")  # 赤5筒
        dora_count += all_tiles.count("0s")  # 赤5索

        return dora_count

    def is_tenpai(self, hand_tiles: List[str], open_melds: List = None) -> bool:
        """
        判断是否听牌

        Args:
            hand_tiles: 手牌列表（13张或10张+副露）
            open_melds: 副露列表

        Returns:
            bool: 是否听牌
        """
        if open_melds is None:
            open_melds = []

        # 尝试加入每一种牌，看是否能和
        all_possible_tiles = []
        for i in range(1, 10):
            all_possible_tiles.extend([f"{i}m", f"{i}p", f"{i}s"])
        for i in range(1, 8):
            all_possible_tiles.append(f"{i}z")

        for test_tile in all_possible_tiles:
            test_hand_info = HandInfo(
                hand_tiles=hand_tiles[:],
                open_melds=open_melds,
                winning_tile=test_tile,
            )
            analysis = self.hand_analyzer.analyze_hand(test_hand_info)
            if analysis.is_complete:
                return True

        return False

    def get_waiting_tiles(
        self, hand_tiles: List[str], open_melds: List = None
    ) -> List[str]:
        """
        获取所有听牌的牌

        Args:
            hand_tiles: 手牌列表（13张或10张+副露）
            open_melds: 副露列表

        Returns:
            List[str]: 所有可以和牌的牌
        """
        if open_melds is None:
            open_melds = []

        waiting_tiles = []

        # 尝试加入每一种牌，看是否能和
        all_possible_tiles = []
        for i in range(1, 10):
            all_possible_tiles.extend([f"{i}m", f"{i}p", f"{i}s"])
        for i in range(1, 8):
            all_possible_tiles.append(f"{i}z")

        for test_tile in all_possible_tiles:
            test_hand_info = HandInfo(
                hand_tiles=hand_tiles[:],
                open_melds=open_melds,
                winning_tile=test_tile,
            )
            analysis = self.hand_analyzer.analyze_hand(test_hand_info)
            if analysis.is_complete:
                waiting_tiles.append(test_tile)

        return waiting_tiles


if __name__ == "__main__":
    # 示例用法
    scorer = MainScorer()

    # 示例1: 平和的手牌
    sample_hand_info = HandInfo(
        hand_tiles=[
            "2m",
            "3m",
            "4m",
            "5p",
            "6p",
            "7p",
            "3s",
            "4s",
            "5s",
            "6s",
            "7s",
            "8s",
            "9z",
        ],
        winning_tile="9z",
        win_type="RON",
    )
    sample_game_state = GameState(
        player_wind="south",
        prevalent_wind="east",
        is_riichi=True,
        dora_indicators=["1m"],
    )

    result = scorer.calculate_score(sample_hand_info, sample_game_state)
    print("=" * 50)
    print("算点结果:")
    print(f"番数: {result.han}")
    print(f"符数: {result.fu}")
    print(f"役种:")
    for yaku in result.yaku_list:
        print(f"  - {yaku.name}: {yaku.han}番")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print(f"点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")
    if result.error:
        print(f"错误: {result.error}")
    print("=" * 50)
