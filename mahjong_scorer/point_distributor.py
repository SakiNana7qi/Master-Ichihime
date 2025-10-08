# mahjong_scorer/point_distributor.py
import math
from .utils.structures import HandInfo, GameState


class PointDistributor:
    def __init__(self, han: int, fu: int, hand_info: HandInfo, game_state: GameState):
        self.han = han
        self.fu = fu
        self.hand_info = hand_info
        self.game_state = game_state
        self.is_dealer = self.game_state.is_dealer

    def calculate_final_score(self) -> dict:
        """计算最终点数分配"""
        base_points = self._calculate_base_points()

        # 本场棒和立直棒
        honba_bonus = self.game_state.honba * 300
        riichi_stick_bonus = self.game_state.kyotaku_sticks * 1000

        payments = []
        total_payment = 0

        if self.hand_info.win_type == "RON":
            # 荣和：放炮者支付
            if self.is_dealer:
                # 庄家荣和：放炮者支付 base_points * 6
                payment = self._ceil_to_hundred(base_points * 6)
            else:
                # 闲家荣和：放炮者支付 base_points * 4
                payment = self._ceil_to_hundred(base_points * 4)

            payment += honba_bonus  # 加上本场费
            payments.append({"from": "放炮者", "amount": payment})
            total_payment = payment

        elif self.hand_info.win_type == "TSUMO":
            # 自摸：其他三家分担
            if self.is_dealer:
                # 庄家自摸：每个闲家支付 base_points * 2
                each_payment = self._ceil_to_hundred(base_points * 2)
                each_payment += honba_bonus // 3  # 本场费均摊（向下取整）
                for i in range(3):
                    payments.append({"from": f"闲家{i+1}", "amount": each_payment})
                total_payment = each_payment * 3
            else:
                # 闲家自摸
                # 庄家支付 base_points * 2
                dealer_payment = self._ceil_to_hundred(base_points * 2)
                dealer_payment += honba_bonus // 3
                payments.append({"from": "庄家", "amount": dealer_payment})

                # 其他闲家各支付 base_points * 1
                other_payment = self._ceil_to_hundred(base_points * 1)
                other_payment += honba_bonus // 3
                for i in range(2):
                    payments.append({"from": f"闲家{i+1}", "amount": other_payment})

                total_payment = dealer_payment + other_payment * 2

        # 和牌者获得的总点数
        winner_gain = total_payment + riichi_stick_bonus

        return {
            "base_points": base_points,
            "winner_gain": winner_gain,
            "payments": payments,
        }

    def _calculate_base_points(self) -> int:
        """计算基本点"""
        # 役满及以上
        if self.han >= 13:
            yakuman_count = self.han // 13
            return 8000 * yakuman_count

        # 累计役满
        if self.han >= 11:
            return 6000  # 三倍满
        if self.han >= 8:
            return 4000  # 倍满
        if self.han >= 6:
            return 3000  # 跳满
        if self.han >= 5:
            return 2000  # 满贯
        if self.han >= 4:
            # 4番根据符数判断
            if self.fu >= 40:
                return 2000  # 满贯
            elif self.fu >= 30:
                base = self.fu * (2 ** (self.han + 2))
                return min(base, 2000)
        if self.han >= 3:
            # 3番根据符数判断
            if self.fu >= 70:
                return 2000  # 满贯
            base = self.fu * (2 ** (self.han + 2))
            return min(base, 2000)

        # 常规计算
        base_points = self.fu * (2 ** (self.han + 2))

        # 满贯封顶
        return min(base_points, 2000)

    def _ceil_to_hundred(self, points: int) -> int:
        """向上取整到百位"""
        return math.ceil(points / 100) * 100
