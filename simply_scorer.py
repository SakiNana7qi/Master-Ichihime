# simply_scorer.py
"""
简单交互式麻将算点器
"""

from mahjong_scorer import MainScorer, HandInfo, GameState


def parse_short_hand(short_hand: str) -> list:
    """
    解析简短格式的手牌
    例如: "466m1p107s122567z" -> ["4m", "6m", "6m", "1p", "1s", "0s", "7s", "1z", "2z", "2z", "5z", "6z", "7z"]
    """
    tiles = []
    current_numbers = []

    for char in short_hand:
        if char.isdigit():
            current_numbers.append(char)
        elif char in ["m", "p", "s", "z"]:
            # 遇到花色，将之前收集的数字转换为牌
            for num in current_numbers:
                tiles.append(f"{num}{char}")
            current_numbers = []

    return tiles


def main():
    print("=" * 60)
    print("雀魂立直麻将算点器 - 交互版")
    print("=" * 60)
    print()

    scorer = MainScorer()

    while True:
        print("\n请输入手牌信息（或输入 'q' 退出）:")
        print("格式说明: 466m1p107s122567z")
        print("  数字后跟花色: m=万, p=筒, s=索, z=字牌")
        print("  0表示赤宝牌: 0m=赤5万, 0p=赤5筒, 0s=赤5索")
        print("  字牌: 1z=东, 2z=南, 3z=西, 4z=北, 5z=白, 6z=发, 7z=中")
        print()

        # 读取手牌
        hand_input = input("手牌 (13张): ").strip()
        if hand_input.lower() == "q":
            print("退出程序。")
            break

        try:
            hand_tiles = parse_short_hand(hand_input)
            if len(hand_tiles) != 13:
                print(f"错误: 手牌应该是13张，您输入了{len(hand_tiles)}张")
                continue
            print(f"解析后的手牌: {' '.join(hand_tiles)}")
        except Exception as e:
            print(f"手牌格式错误: {e}")
            continue

        # 读取和牌张
        winning_input = input("和牌张 (如 2m): ").strip()
        if not winning_input:
            print("错误: 和牌张不能为空")
            continue
        print(f"和牌张: {winning_input}")

        # 读取和牌方式
        win_type_input = input("和牌方式 (RON/TSUMO, 默认RON): ").strip().upper()
        if not win_type_input:
            win_type_input = "RON"
        if win_type_input not in ["RON", "TSUMO"]:
            print("错误: 和牌方式只能是 RON 或 TSUMO")
            continue

        # 读取游戏状态
        print("\n游戏状态:")
        player_wind = input("自风 (east/south/west/north, 默认east): ").strip().lower()
        if not player_wind:
            player_wind = "east"
        if player_wind not in ["east", "south", "west", "north"]:
            print("错误: 自风必须是 east/south/west/north")
            continue

        prevalent_wind = (
            input("场风 (east/south/west/north, 默认east): ").strip().lower()
        )
        if not prevalent_wind:
            prevalent_wind = "east"
        if prevalent_wind not in ["east", "south", "west", "north"]:
            print("错误: 场风必须是 east/south/west/north")
            continue

        is_riichi_input = input("是否立直? (y/n, 默认n): ").strip().lower()
        is_riichi = is_riichi_input == "y"

        dora_input = input("宝牌指示牌 (用空格分隔，如: 1m 5p, 默认无): ").strip()
        dora_indicators = dora_input.split() if dora_input else []

        # 创建手牌和游戏状态对象
        hand_info = HandInfo(
            hand_tiles=hand_tiles,
            winning_tile=winning_input,
            win_type=win_type_input,
        )

        game_state = GameState(
            player_wind=player_wind,
            prevalent_wind=prevalent_wind,
            is_riichi=is_riichi,
            dora_indicators=dora_indicators,
        )

        # 计算得分
        print("\n" + "=" * 60)
        print("计算结果:")
        print("=" * 60)

        try:
            result = scorer.calculate_score(hand_info, game_state)

            if result.error:
                print(f"错误: {result.error}")
            else:
                # 计算役的番数和宝牌番数
                yaku_han = sum(y.han for y in result.yaku_list)
                dora_han = result.han - yaku_han

                print(f"\n总番数: {result.han}番")
                print(f"符数: {result.fu}符")

                print("\n成立的役种:")
                if result.yaku_list:
                    for yaku in result.yaku_list:
                        print(f"  - {yaku.name}: {yaku.han}番")
                    print(f"  役种合计: {yaku_han}番")
                else:
                    print("  无役")

                if dora_han > 0:
                    print(f"\n宝牌:")
                    print(f"  - 宝牌: {dora_han}番")

                print(f"\n基本点: {result.base_points}")

                # 显示当前情况的点数
                is_dealer = game_state.is_dealer
                print(f"\n当前情况 ({'庄家' if is_dealer else '闲家'}):")
                print(f"  和牌者获得: {result.winner_gain}点")
                print(f"  点数支付:")
                for payment in result.payments:
                    print(f"    {payment['from']}: {payment['amount']}点")

                # 额外显示另一种情况的点数（庄/闲对比）
                import math

                def ceil_to_hundred(points):
                    return math.ceil(points / 100) * 100

                print(f"\n参考点数:")
                if win_type_input == "RON":
                    # 荣和
                    dealer_payment = ceil_to_hundred(result.base_points * 6)
                    non_dealer_payment = ceil_to_hundred(result.base_points * 4)
                    print(f"  庄家荣和: {dealer_payment}点 (放炮者支付)")
                    print(f"  闲家荣和: {non_dealer_payment}点 (放炮者支付)")
                else:
                    # 自摸
                    dealer_each = ceil_to_hundred(result.base_points * 2)
                    dealer_total = dealer_each * 3
                    non_dealer_from_dealer = ceil_to_hundred(result.base_points * 2)
                    non_dealer_from_non = ceil_to_hundred(result.base_points * 1)
                    non_dealer_total = non_dealer_from_dealer + non_dealer_from_non * 2
                    print(f"  庄家自摸: {dealer_total}点 (每家{dealer_each}点)")
                    print(
                        f"  闲家自摸: {non_dealer_total}点 (庄家{non_dealer_from_dealer}点, 闲家各{non_dealer_from_non}点)"
                    )
        except Exception as e:
            print(f"计算错误: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "=" * 60)

        # 询问是否继续
        continue_input = input("\n是否继续计算? (y/n, 默认y): ").strip().lower()
        if continue_input == "n":
            print("退出程序。")
            break


if __name__ == "__main__":
    main()
