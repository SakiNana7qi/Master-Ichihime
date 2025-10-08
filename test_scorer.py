# test_scorer.py
"""
雀魂立直麻将算点器使用示例
"""

from mahjong_scorer import MainScorer, HandInfo, GameState, Meld


def example_1_pinfu():
    """示例1: 平和 + 立直"""
    print("\n" + "=" * 60)
    print("示例 1: 平和 + 立直")
    print("=" * 60)

    scorer = MainScorer()

    # 手牌：2m3m4m 5p6p7p 3s4s5s 6s7s8s  雀头：2p2p  和牌：2p（两面听）
    hand_info = HandInfo(
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
            "2p",
        ],
        winning_tile="2p",
        win_type="RON",
    )

    game_state = GameState(
        player_wind="south",
        prevalent_wind="east",
        is_riichi=True,
        dora_indicators=["1m"],  # 宝牌指示牌为1m，宝牌为2m
    )

    result = scorer.calculate_score(hand_info, game_state)

    print(f"番数: {result.han}番")
    print(f"符数: {result.fu}符")
    print("役种:")
    for yaku in result.yaku_list:
        print(f"  - {yaku.name}: {yaku.han}番")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print("点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")


def example_2_tanyao():
    """示例2: 断幺九 + 自摸"""
    print("\n" + "=" * 60)
    print("示例 2: 断幺九 + 门前清自摸和")
    print("=" * 60)

    scorer = MainScorer()

    # 手牌：2m3m4m 3p4p5p 4s5s6s 5s6s7s  雀头：8p8p
    hand_info = HandInfo(
        hand_tiles=[
            "2m",
            "3m",
            "4m",
            "3p",
            "4p",
            "5p",
            "4s",
            "5s",
            "6s",
            "5s",
            "6s",
            "7s",
            "8p",
        ],
        winning_tile="8p",
        win_type="TSUMO",
    )

    game_state = GameState(
        player_wind="east",  # 庄家
        prevalent_wind="east",
    )

    result = scorer.calculate_score(hand_info, game_state)

    print(f"番数: {result.han}番")
    print(f"符数: {result.fu}符")
    print("役种:")
    for yaku in result.yaku_list:
        print(f"  - {yaku.name}: {yaku.han}番")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print("点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")


def example_3_toitoi():
    """示例3: 对对和 + 三暗刻"""
    print("\n" + "=" * 60)
    print("示例 3: 对对和 + 三暗刻")
    print("=" * 60)

    scorer = MainScorer()

    # 手牌：2m2m2m 3p3p3p 4s4s4s 5s5s5s  雀头：6z6z
    hand_info = HandInfo(
        hand_tiles=[
            "2m",
            "2m",
            "2m",
            "3p",
            "3p",
            "3p",
            "4s",
            "4s",
            "4s",
            "5s",
            "5s",
            "5s",
            "6z",
        ],
        winning_tile="6z",
        win_type="RON",
    )

    game_state = GameState(
        player_wind="south",
        prevalent_wind="east",
    )

    result = scorer.calculate_score(hand_info, game_state)

    print(f"番数: {result.han}番")
    print(f"符数: {result.fu}符")
    print("役种:")
    for yaku in result.yaku_list:
        print(f"  - {yaku.name}: {yaku.han}番")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print("点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")


def example_4_chinitsu():
    """示例4: 清一色"""
    print("\n" + "=" * 60)
    print("示例 4: 清一色")
    print("=" * 60)

    scorer = MainScorer()

    # 手牌：全是万子 1m2m3m 4m5m6m 7m8m9m 2m3m4m  雀头：5m5m
    hand_info = HandInfo(
        hand_tiles=[
            "1m",
            "2m",
            "3m",
            "4m",
            "5m",
            "6m",
            "7m",
            "8m",
            "9m",
            "2m",
            "3m",
            "4m",
            "5m",
        ],
        winning_tile="5m",
        win_type="TSUMO",
    )

    game_state = GameState(
        player_wind="south",
        prevalent_wind="east",
        is_riichi=True,
    )

    result = scorer.calculate_score(hand_info, game_state)

    print(f"番数: {result.han}番")
    print(f"符数: {result.fu}符")
    print("役种:")
    for yaku in result.yaku_list:
        print(f"  - {yaku.name}: {yaku.han}番")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print("点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")


def example_5_yakuman():
    """示例5: 大三元（役满）"""
    print("\n" + "=" * 60)
    print("示例 5: 大三元（役满）")
    print("=" * 60)

    scorer = MainScorer()

    # 手牌：三元牌刻子 + 其他
    hand_info = HandInfo(
        hand_tiles=[
            "5z",
            "5z",
            "5z",
            "6z",
            "6z",
            "6z",
            "7z",
            "7z",
            "7z",
            "2m",
            "3m",
            "4m",
            "1p",
        ],
        winning_tile="1p",
        win_type="RON",
    )

    game_state = GameState(
        player_wind="east",
        prevalent_wind="east",
    )

    result = scorer.calculate_score(hand_info, game_state)

    print(f"番数: {result.han}番")
    print(f"符数: {result.fu}符")
    print("役种:")
    for yaku in result.yaku_list:
        print(f"  - {yaku.name}: {yaku.han}番")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print("点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")


def example_6_tenpai_check():
    """示例6: 听牌判断"""
    print("\n" + "=" * 60)
    print("示例 6: 听牌判断")
    print("=" * 60)

    scorer = MainScorer()

    # 13张牌，检查是否听牌
    hand_tiles = [
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
        "2p",
    ]

    is_tenpai = scorer.is_tenpai(hand_tiles)
    print(f"手牌: {' '.join(hand_tiles)}")
    print(f"是否听牌: {'是' if is_tenpai else '否'}")

    if is_tenpai:
        waiting_tiles = scorer.get_waiting_tiles(hand_tiles)
        print(f"听牌: {' '.join(waiting_tiles)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("雀魂立直麻将算点器 - 使用示例")
    print("=" * 60)

    example_1_pinfu()
    example_2_tanyao()
    example_3_toitoi()
    example_4_chinitsu()
    example_5_yakuman()
    example_6_tenpai_check()

    print("\n" + "=" * 60)
    print("示例运行完毕")
    print("=" * 60)
