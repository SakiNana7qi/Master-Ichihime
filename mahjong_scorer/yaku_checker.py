# mahjong_scorer/yaku_checker.py
from typing import List
from collections import Counter
from .utils.structures import HandInfo, GameState, YakuResult, HandAnalysis
from .utils.constants import *


class YakuChecker:
    def __init__(
        self, hand_info: HandInfo, game_state: GameState, analysis: HandAnalysis
    ):
        self.hand_info = hand_info
        self.game_state = game_state
        self.analysis = analysis
        self.all_tiles = self.hand_info.hand_tiles + [self.hand_info.winning_tile]

    def check_all_yaku(self) -> List[YakuResult]:
        """检查所有役种"""
        yaku_list = []

        # 首先检查役满
        yakuman_list = self._check_yakuman()
        if yakuman_list:
            return yakuman_list

        # 场景役（特殊条件役）
        yaku_list.extend(self._check_situational_yaku())

        # 牌型役
        yaku_list.extend(self._check_pattern_yaku())

        # 役牌
        yaku_list.extend(self._check_honor_yaku())

        # 处理食下减番
        yaku_list = self._apply_kuisagari(yaku_list)

        # 处理役种复合规则
        yaku_list = self._resolve_conflicts(yaku_list)

        return yaku_list

    def _check_yakuman(self) -> List[YakuResult]:
        """检查役满"""
        yakuman_list = []

        # 天和/地和
        if self.game_state.is_tenhou:
            return [YakuResult(TENHOU, 13)]
        if self.game_state.is_chihou:
            return [YakuResult(CHIIHOU, 13)]

        # 国士无双
        if self._is_kokushi():
            if self._is_kokushi_13():
                return [YakuResult(KOKUSHI_13, 26)]
            else:
                return [YakuResult(KOKUSHI, 13)]

        # 四暗刻
        if self._is_suuankou():
            if self.analysis.wait_type == "tanki":
                return [YakuResult(SUUANKOU_TANKI, 26)]
            else:
                return [YakuResult(SUUANKOU, 13)]

        # 大三元
        if self._is_daisangen():
            yakuman_list.append(YakuResult(DAISANGEN, 13))

        # 四喜和
        if self._is_daisuushii():
            return [YakuResult(DAISUUSHII, 26)]
        if self._is_shousuushii():
            yakuman_list.append(YakuResult(SHOUSUUSHII, 13))

        # 字一色
        if self._is_tsuuiisou():
            yakuman_list.append(YakuResult(TSUUIISOU, 13))

        # 绿一色
        if self._is_ryuuiisou():
            yakuman_list.append(YakuResult(RYUUIISOU, 13))

        # 清老头
        if self._is_chinroutou():
            yakuman_list.append(YakuResult(CHINROUTOU, 13))

        # 四杠子
        if self._is_suukantsu():
            yakuman_list.append(YakuResult(SUUKANTSU, 13))

        # 九莲宝灯
        if self._is_chuurenpoutou():
            if self._is_chuurenpoutou_9():
                return [YakuResult(CHUURENPOUTOU_9, 26)]
            else:
                return [YakuResult(CHUURENPOUTOU, 13)]

        return yakuman_list

    def _check_situational_yaku(self) -> List[YakuResult]:
        """检查场景役（依赖特殊条件）"""
        yaku_list = []

        # 立直/两立直
        if self.game_state.is_double_riichi:
            yaku_list.append(YakuResult(DOUBLE_RIICHI, 2))
        elif self.game_state.is_riichi:
            yaku_list.append(YakuResult(RIICHI, 1))

        # 一发
        if self.game_state.is_ippatsu:
            yaku_list.append(YakuResult(IPPATSU, 1))

        # 门前清自摸和
        if self.analysis.is_menzenchin and self.hand_info.win_type == "TSUMO":
            yaku_list.append(YakuResult(TSUMO, 1))

        # 海底捞月
        if self.game_state.is_haitei:
            yaku_list.append(YakuResult(HAITEI, 1))

        # 河底捞鱼
        if self.game_state.is_houtei:
            yaku_list.append(YakuResult(HOUTEI, 1))

        # 岭上开花
        if self.game_state.is_rinshan:
            yaku_list.append(YakuResult(RINSHAN, 1))

        # 抢杠
        if self.game_state.is_chankan:
            yaku_list.append(YakuResult(CHANKAN, 1))

        return yaku_list

    def _check_pattern_yaku(self) -> List[YakuResult]:
        """检查牌型役"""
        yaku_list = []

        # 七对子
        if self._is_chitoitsu():
            yaku_list.append(YakuResult(CHIITOITSU, 2))
            return yaku_list  # 七对子不能和其他牌型役复合

        # 平和
        if self._is_pinfu():
            yaku_list.append(YakuResult(PINFU, 1))

        # 断幺九
        if self._is_tanyao():
            yaku_list.append(YakuResult(TANYAO, 1))

        # 一杯口
        if self._is_iipeikou():
            yaku_list.append(YakuResult(IIPEIKOU, 1))

        # 两杯口
        if self._is_ryanpeikou():
            yaku_list.append(YakuResult(RYANPEIKOU, 3))

        # 三色同顺
        if self._is_sanshoku_doujun():
            yaku_list.append(YakuResult(SANSHOKU_DOUJUN, 2))

        # 三色同刻
        if self._is_sanshoku_doukou():
            yaku_list.append(YakuResult(SANSHOKU_DOUKOU, 2))

        # 一气通贯
        if self._is_ittsu():
            yaku_list.append(YakuResult(ITTSU, 2))

        # 混全带幺九
        if self._is_chanta():
            yaku_list.append(YakuResult(CHANTA, 2))

        # 纯全带幺九
        if self._is_junchan():
            yaku_list.append(YakuResult(JUNCHAN, 3))

        # 对对和
        if self._is_toitoi():
            yaku_list.append(YakuResult(TOITOI, 2))

        # 三暗刻
        if self._is_sanankou():
            yaku_list.append(YakuResult(SANANKOU, 2))

        # 三杠子
        if self._is_sankantsu():
            yaku_list.append(YakuResult(SANKANTSU, 2))

        # 混老头
        if self._is_honroutou():
            yaku_list.append(YakuResult(HONROUTOU, 2))

        # 混一色
        if self._is_honitsu():
            yaku_list.append(YakuResult(HONITSU, 3))

        # 清一色
        if self._is_chinitsu():
            yaku_list.append(YakuResult(CHINITSU, 6))

        return yaku_list

    def _check_honor_yaku(self) -> List[YakuResult]:
        """检查役牌"""
        yaku_list = []

        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        tile_counts = Counter(all_tiles_in_hand)

        # 三元牌
        if tile_counts.get(DRAGON_WHITE, 0) >= 3:
            yaku_list.append(YakuResult(HAKU, 1))
        if tile_counts.get(DRAGON_GREEN, 0) >= 3:
            yaku_list.append(YakuResult(HATSU, 1))
        if tile_counts.get(DRAGON_RED, 0) >= 3:
            yaku_list.append(YakuResult(CHUN, 1))

        # 自风
        player_wind_tile = self.game_state.player_wind_tile
        if tile_counts.get(player_wind_tile, 0) >= 3:
            wind_name = WIND_NAMES[self.game_state.player_wind]
            yaku_list.append(YakuResult(f"自风牌 {wind_name}", 1))

        # 场风
        prevalent_wind_tile = self.game_state.prevalent_wind_tile
        if (
            prevalent_wind_tile != player_wind_tile
            and tile_counts.get(prevalent_wind_tile, 0) >= 3
        ):
            wind_name = WIND_NAMES[self.game_state.prevalent_wind]
            yaku_list.append(YakuResult(f"场风牌 {wind_name}", 1))

        # 小三元
        if self._is_shousangen():
            yaku_list.append(YakuResult(SHOUSANGEN, 2))

        return yaku_list

    # ===== 役种判断函数 =====

    def _is_pinfu(self) -> bool:
        """判断平和"""
        if not self.analysis.is_menzenchin:
            return False
        if self.hand_info.win_type != "RON":
            # 平和必须荣和（雀魂规则）
            return False

        # 必须是4顺子1雀头
        if len(self.analysis.decomposed_melds) != 4:
            return False
        for meld in self.analysis.decomposed_melds:
            if meld[0] == meld[1]:  # 是刻子
                return False

        # 雀头不能是役牌
        if len(self.analysis.pair) < 1:
            return False
        pair_tile = self.analysis.pair[0]
        player_wind_tile = self.game_state.player_wind_tile
        prevalent_wind_tile = self.game_state.prevalent_wind_tile
        if pair_tile in [player_wind_tile, prevalent_wind_tile, "5z", "6z", "7z"]:
            return False

        # 必须是两面听
        if self.analysis.wait_type != "ryammen":
            return False

        return True

    def _is_tanyao(self) -> bool:
        """判断断幺九"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        for tile in all_tiles_in_hand:
            if tile in TERMINALS_AND_HONORS:
                return False
        return True

    def _is_chitoitsu(self) -> bool:
        """判断七对子"""
        return len(self.analysis.decomposed_melds) == 7 and all(
            len(meld) == 2 for meld in self.analysis.decomposed_melds
        )

    def _is_iipeikou(self) -> bool:
        """判断一杯口"""
        if not self.analysis.is_menzenchin:
            return False

        # 找出所有顺子
        sequences = []
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] != meld[1]:  # 顺子
                sequences.append(tuple(sorted(meld)))

        # 检查是否有重复
        seq_counts = Counter(sequences)
        return any(count >= 2 for count in seq_counts.values())

    def _is_ryanpeikou(self) -> bool:
        """判断两杯口"""
        if not self.analysis.is_menzenchin:
            return False

        # 找出所有顺子
        sequences = []
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] != meld[1]:  # 顺子
                sequences.append(tuple(sorted(meld)))

        # 检查是否有两组重复
        seq_counts = Counter(sequences)
        pairs = sum(1 for count in seq_counts.values() if count == 2)
        return pairs == 2

    def _is_sanshoku_doujun(self) -> bool:
        """判断三色同顺"""
        # 找出所有顺子的数字模式
        sequences = {}  # {数字模式: [花色]}
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] != meld[1]:  # 顺子
                # 提取数字模式（如123）
                tile1 = meld[0]
                if tile1[-1] not in ["m", "p", "s"]:  # 字牌不能组成顺子
                    continue
                num1 = int(tile1[0])
                suit = tile1[1]
                pattern = (num1, num1 + 1, num1 + 2)
                if pattern not in sequences:
                    sequences[pattern] = []
                sequences[pattern].append(suit)

        # 检查是否有同样的数字模式在三种花色中
        for pattern, suits in sequences.items():
            if len(set(suits)) == 3:
                return True
        return False

    def _is_sanshoku_doukou(self) -> bool:
        """判断三色同刻"""
        # 找出所有刻子的数字
        triplets = {}  # {数字: [花色]}
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] == meld[1]:  # 刻子
                tile = meld[0]
                if tile[-1] not in ["m", "p", "s"]:  # 字牌排除
                    continue
                num = int(tile[0])
                suit = tile[1]
                if num not in triplets:
                    triplets[num] = []
                triplets[num].append(suit)

        # 同样考虑副露的刻子
        for meld in self.hand_info.open_melds:
            if meld.meld_type in ["pon", "minkan", "ankan", "kakan"]:
                tile = meld.tiles[0]
                if tile[-1] not in ["m", "p", "s"]:
                    continue
                num = int(tile[0])
                suit = tile[1]
                if num not in triplets:
                    triplets[num] = []
                triplets[num].append(suit)

        # 检查是否有同样的数字在三种花色中
        for num, suits in triplets.items():
            if len(set(suits)) == 3:
                return True
        return False

    def _is_ittsu(self) -> bool:
        """判断一气通贯"""
        # 找出所有顺子
        sequences = []
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] != meld[1]:  # 顺子
                tile = meld[0]
                if tile[-1] not in ["m", "p", "s"]:
                    continue
                num = int(tile[0])
                suit = tile[1]
                sequences.append((num, suit))

        # 同样考虑副露的顺子
        for meld in self.hand_info.open_melds:
            if meld.meld_type == "chi":
                tile = sorted(meld.tiles)[0]
                if tile[-1] not in ["m", "p", "s"]:
                    continue
                num = int(tile[0])
                suit = tile[1]
                sequences.append((num, suit))

        # 检查是否有123, 456, 789同花色
        for suit in ["m", "p", "s"]:
            if (
                (1, suit) in sequences
                and (4, suit) in sequences
                and (7, suit) in sequences
            ):
                return True
        return False

    def _is_chanta(self) -> bool:
        """判断混全带幺九"""
        has_honor = False

        # 检查雀头
        if len(self.analysis.pair) > 0:
            pair_tile = self.analysis.pair[0]
            if pair_tile not in TERMINALS_AND_HONORS:
                return False
            if pair_tile in HONORS:
                has_honor = True

        # 检查所有面子
        for meld in self.analysis.decomposed_melds:
            meld_has_terminal_or_honor = False
            for tile in meld:
                if tile in TERMINALS_AND_HONORS:
                    meld_has_terminal_or_honor = True
                    if tile in HONORS:
                        has_honor = True
            if not meld_has_terminal_or_honor:
                return False

        # 必须含有字牌
        return has_honor

    def _is_junchan(self) -> bool:
        """判断纯全带幺九"""
        # 检查雀头
        if len(self.analysis.pair) > 0:
            pair_tile = self.analysis.pair[0]
            if pair_tile not in TERMINALS:
                return False

        # 检查所有面子
        for meld in self.analysis.decomposed_melds:
            meld_has_terminal = False
            for tile in meld:
                if tile in TERMINALS:
                    meld_has_terminal = True
                elif tile in HONORS:
                    return False  # 纯全带不能有字牌
            if not meld_has_terminal:
                return False

        return True

    def _is_toitoi(self) -> bool:
        """判断对对和"""
        # 所有面子都是刻子
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] != meld[1]:  # 是顺子
                return False
        return True

    def _is_sanankou(self) -> bool:
        """判断三暗刻"""
        ankou_count = 0

        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] == meld[1]:  # 刻子
                # 判断是否是暗刻
                if self.hand_info.win_type == "TSUMO":
                    ankou_count += 1
                else:
                    # 荣和时，如果和牌张不在这个刻子中，则是暗刻
                    if self.hand_info.winning_tile not in meld:
                        ankou_count += 1

        return ankou_count == 3

    def _is_sankantsu(self) -> bool:
        """判断三杠子"""
        kantsu_count = sum(
            1
            for meld in self.hand_info.open_melds
            if meld.meld_type in ["minkan", "ankan", "kakan"]
        )
        return kantsu_count == 3

    def _is_honroutou(self) -> bool:
        """判断混老头"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        # 所有牌都必须是幺九牌
        for tile in all_tiles_in_hand:
            if tile not in TERMINALS_AND_HONORS:
                return False

        # 必须同时包含字牌和老头牌
        has_honors = any(tile in HONORS for tile in all_tiles_in_hand)
        has_terminals = any(tile in TERMINALS for tile in all_tiles_in_hand)

        return has_honors and has_terminals

    def _is_shousangen(self) -> bool:
        """判断小三元"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        tile_counts = Counter(all_tiles_in_hand)

        # 三元牌中有2个刻子和1个雀头
        dragon_triplets = sum(
            1 for dragon in DRAGONS if tile_counts.get(dragon, 0) >= 3
        )
        dragon_pairs = sum(1 for dragon in DRAGONS if tile_counts.get(dragon, 0) == 2)

        return dragon_triplets == 2 and dragon_pairs == 1

    def _is_honitsu(self) -> bool:
        """判断混一色"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        # 统计花色
        suits = set()
        has_honors = False
        for tile in all_tiles_in_hand:
            if tile[-1] == "z":
                has_honors = True
            else:
                suits.add(tile[-1])

        # 只有一种数牌花色+字牌
        return len(suits) == 1 and has_honors

    def _is_chinitsu(self) -> bool:
        """判断清一色"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        # 统计花色
        suits = set()
        for tile in all_tiles_in_hand:
            if tile[-1] == "z":
                return False  # 有字牌就不是清一色
            suits.add(tile[-1])

        # 只有一种数牌花色
        return len(suits) == 1

    # ===== 役满判断函数 =====

    def _is_kokushi(self) -> bool:
        """判断国士无双"""
        return self.analysis.wait_type == "kokushi"

    def _is_kokushi_13(self) -> bool:
        """判断国士无双十三面"""
        if not self._is_kokushi():
            return False
        # 13种幺九牌各一张
        tile_counts = Counter(self.all_tiles)
        for tile in TERMINALS_AND_HONORS:
            if tile_counts.get(tile, 0) != 1:
                return False
        return True

    def _is_suuankou(self) -> bool:
        """判断四暗刻"""
        if not self.analysis.is_menzenchin:
            return False

        ankou_count = 0
        for meld in self.analysis.decomposed_melds:
            if len(meld) == 3 and meld[0] == meld[1]:  # 刻子
                if self.hand_info.win_type == "TSUMO":
                    ankou_count += 1
                else:
                    # 荣和时，如果和牌张不在这个刻子中，则是暗刻
                    if self.hand_info.winning_tile not in meld:
                        ankou_count += 1

        return ankou_count == 4

    def _is_daisangen(self) -> bool:
        """判断大三元"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        tile_counts = Counter(all_tiles_in_hand)

        # 三元牌都是刻子
        return all(tile_counts.get(dragon, 0) >= 3 for dragon in DRAGONS)

    def _is_shousuushii(self) -> bool:
        """判断小四喜"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        tile_counts = Counter(all_tiles_in_hand)

        # 风牌中有3个刻子和1个雀头
        wind_triplets = sum(1 for wind in WINDS if tile_counts.get(wind, 0) >= 3)
        wind_pairs = sum(1 for wind in WINDS if tile_counts.get(wind, 0) == 2)

        return wind_triplets == 3 and wind_pairs == 1

    def _is_daisuushii(self) -> bool:
        """判断大四喜"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        tile_counts = Counter(all_tiles_in_hand)

        # 风牌都是刻子
        return all(tile_counts.get(wind, 0) >= 3 for wind in WINDS)

    def _is_tsuuiisou(self) -> bool:
        """判断字一色"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        # 所有牌都是字牌
        return all(tile in HONORS for tile in all_tiles_in_hand)

    def _is_ryuuiisou(self) -> bool:
        """判断绿一色"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        # 所有牌都是绿牌
        return all(tile in GREEN_TILES for tile in all_tiles_in_hand)

    def _is_chinroutou(self) -> bool:
        """判断清老头"""
        all_tiles_in_hand = self.all_tiles[:]
        for meld in self.hand_info.open_melds:
            all_tiles_in_hand.extend(meld.tiles)

        # 所有牌都是老头牌（1和9）
        return all(tile in TERMINALS for tile in all_tiles_in_hand)

    def _is_suukantsu(self) -> bool:
        """判断四杠子"""
        kantsu_count = sum(
            1
            for meld in self.hand_info.open_melds
            if meld.meld_type in ["minkan", "ankan", "kakan"]
        )
        return kantsu_count == 4

    def _is_chuurenpoutou(self) -> bool:
        """判断九莲宝灯"""
        if not self.analysis.is_menzenchin:
            return False
        if not self._is_chinitsu():
            return False

        # 统计牌的数量
        tile_counts = Counter(self.all_tiles)

        # 获取花色
        suit = self.all_tiles[0][-1]
        if suit == "z":
            return False

        # 必须是1112345678999的形态
        expected = {
            f"1{suit}": 3,
            f"2{suit}": 1,
            f"3{suit}": 1,
            f"4{suit}": 1,
            f"5{suit}": 1,
            f"6{suit}": 1,
            f"7{suit}": 1,
            f"8{suit}": 1,
            f"9{suit}": 3,
        }

        # 实际牌型应该是1112345678999中的一张多余
        # 检查基础形态
        for tile, count in expected.items():
            actual_count = tile_counts.get(tile, 0)
            if actual_count < count:
                return False

        return True

    def _is_chuurenpoutou_9(self) -> bool:
        """判断纯正九莲宝灯（九面听）"""
        if not self._is_chuurenpoutou():
            return False

        # 除了和牌张之外，必须是1112345678999
        hand_without_winning = self.hand_info.hand_tiles[:]
        tile_counts = Counter(hand_without_winning)

        suit = self.all_tiles[0][-1]
        expected = {
            f"1{suit}": 3,
            f"2{suit}": 1,
            f"3{suit}": 1,
            f"4{suit}": 1,
            f"5{suit}": 1,
            f"6{suit}": 1,
            f"7{suit}": 1,
            f"8{suit}": 1,
            f"9{suit}": 3,
        }

        return tile_counts == expected

    # ===== 辅助函数 =====

    def _apply_kuisagari(self, yaku_list: List[YakuResult]) -> List[YakuResult]:
        """应用食下减番规则"""
        if self.analysis.is_menzenchin:
            return yaku_list

        result = []
        for yaku in yaku_list:
            if yaku.name in KUISAGARI_YAKU:
                # 减少番数
                new_han = KUISAGARI_YAKU[yaku.name]
                result.append(YakuResult(yaku.name, new_han))
            else:
                result.append(yaku)
        return result

    def _resolve_conflicts(self, yaku_list: List[YakuResult]) -> List[YakuResult]:
        """处理役种复合规则"""
        yaku_names = [y.name for y in yaku_list]

        # 处理互相排斥的役种
        for yaku1, yaku2 in MUTUALLY_EXCLUSIVE_YAKU:
            if yaku1 in yaku_names and yaku2 in yaku_names:
                # 保留番数较高的
                han1 = YAKU_HAN.get(yaku1, 0)
                han2 = YAKU_HAN.get(yaku2, 0)
                if han1 > han2:
                    yaku_list = [y for y in yaku_list if y.name != yaku2]
                else:
                    yaku_list = [y for y in yaku_list if y.name != yaku1]

        return yaku_list
