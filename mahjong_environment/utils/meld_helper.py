# mahjong_environment/utils/meld_helper.py
"""
副露（鸣牌）辅助工具
处理吃、碰、杠的具体逻辑
"""

from typing import List, Tuple, Optional
from mahjong_scorer.utils.structures import Meld
from .tile_utils import normalize_tile


class MeldHelper:
    """副露辅助类"""

    @staticmethod
    def can_chi(hand_tiles: List[str], discard_tile: str) -> List[Tuple[str, str]]:
        """
        检查是否可以吃牌，并返回所有可能的吃牌组合

        Args:
            hand_tiles: 手牌列表
            discard_tile: 被打出的牌

        Returns:
            List[Tuple[str, str]]: 可以吃的组合列表，每个元组包含需要的两张牌
        """
        combinations = []

        # 只能吃数牌
        if len(discard_tile) != 2 or discard_tile[1] not in ["m", "p", "s"]:
            return combinations

        discard_normalized = normalize_tile(discard_tile)
        suit = discard_tile[1]

        try:
            discard_num = int(discard_normalized[0])
        except ValueError:
            return combinations

        # 左吃：discard + 1 + 2 (例如：吃3，需要4和5)
        if discard_num <= 7:
            tile1 = f"{discard_num + 1}{suit}"
            tile2 = f"{discard_num + 2}{suit}"
            if tile1 in hand_tiles and tile2 in hand_tiles:
                combinations.append((tile1, tile2))

        # 中吃：discard - 1 + 1 (例如：吃4，需要3和5)
        if 2 <= discard_num <= 8:
            tile1 = f"{discard_num - 1}{suit}"
            tile2 = f"{discard_num + 1}{suit}"
            if tile1 in hand_tiles and tile2 in hand_tiles:
                combinations.append((tile1, tile2))

        # 右吃：discard - 2 - 1 (例如：吃5，需要3和4)
        if discard_num >= 3:
            tile1 = f"{discard_num - 2}{suit}"
            tile2 = f"{discard_num - 1}{suit}"
            if tile1 in hand_tiles and tile2 in hand_tiles:
                combinations.append((tile1, tile2))

        return combinations

    @staticmethod
    def create_chi_meld(
        chi_type: str, hand_tiles: List[str], discard_tile: str, from_player: int
    ) -> Optional[Tuple[Meld, List[str]]]:
        """
        创建吃的副露

        Args:
            chi_type: 吃的类型 ('left', 'middle', 'right')
            hand_tiles: 手牌列表
            discard_tile: 被吃的牌
            from_player: 打出这张牌的玩家

        Returns:
            Optional[Tuple[Meld, List[str]]]: (副露对象, 需要从手牌移除的牌)
        """
        combinations = MeldHelper.can_chi(hand_tiles, discard_tile)

        if not combinations:
            return None

        # 根据类型选择组合
        type_map = {"left": 0, "middle": 1, "right": 2}
        idx = type_map.get(chi_type, 0)

        if idx >= len(combinations):
            return None

        tile1, tile2 = combinations[idx]

        # 创建副露（按数字顺序排列）
        meld_tiles = sorted([discard_tile, tile1, tile2])
        meld = Meld(meld_type="chi", tiles=meld_tiles, from_who=from_player)

        return meld, [tile1, tile2]

    @staticmethod
    def can_pon(hand_tiles: List[str], discard_tile: str) -> bool:
        """
        检查是否可以碰

        Args:
            hand_tiles: 手牌列表
            discard_tile: 被打出的牌

        Returns:
            bool: 是否可以碰
        """
        discard_normalized = normalize_tile(discard_tile)
        count = sum(1 for t in hand_tiles if normalize_tile(t) == discard_normalized)
        return count >= 2

    @staticmethod
    def create_pon_meld(
        hand_tiles: List[str], discard_tile: str, from_player: int
    ) -> Optional[Tuple[Meld, List[str]]]:
        """
        创建碰的副露

        Args:
            hand_tiles: 手牌列表
            discard_tile: 被碰的牌
            from_player: 打出这张牌的玩家

        Returns:
            Optional[Tuple[Meld, List[str]]]: (副露对象, 需要从手牌移除的牌)
        """
        if not MeldHelper.can_pon(hand_tiles, discard_tile):
            return None

        discard_normalized = normalize_tile(discard_tile)

        # 找到手中的两张相同的牌
        matching_tiles = [
            t for t in hand_tiles if normalize_tile(t) == discard_normalized
        ][:2]

        if len(matching_tiles) < 2:
            return None

        # 创建副露
        meld_tiles = [discard_tile] + matching_tiles
        meld = Meld(meld_type="pon", tiles=meld_tiles, from_who=from_player)

        return meld, matching_tiles

    @staticmethod
    def can_minkan(hand_tiles: List[str], discard_tile: str) -> bool:
        """
        检查是否可以明杠

        Args:
            hand_tiles: 手牌列表
            discard_tile: 被打出的牌

        Returns:
            bool: 是否可以明杠
        """
        discard_normalized = normalize_tile(discard_tile)
        count = sum(1 for t in hand_tiles if normalize_tile(t) == discard_normalized)
        return count >= 3

    @staticmethod
    def create_minkan_meld(
        hand_tiles: List[str], discard_tile: str, from_player: int
    ) -> Optional[Tuple[Meld, List[str]]]:
        """
        创建明杠的副露

        Args:
            hand_tiles: 手牌列表
            discard_tile: 被杠的牌
            from_player: 打出这张牌的玩家

        Returns:
            Optional[Tuple[Meld, List[str]]]: (副露对象, 需要从手牌移除的牌)
        """
        if not MeldHelper.can_minkan(hand_tiles, discard_tile):
            return None

        discard_normalized = normalize_tile(discard_tile)

        # 找到手中的三张相同的牌
        matching_tiles = [
            t for t in hand_tiles if normalize_tile(t) == discard_normalized
        ][:3]

        if len(matching_tiles) < 3:
            return None

        # 创建副露
        meld_tiles = [discard_tile] + matching_tiles
        meld = Meld(meld_type="minkan", tiles=meld_tiles, from_who=from_player)

        return meld, matching_tiles

    @staticmethod
    def can_ankan(hand_tiles: List[str]) -> List[str]:
        """
        检查可以暗杠的牌

        Args:
            hand_tiles: 手牌列表

        Returns:
            List[str]: 可以暗杠的牌列表
        """
        tile_counts = {}
        for tile in hand_tiles:
            normalized = normalize_tile(tile)
            tile_counts[normalized] = tile_counts.get(normalized, 0) + 1

        # 找到所有有4张的牌
        ankan_tiles = [tile for tile, count in tile_counts.items() if count == 4]

        return ankan_tiles

    @staticmethod
    def create_ankan_meld(
        hand_tiles: List[str], target_tile: str
    ) -> Optional[Tuple[Meld, List[str]]]:
        """
        创建暗杠的副露

        Args:
            hand_tiles: 手牌列表
            target_tile: 要杠的牌（任意一张，用于确定类型）

        Returns:
            Optional[Tuple[Meld, List[str]]]: (副露对象, 需要从手牌移除的牌)
        """
        target_normalized = normalize_tile(target_tile)

        # 找到手中的四张相同的牌
        matching_tiles = [
            t for t in hand_tiles if normalize_tile(t) == target_normalized
        ]

        if len(matching_tiles) < 4:
            return None

        # 创建副露（暗杠from_who=-1表示自己）
        meld = Meld(meld_type="ankan", tiles=matching_tiles, from_who=-1)

        return meld, matching_tiles

    @staticmethod
    def can_kakan(hand_tiles: List[str], open_melds: List[Meld]) -> List[str]:
        """
        检查可以加杠的牌

        Args:
            hand_tiles: 手牌列表
            open_melds: 已有的副露列表

        Returns:
            List[str]: 可以加杠的牌列表
        """
        kakan_tiles = []

        # 找到所有碰过的牌
        for meld in open_melds:
            if meld.meld_type == "pon":
                pon_tile = normalize_tile(meld.tiles[0])
                # 检查手中是否有第四张
                for tile in hand_tiles:
                    if normalize_tile(tile) == pon_tile:
                        kakan_tiles.append(tile)
                        break

        return kakan_tiles

    @staticmethod
    def create_kakan_meld(
        hand_tiles: List[str],
        open_melds: List[Meld],
        target_tile: str,
    ) -> Optional[Tuple[Meld, Meld, str]]:
        """
        创建加杠的副露（将碰升级为杠）

        Args:
            hand_tiles: 手牌列表
            open_melds: 已有的副露列表
            target_tile: 要加杠的牌

        Returns:
            Optional[Tuple[Meld, Meld, str]]: (新的杠副露, 原来的碰副露, 要移除的手牌)
        """
        target_normalized = normalize_tile(target_tile)

        # 找到对应的碰副露
        original_pon = None
        for meld in open_melds:
            if meld.meld_type == "pon":
                if normalize_tile(meld.tiles[0]) == target_normalized:
                    original_pon = meld
                    break

        if original_pon is None:
            return None

        # 检查手中是否有这张牌
        if target_tile not in hand_tiles:
            return None

        # 创建新的加杠副露
        kakan_tiles = original_pon.tiles + [target_tile]
        kakan_meld = Meld(
            meld_type="kakan", tiles=kakan_tiles, from_who=original_pon.from_who
        )

        return kakan_meld, original_pon, target_tile
