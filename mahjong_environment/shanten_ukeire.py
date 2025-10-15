"""
精确向听与 ukeire 计算（无外部依赖，优先使用纯 Python 实现）。

提供函数：
    compute_shanten_and_ukeire(hand34_counts: np.ndarray) -> tuple[int, int]

约定：
- hand34_counts: 长度34的数组（万0-8, 筒9-17, 索18-26, 字27-33），均为0..4。
- 返回：(标准手最小向听, ukeire_tile_types)，其中 ukeire 为能使向听减少1的牌种类数。

实现说明：
- 标准手向听：使用枚举雀头位置 + 各花色/字牌回溯提取面子/搭子，最小化 shanten。
- 七对子/国士无双：分别计算并取最小值（与标准手最小值取 min）。
- ukeire：尝试摸入任意尚未用尽（<4张）的34种牌，看是否能使向听下降1，统计种类数。

注意：这是高质量但仍简化的实现，适用于训练 shaping 与策略引导。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple
import numpy as np


MAN_START = 0
PIN_START = 9
SOU_START = 18
HONOR_START = 27


def _is_honor(i: int) -> bool:
    return i >= HONOR_START


def _split_suits(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        counts[MAN_START : MAN_START + 9].copy(),
        counts[PIN_START : PIN_START + 9].copy(),
        counts[SOU_START : SOU_START + 9].copy(),
        counts[HONOR_START : HONOR_START + 7].copy(),
    )


def _tuple(a: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(x) for x in a.tolist())


@lru_cache(maxsize=None)
def _best_m_t_for_suit(counts_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """
    对单一花色(9张)的计数，返回能获得的最大(面子m, 搭子t)。
    使用回溯搜索：优先尝试刻子/顺子，探索所有组合，记录最大 m,t。
    """
    counts = list(counts_tuple)

    # 跳过前导0
    i = 0
    while i < 9 and counts[i] == 0:
        i += 1
    if i == 9:
        return 0, 0

    best_m, best_t = 0, 0

    # 选项1：刻子
    if counts[i] >= 3:
        counts[i] -= 3
        m, t = _best_m_t_for_suit(tuple(counts))
        counts[i] += 3
        m += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t

    # 选项2：顺子
    if i <= 6 and counts[i + 1] > 0 and counts[i + 2] > 0:
        counts[i] -= 1
        counts[i + 1] -= 1
        counts[i + 2] -= 1
        m, t = _best_m_t_for_suit(tuple(counts))
        counts[i] += 1
        counts[i + 1] += 1
        counts[i + 2] += 1
        m += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t

    # 选项3：搭子（对子 / 两连 / 跳张）
    # 对子
    if counts[i] >= 2:
        counts[i] -= 2
        m, t = _best_m_t_for_suit(tuple(counts))
        counts[i] += 2
        t += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t
    # 两连
    if i <= 7 and counts[i + 1] > 0:
        counts[i] -= 1
        counts[i + 1] -= 1
        m, t = _best_m_t_for_suit(tuple(counts))
        counts[i] += 1
        counts[i + 1] += 1
        t += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t
    # 跳张
    if i <= 6 and counts[i + 2] > 0:
        counts[i] -= 1
        counts[i + 2] -= 1
        m, t = _best_m_t_for_suit(tuple(counts))
        counts[i] += 1
        counts[i + 2] += 1
        t += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t

    # 选项4：弃用该张（不组任何）
    counts[i] -= 1
    m, t = _best_m_t_for_suit(tuple(counts))
    counts[i] += 1
    if (m > best_m) or (m == best_m and t > best_t):
        best_m, best_t = m, t

    return best_m, best_t


@lru_cache(maxsize=None)
def _best_m_t_for_honors(counts_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """
    字牌(7张)只能以刻子为面子，以对子为搭子，回溯搜索最大 m,t。
    """
    counts = list(counts_tuple)
    i = 0
    while i < 7 and counts[i] == 0:
        i += 1
    if i == 7:
        return 0, 0

    best_m, best_t = 0, 0

    # 刻子
    if counts[i] >= 3:
        counts[i] -= 3
        m, t = _best_m_t_for_honors(tuple(counts))
        counts[i] += 3
        m += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t

    # 对子作搭子
    if counts[i] >= 2:
        counts[i] -= 2
        m, t = _best_m_t_for_honors(tuple(counts))
        counts[i] += 2
        t += 1
        if (m > best_m) or (m == best_m and t > best_t):
            best_m, best_t = m, t

    # 弃用
    counts[i] -= 1
    m, t = _best_m_t_for_honors(tuple(counts))
    counts[i] += 1
    if (m > best_m) or (m == best_m and t > best_t):
        best_m, best_t = m, t

    return best_m, best_t


def _standard_shanten(counts: np.ndarray) -> int:
    """标准手（四面子一雀头）向听，取最小。"""
    man, pin, sou, honors = _split_suits(counts)

    # 无雀头情形（允许 t 最大化但最终公式惩罚）
    m1, t1 = _best_m_t_for_suit(_tuple(man))
    m2, t2 = _best_m_t_for_suit(_tuple(pin))
    m3, t3 = _best_m_t_for_suit(_tuple(sou))
    mh, th = _best_m_t_for_honors(_tuple(honors))
    m_total = m1 + m2 + m3 + mh
    t_total = t1 + t2 + t3 + th
    sh_no_pair = 8 - 2 * m_total - min(4 - m_total, t_total) - 0

    best = sh_no_pair

    # 穷举雀头放置：对每一种有>=2张的牌，先取作雀头，再计算最佳 m,t
    for i in range(34):
        if counts[i] < 2:
            continue
        counts[i] -= 2
        m1, t1 = _best_m_t_for_suit(_tuple(counts[MAN_START : MAN_START + 9]))
        m2, t2 = _best_m_t_for_suit(_tuple(counts[PIN_START : PIN_START + 9]))
        m3, t3 = _best_m_t_for_suit(_tuple(counts[SOU_START : SOU_START + 9]))
        mh, th = _best_m_t_for_honors(_tuple(counts[HONOR_START : HONOR_START + 7]))
        counts[i] += 2
        m_total = m1 + m2 + m3 + mh
        t_total = t1 + t2 + t3 + th
        sh = 8 - 2 * m_total - min(4 - m_total, t_total) - 1
        if sh < best:
            best = sh

    return max(-1, best)


def _chiitoi_shanten(counts: np.ndarray) -> int:
    """七对子向听：6 - 对子数 + max(0, 7 - 不同牌种数)。"""
    pairs = int(np.sum(counts >= 2))
    uniques = int(np.sum(counts > 0))
    need_pairs = 6 - pairs
    need_uniques = max(0, 7 - uniques)
    return max(0, need_pairs + need_uniques)


def _kokushi_shanten(counts: np.ndarray) -> int:
    """国士向听：13 - 幺九种类数 - 是否已有幺九对子。"""
    yaojiu_indices = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
    has_pair = 0
    distinct = 0
    for i in yaojiu_indices:
        if counts[i] > 0:
            distinct += 1
            if counts[i] >= 2:
                has_pair = 1
    return 13 - distinct - has_pair


def compute_shanten_and_ukeire(hand34_counts: np.ndarray) -> Tuple[int, int]:
    """
    返回最小向听与 ukeire（能使向听下降1的牌种类数）。
    """
    counts = np.asarray(hand34_counts, dtype=np.int8).clip(0, 4)

    std = _standard_shanten(counts.copy())
    chiitoi = _chiitoi_shanten(counts)
    kokushi = _kokushi_shanten(counts)
    shanten = min(std, chiitoi, kokushi)

    # 计算 ukeire（按牌种类数）
    ukeire_types = 0
    if shanten > -1:  # 非和牌状态才有下降空间
        for i in range(34):
            if counts[i] >= 4:
                continue
            counts[i] += 1
            s2 = min(
                _standard_shanten(counts.copy()),
                _chiitoi_shanten(counts),
                _kokushi_shanten(counts),
            )
            counts[i] -= 1
            if s2 < shanten:
                ukeire_types += 1

    return int(shanten), int(ukeire_types)




