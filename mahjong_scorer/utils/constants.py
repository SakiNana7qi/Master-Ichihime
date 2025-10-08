# mahjong_scorer/utils/constants.py

# ===== 役种名称常量 =====

# 1番役
RIICHI = "立直"
IPPATSU = "一发"
TSUMO = "门前清自摸和"
PINFU = "平和"
TANYAO = "断幺九"
IIPEIKOU = "一杯口"
HAKU = "役牌 白"
HATSU = "役牌 发"
CHUN = "役牌 中"
SEAT_WIND = "自风牌"  # 动态添加（东南西北）
PREVALENT_WIND = "场风牌"  # 动态添加（东南西北）
HAITEI = "海底捞月"
HOUTEI = "河底捞鱼"
RINSHAN = "岭上开花"
CHANKAN = "抢杠"

# 2番役
DOUBLE_RIICHI = "两立直"
CHIITOITSU = "七对子"
CHANTA = "混全带幺九"
ITTSU = "一气通贯"
SANSHOKU_DOUJUN = "三色同顺"
SANSHOKU_DOUKOU = "三色同刻"
SANKANTSU = "三杠子"
TOITOI = "对对和"
SANANKOU = "三暗刻"
SHOUSANGEN = "小三元"
HONROUTOU = "混老头"

# 3番役
HONITSU = "混一色"
JUNCHAN = "纯全带幺九"
RYANPEIKOU = "两杯口"

# 6番役
CHINITSU = "清一色"

# 役满 (13番)
KOKUSHI = "国士无双"
KOKUSHI_13 = "国士无双十三面"
SUUANKOU = "四暗刻"
SUUANKOU_TANKI = "四暗刻单骑"
DAISANGEN = "大三元"
SHOUSUUSHII = "小四喜"
DAISUUSHII = "大四喜"
TSUUIISOU = "字一色"
RYUUIISOU = "绿一色"
CHINROUTOU = "清老头"
SUUKANTSU = "四杠子"
CHUURENPOUTOU = "九莲宝灯"
CHUURENPOUTOU_9 = "纯正九莲宝灯"
TENHOU = "天和"
CHIIHOU = "地和"

# 役种番数映射
YAKU_HAN = {
    # 1番
    RIICHI: 1,
    IPPATSU: 1,
    TSUMO: 1,
    PINFU: 1,
    TANYAO: 1,
    IIPEIKOU: 1,
    HAKU: 1,
    HATSU: 1,
    CHUN: 1,
    SEAT_WIND: 1,
    PREVALENT_WIND: 1,
    HAITEI: 1,
    HOUTEI: 1,
    RINSHAN: 1,
    CHANKAN: 1,
    # 2番
    DOUBLE_RIICHI: 2,
    CHIITOITSU: 2,
    CHANTA: 2,
    ITTSU: 2,
    SANSHOKU_DOUJUN: 2,
    SANSHOKU_DOUKOU: 2,
    SANKANTSU: 2,
    TOITOI: 2,
    SANANKOU: 2,
    SHOUSANGEN: 2,
    HONROUTOU: 2,
    # 3番
    HONITSU: 3,
    JUNCHAN: 3,
    RYANPEIKOU: 3,
    # 6番
    CHINITSU: 6,
    # 役满
    KOKUSHI: 13,
    KOKUSHI_13: 26,  # 双倍役满
    SUUANKOU: 13,
    SUUANKOU_TANKI: 26,  # 双倍役满
    DAISANGEN: 13,
    SHOUSUUSHII: 13,
    DAISUUSHII: 26,  # 双倍役满
    TSUUIISOU: 13,
    RYUUIISOU: 13,
    CHINROUTOU: 13,
    SUUKANTSU: 13,
    CHUURENPOUTOU: 13,
    CHUURENPOUTOU_9: 26,  # 双倍役满
    TENHOU: 13,
    CHIIHOU: 13,
}

# ===== 牌的常量 =====

# 数牌
MANZU = [f"{i}m" for i in range(1, 10)]  # 万子
PINZU = [f"{i}p" for i in range(1, 10)]  # 筒子
SOUZU = [f"{i}s" for i in range(1, 10)]  # 索子

# 字牌
HONORS = [f"{i}z" for i in range(1, 8)]  # 字牌
WINDS = ["1z", "2z", "3z", "4z"]  # 东南西北
DRAGONS = ["5z", "6z", "7z"]  # 白发中

# 幺九牌
TERMINALS = ["1m", "9m", "1p", "9p", "1s", "9s"]  # 老头牌
TERMINALS_AND_HONORS = TERMINALS + HONORS  # 幺九牌

# 特殊役用牌
GREEN_TILES = ["2s", "3s", "4s", "6s", "8s", "6z"]  # 绿一色用牌

# 简单牌（中张牌）
SIMPLES = (
    [f"{i}m" for i in range(2, 9)]
    + [f"{i}p" for i in range(2, 9)]
    + [f"{i}s" for i in range(2, 9)]
)

# 风对应关系
WIND_MAP = {"east": "1z", "south": "2z", "west": "3z", "north": "4z"}
WIND_NAMES = {"east": "东", "south": "南", "west": "西", "north": "北"}

# 龙牌对应
DRAGON_WHITE = "5z"  # 白
DRAGON_GREEN = "6z"  # 发
DRAGON_RED = "7z"  # 中
DRAGON_CHUN = "7z"  # 中（别名）

# ===== 役种复合规则 =====

# 互相排斥的役种（不能同时成立）
MUTUALLY_EXCLUSIVE_YAKU = [
    (RIICHI, DOUBLE_RIICHI),  # 立直和两立直
    (PINFU, CHIITOITSU),  # 平和和七对子
    (PINFU, TOITOI),  # 平和和对对和
    (IIPEIKOU, CHIITOITSU),  # 一杯口和七对子
    (IIPEIKOU, RYANPEIKOU),  # 一杯口和两杯口
    (CHANTA, JUNCHAN),  # 混全带和纯全带
    (CHANTA, HONROUTOU),  # 混全带和混老头
    (HONITSU, CHINITSU),  # 混一色和清一色
    (TANYAO, CHANTA),  # 断幺和混全带
    (TANYAO, JUNCHAN),  # 断幺和纯全带
    (TANYAO, HONROUTOU),  # 断幺和混老头
]

# 食下时减少番数的役种（门前时3番，食下时2番）
KUISAGARI_YAKU = {
    HONITSU: 2,  # 混一色：门前3番，食下2番
    CHINITSU: 5,  # 清一色：门前6番，食下5番
    CHANTA: 1,  # 混全带：门前2番，食下1番
    JUNCHAN: 2,  # 纯全带：门前3番，食下2番
    ITTSU: 1,  # 一气通贯：门前2番，食下1番
    SANSHOKU_DOUJUN: 1,  # 三色同顺：门前2番，食下1番
}
