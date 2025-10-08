# 雀魂立直麻将算点器

一个完整的日本麻将（立直麻将）算点系统，用于雀魂立直麻将AI的环境实现。

## 功能特性

- ✅ **完整的役种判断**：支持所有常见役种和役满
  - 1番役：立直、平和、断幺九、一杯口、役牌等
  - 2番役：七对子、三色同顺、一气通贯、对对和等
  - 3番役：混一色、两杯口、纯全带等
  - 6番役：清一色
  - 役满：国士无双、四暗刻、大三元、字一色、绿一色等
  
- ✅ **精确的符数计算**：完整实现日麻符数规则
  - 底符、听牌形态符
  - 刻子、杠子符数（区分明暗、幺九中张）
  - 雀头符数（役牌加符）
  
- ✅ **点数分配系统**：自动计算点数支付
  - 荣和/自摸点数计算
  - 庄家/闲家点数差异
  - 本场棒、立直棒处理
  
- ✅ **宝牌计算**：支持表宝牌、里宝牌、赤宝牌

- ✅ **手牌分析**：自动分解面子和雀头
  - 常规牌型（4面子1雀头）
  - 特殊牌型（七对子、国士无双）
  - 听牌判断和听牌牌型识别

## 安装

将 `mahjong_scorer` 文件夹放入你的项目中，确保Python版本 >= 3.7。

```bash
# 项目结构
your_project/
  ├── mahjong_scorer/
  │   ├── __init__.py
  │   ├── main_scorer.py
  │   ├── hand_analyzer.py
  │   ├── yaku_checker.py
  │   ├── fu_calculator.py
  │   ├── point_distributor.py
  │   └── utils/
  │       ├── __init__.py
  │       ├── constants.py
  │       ├── structures.py
  │       └── tile_converter.py
  └── your_code.py
```

## 使用方法

### 基本用法

```python
from mahjong_scorer import MainScorer, HandInfo, GameState

# 创建算点器实例
scorer = MainScorer()

# 定义手牌信息
hand_info = HandInfo(
    hand_tiles=["2m", "3m", "4m", "5p", "6p", "7p", "3s", "4s", "5s", "6s", "7s", "8s", "2p"],
    winning_tile="2p",
    win_type="RON"  # 或 "TSUMO"
)

# 定义游戏状态
game_state = GameState(
    player_wind="south",      # 自风：east/south/west/north
    prevalent_wind="east",    # 场风：east/south/west/north
    is_riichi=True,           # 是否立直
    dora_indicators=["1m"]    # 宝牌指示牌
)

# 计算得分
result = scorer.calculate_score(hand_info, game_state)

# 查看结果
print(f"番数: {result.han}番")
print(f"符数: {result.fu}符")
print(f"和牌者获得: {result.winner_gain}点")
for yaku in result.yaku_list:
    print(f"  - {yaku.name}: {yaku.han}番")
```

### 牌的表示方法

- 数牌：`1m`-`9m`（万子）、`1p`-`9p`（筒子）、`1s`-`9s`（索子）
- 字牌：`1z`（东）、`2z`（南）、`3z`（西）、`4z`（北）、`5z`（白）、`6z`（发）、`7z`（中）
- 赤宝牌：`0m`（赤5万）、`0p`（赤5筒）、`0s`（赤5索）

### 副露的表示

```python
from mahjong_scorer import Meld

# 吃
chi_meld = Meld(
    meld_type="chi",
    tiles=["2m", "3m", "4m"],
    from_who=1  # 从哪家吃的（0=上家，1=对家，2=下家）
)

# 碰
pon_meld = Meld(
    meld_type="pon",
    tiles=["5p", "5p", "5p"],
    from_who=0
)

# 明杠
minkan_meld = Meld(
    meld_type="minkan",
    tiles=["7s", "7s", "7s", "7s"],
    from_who=1
)

# 暗杠
ankan_meld = Meld(
    meld_type="ankan",
    tiles=["1z", "1z", "1z", "1z"],
    from_who=-1  # 暗杠不从他家拿牌
)

# 使用副露
hand_info = HandInfo(
    hand_tiles=["2m", "3m", "4m", "5p", "6p", "7p", "8p"],  # 剩余手牌
    open_melds=[chi_meld, pon_meld],  # 副露
    winning_tile="8p",
    win_type="RON"
)
```

### 完整示例

```python
from mahjong_scorer import MainScorer, HandInfo, GameState

scorer = MainScorer()

# 示例1：清一色 + 立直 + 一气通贯
hand_info = HandInfo(
    hand_tiles=["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2m", "3m", "4m", "5m"],
    winning_tile="5m",
    win_type="TSUMO"
)

game_state = GameState(
    player_wind="south",
    prevalent_wind="east",
    is_riichi=True,
    dora_indicators=["1m"]
)

result = scorer.calculate_score(hand_info, game_state)

if result.error:
    print(f"错误: {result.error}")
else:
    print(f"番数: {result.han}番")
    print(f"符数: {result.fu}符")
    print(f"基本点: {result.base_points}")
    print(f"和牌者获得: {result.winner_gain}点")
    print("役种:")
    for yaku in result.yaku_list:
        print(f"  {yaku.name} ({yaku.han}番)")
    print("点数支付:")
    for payment in result.payments:
        print(f"  {payment['from']}: {payment['amount']}点")
```

### 听牌判断

```python
from mahjong_scorer import MainScorer

scorer = MainScorer()

# 检查是否听牌
hand_tiles = ["2m", "3m", "4m", "5p", "6p", "7p", "3s", "4s", "5s", "6s", "7s", "8s", "2p"]
is_tenpai = scorer.is_tenpai(hand_tiles)

if is_tenpai:
    # 获取所有听牌
    waiting_tiles = scorer.get_waiting_tiles(hand_tiles)
    print(f"听牌: {waiting_tiles}")
```

## API参考

### MainScorer

主算点器类。

#### 方法

- `calculate_score(hand_info: HandInfo, game_state: GameState) -> ScoreResult`
  - 计算和牌得分
  
- `is_tenpai(hand_tiles: List[str], open_melds: List = None) -> bool`
  - 判断是否听牌
  
- `get_waiting_tiles(hand_tiles: List[str], open_melds: List = None) -> List[str]`
  - 获取所有听牌的牌

### 数据结构

#### HandInfo

```python
@dataclass
class HandInfo:
    hand_tiles: List[Tile]              # 手牌列表
    open_melds: List[Meld] = []         # 副露列表
    winning_tile: Tile = ""             # 和牌张
    win_type: Literal["RON", "TSUMO"] = "RON"  # 和牌方式
```

#### GameState

```python
@dataclass
class GameState:
    player_wind: str = "east"                    # 自风
    prevalent_wind: str = "east"                 # 场风
    honba: int = 0                               # 本场数
    kyotaku_sticks: int = 0                      # 立直棒数
    dora_indicators: List[Tile] = []             # 宝牌指示牌
    ura_dora_indicators: List[Tile] = []         # 里宝牌指示牌
    is_riichi: bool = False                      # 立直
    is_double_riichi: bool = False               # 两立直
    is_ippatsu: bool = False                     # 一发
    is_rinshan: bool = False                     # 岭上开花
    is_chankan: bool = False                     # 抢杠
    is_haitei: bool = False                      # 海底捞月
    is_houtei: bool = False                      # 河底捞鱼
    is_tenhou: bool = False                      # 天和
    is_chihou: bool = False                      # 地和
```

#### ScoreResult

```python
@dataclass
class ScoreResult:
    han: int = 0                                 # 总番数
    fu: int = 0                                  # 符数
    base_points: int = 0                         # 基本点
    winner_gain: int = 0                         # 和牌者获得点数
    payments: List[dict] = []                    # 点数支付详情
    yaku_list: List[YakuResult] = []             # 役种列表
    error: str = ""                              # 错误信息
```

## 支持的役种

### 1番役
- 立直、一发、门前清自摸和、平和、断幺九
- 一杯口、役牌（白、发、中、自风、场风）
- 海底捞月、河底捞鱼、岭上开花、抢杠

### 2番役
- 两立直、七对子、混全带幺九、一气通贯
- 三色同顺、三色同刻、三杠子、对对和
- 三暗刻、小三元、混老头

### 3番役
- 混一色、纯全带幺九、两杯口

### 6番役
- 清一色

### 役满（13番）
- 国士无双、四暗刻、大三元、小四喜、大四喜
- 字一色、绿一色、清老头、四杠子、九莲宝灯
- 天和、地和

### 双倍役满（26番）
- 国士无双十三面、四暗刻单骑、纯正九莲宝灯、大四喜

## 运行测试

### 自动测试

```bash
python test_scorer.py
```

这将运行一系列示例，包括：
1. 平和 + 立直
2. 断幺九 + 门前清自摸和
3. 对对和 + 三暗刻
4. 清一色 + 一气通贯
5. 大三元（役满）
6. 听牌判断

### 交互式算点器

```bash
python simply_scorer.py
```

这是一个交互式工具，支持：
- 简短格式输入手牌（如：`123m456p789s1177z`）
- 逐步输入和牌张、和牌方式、游戏状态
- 自动显示庄家和闲家的点数对比
- 清晰显示役种、宝牌、符数等详细信息

**输入格式说明：**
- 数字后跟花色：`m`=万子, `p`=筒子, `s`=索子, `z`=字牌
- 赤宝牌：`0m`=赤5万, `0p`=赤5筒, `0s`=赤5索
- 字牌：`1z`=东, `2z`=南, `3z`=西, `4z`=北, `5z`=白, `6z`=发, `7z`=中

**示例输入：**
```
手牌: 123m456p789s1177z
和牌张: 7z
和牌方式: RON
自风: east
场风: south
是否立直: y
宝牌指示牌: 1m
```

## 注意事项

1. **庄家/闲家判断**：
   - 庄家 = 自风为 `east` 的玩家
   - 闲家 = 自风为 `south/west/north` 的玩家
   - 庄家和牌点数是闲家的 1.5 倍

2. **平和规则**：本实现遵循雀魂规则，平和必须是荣和（不能自摸）

3. **食下减番**：部分役种在副露后会减少番数
   - 混一色：门前3番 → 食下2番
   - 清一色：门前6番 → 食下5番
   - 混全带：门前2番 → 食下1番
   - 纯全带：门前3番 → 食下2番
   - 一气通贯、三色同顺：门前2番 → 食下1番

4. **役种复合**：自动处理互相排斥的役种
   - 立直与两立直
   - 一杯口与两杯口
   - 混全带与纯全带
   - 混一色与清一色
   - 等等

5. **宝牌计算**：
   - 表宝牌：所有情况都计算
   - 里宝牌：只有立直时才计算
   - 赤宝牌：`0m/0p/0s` 自动识别并计算

## 项目结构

```
Master-Ichihime/
├── mahjong_scorer/          # 算点器包
│   ├── __init__.py          # 包初始化，导出主要接口
│   ├── main_scorer.py       # 主算点器
│   ├── hand_analyzer.py     # 手牌分析器（面子分解）
│   ├── yaku_checker.py      # 役种判断器
│   ├── fu_calculator.py     # 符数计算器
│   ├── point_distributor.py # 点数分配器
│   ├── README.md            # 算点器简要说明
│   └── utils/
│       ├── __init__.py      # 工具模块初始化
│       ├── constants.py     # 常量定义（役种名称、牌的常量等）
│       ├── structures.py    # 数据结构定义
│       └── tile_converter.py # 牌的转换函数
├── test_scorer.py           # 自动测试脚本
├── simply_scorer.py         # 交互式算点器
└── README_SCORER.md         # 本文档（详细说明）
```

## 贡献

欢迎提出问题和改进建议！

## 许可证

本项目遵循项目根目录的LICENSE文件。

## 作者

Master-Ichihime

---

**用于雀魂立直麻将AI项目**
