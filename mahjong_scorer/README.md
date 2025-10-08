# 麻将算点器 (Mahjong Scorer)

雀魂立直麻将的完整算点系统。

## 快速开始

```python
from mahjong_scorer import MainScorer, HandInfo, GameState

scorer = MainScorer()

# 定义手牌和游戏状态
hand_info = HandInfo(
    hand_tiles=["2m", "3m", "4m", "5p", "6p", "7p", "3s", "4s", "5s", "6s", "7s", "8s", "2p"],
    winning_tile="2p",
    win_type="RON"
)

game_state = GameState(
    player_wind="south",
    prevalent_wind="east",
    is_riichi=True,
    dora_indicators=["1m"]
)

# 计算得分
result = scorer.calculate_score(hand_info, game_state)
print(f"番数: {result.han}番, 符数: {result.fu}符")
print(f"获得: {result.winner_gain}点")
```

## 主要功能

- ✅ 完整的役种判断（1番役到役满）
- ✅ 精确的符数计算
- ✅ 点数分配系统
- ✅ 宝牌计算（表宝牌、里宝牌、赤宝牌）
- ✅ 手牌分析和面子分解
- ✅ 听牌判断

## 文件说明

- `main_scorer.py` - 主算点器，提供统一接口
- `hand_analyzer.py` - 手牌分析，面子分解
- `yaku_checker.py` - 役种判断
- `fu_calculator.py` - 符数计算
- `point_distributor.py` - 点数分配
- `utils/` - 工具模块（常量、数据结构、牌转换）

## 更多文档

详细使用说明请参考项目根目录的 `README_SCORER.md`。

## 测试

运行测试脚本：
```bash
python test_scorer.py
```
