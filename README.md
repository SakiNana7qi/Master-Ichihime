# Master-Ichihime

雀魂立直麻将AI开发项目

## 项目简介

本项目旨在开发一个完整的雀魂立直麻将AI系统，包括：

1. **麻将算点器** (`mahjong_scorer/`) - 完整的日本立直麻将规则引擎
2. **麻将环境** (`mahjong_environment/`) - 基于Gymnasium的多智能体强化学习环境
3. **麻将AI Agent** (`mahjong_agent/`) - 基于PPO算法的强化学习AI 🆕

## 项目结构

```
Master-Ichihime/
├── mahjong_scorer/              # 麻将算点器
│   ├── main_scorer.py          # 主算点器类
│   ├── hand_analyzer.py        # 手牌分析
│   ├── yaku_checker.py         # 役种判断
│   ├── fu_calculator.py        # 符数计算
│   ├── point_distributor.py    # 点数分配
│   └── utils/                  # 工具模块
│       ├── structures.py       # 数据结构
│       ├── constants.py        # 常量定义
│       └── tile_converter.py   # 牌转换工具
│
├── mahjong_environment/         # 麻将环境
│   ├── mahjong_env.py          # 主环境类
│   ├── game_state.py           # 游戏状态管理
│   ├── player_state.py         # 玩家状态管理
│   ├── utils/                  # 工具模块
│   │   ├── action_encoder.py      # 动作编码
│   │   ├── legal_actions_helper.py # 合法动作检测
│   │   ├── meld_helper.py         # 副露辅助
│   │   └── tile_utils.py          # 牌工具
│   ├── test_env.py             # 测试脚本
│   ├── example_random_agent.py # 随机智能体示例
│   └── README.md               # 环境文档
│
├── mahjong_agent/               # 麻将AI Agent 🆕
│   ├── model.py                # Actor-Critic神经网络
│   ├── rollout_buffer.py       # 经验缓冲区
│   ├── ppo_updater.py          # PPO算法更新器
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   ├── config.py               # 配置文件
│   └── README.md               # Agent文档
│
├── quickstart_agent.py          # Agent快速入门 🆕
├── AGENT_GUIDE.md              # Agent完整指南 🆕
├── README_SCORER.md            # 算点器详细文档
└── README.md                   # 本文件
```

## 快速开始

### 🚀 立即开始训练AI

**最快的方式 - 使用快速入门脚本:**

```bash
# 交互式菜单，包含模型演示、环境交互、训练等
python quickstart_agent.py
```

**或者直接开始训练:**

```bash
# Windows
train_quickstart.bat

# Linux/Mac
chmod +x train_quickstart.sh
./train_quickstart.sh
```

### 📦 安装依赖

#### 基础依赖（算点器 + 环境）
```bash
pip install numpy gymnasium
```

#### AI Agent依赖（训练AI需要）
```bash
pip install torch tensorboard tqdm matplotlib
# 或直接安装所有依赖
pip install -r mahjong_agent/requirements.txt
```

### 🤖 训练和评估AI

#### 快速训练（用于测试）
```bash
python -m mahjong_agent.train --config fast --device cuda
```

#### 标准训练（推荐）
```bash
python -m mahjong_agent.train --config default --device cuda --seed 42
```

#### 并行高吞吐训练（多线程/多环境）
```bash
# 预设将启用多环境与多线程（可在 AGENT_GUIDE.md 中查看细节）
python -m mahjong_agent.train --config multithread --device cuda

# PowerShell 可选设置（建议与 CPU 线程数一致）
$env:OMP_NUM_THREADS="32"; $env:MKL_NUM_THREADS="32"
```

#### 评估模型
```bash
# 评估训练好的模型
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --episodes 100

# 交互式演示
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --interactive
```

#### 监控训练
```bash
# 启动TensorBoard查看训练曲线
tensorboard --logdir logs/
```

#### 完整教程
查看 [AGENT_GUIDE.md](AGENT_GUIDE.md) 获取详细的训练指南、配置说明和最佳实践。

---

### 2. 使用算点器

```python
from mahjong_scorer.main_scorer import MainScorer
from mahjong_scorer.utils.structures import HandInfo, GameState

# 创建算点器
scorer = MainScorer()

# 定义手牌和游戏状态
hand_info = HandInfo(
    hand_tiles=["2m", "3m", "4m", "5p", "6p", "7p", "3s", "4s", "5s", 
                "6s", "7s", "8s", "9z"],
    winning_tile="9z",
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

print(f"番数: {result.han}")
print(f"符数: {result.fu}")
print(f"得点: {result.winner_gain}")
for yaku in result.yaku_list:
    print(f"  - {yaku.name}: {yaku.han}番")
```

### 3. 使用麻将环境

```python
from mahjong_environment import MahjongEnv
import random
import numpy as np

# 创建环境
env = MahjongEnv(render_mode="human", seed=42)

# 重置环境
obs, info = env.reset()

# 游戏循环
while not env.terminations[env.agent_selection]:
    # 获取合法动作
    action_mask = obs["action_mask"]
    legal_actions = np.where(action_mask == 1)[0]
    
    # 选择动作（随机）
    action = random.choice(legal_actions)
    
    # 执行动作
    env.step(action)
    
    # 渲染
    env.render()
    
    # 获取新观测
    if env.agent_selection:
        obs = env.observe(env.agent_selection)

# 查看结果
for agent in env.possible_agents:
    print(f"{agent}: 奖励 = {env.rewards[agent]}")
```

### 4. 运行测试

```bash
# 测试算点器
python test_scorer.py

# 测试环境
cd mahjong_environment
python test_env.py

# 运行随机智能体示例
python example_random_agent.py
```

## 功能特性

### 算点器特性

✅ 完整的日本立直麻将规则  
✅ 支持所有标准役种（1番~役满）  
✅ 准确的符数计算  
✅ 支持赤宝牌  
✅ 支持吃、碰、杠  
✅ 支持特殊和牌（七对子、国士无双等）  
✅ 自动点数分配（考虑庄家、本场、立直棒）

详见：[算点器文档](README_SCORER.md)

### 环境特性

✅ PettingZoo风格的多智能体API  
✅ 完整的游戏流程模拟  
✅ 部分可观察（每个玩家只看到自己手牌）  
✅ 动作掩码（自动提供合法动作）  
✅ 集成算点器（自动计算得分）  
✅ 支持渲染（文本模式）  
✅ 完整的吃碰杠立直逻辑

详见：[环境文档](mahjong_environment/README.md)

## API文档

### 算点器API

主要类：`MainScorer`

```python
class MainScorer:
    def calculate_score(
        self, 
        hand_info: HandInfo, 
        game_state: GameState
    ) -> ScoreResult:
        """计算和牌得分"""
        
    def is_tenpai(
        self, 
        hand_tiles: List[str], 
        open_melds: List[Meld] = None
    ) -> bool:
        """判断是否听牌"""
        
    def get_waiting_tiles(
        self, 
        hand_tiles: List[str], 
        open_melds: List[Meld] = None
    ) -> List[str]:
        """获取所有听牌的牌"""
```

### 环境API

主要类：`MahjongEnv`

```python
class MahjongEnv:
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """重置环境到初始状态"""
        
    def step(self, action: int):
        """执行一个动作"""
        
    def observe(self, agent: str) -> Dict:
        """获取指定智能体的观测"""
        
    def render(self):
        """渲染当前游戏状态"""
```

## 开发计划

### 已完成 ✅

- [x] 完整的算点器实现
- [x] 基础环境框架
- [x] 观测空间和动作空间设计
- [x] 游戏状态管理
- [x] 吃碰杠立直逻辑
- [x] 和牌判断和结算
- [x] 测试脚本和示例代码

### 进行中 🚧

- [ ] AI智能体开发
- [ ] 强化学习训练流程
- [ ] 性能优化

### 计划中 📋

- [ ] 完善振听判断
- [ ] 实现流局听牌费
- [ ] 实现抢杠
- [ ] 支持多家和牌
- [ ] Web界面
- [ ] 预训练模型

## 技术栈

- **Python 3.8+**
- **NumPy**: 数值计算
- **Gymnasium**: 强化学习环境标准
- 未来可能使用：
  - PyTorch / TensorFlow: 深度学习
  - Ray RLlib: 分布式训练
  - Stable-Baselines3: RL算法库

## 牌的表示

本项目使用字符串表示麻将牌：

- **万子**: `1m`, `2m`, ..., `9m`
- **筒子**: `1p`, `2p`, ..., `9p`
- **索子**: `1s`, `2s`, ..., `9s`
- **字牌**: `1z`(东), `2z`(南), `3z`(西), `4z`(北), `5z`(白), `6z`(发), `7z`(中)
- **赤宝牌**: `0m`(赤5万), `0p`(赤5筒), `0s`(赤5索)

## 贡献指南

欢迎提交Issue和Pull Request！

### 开发建议

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 相关资源

- [日本麻将规则Wiki](https://ja.wikipedia.org/wiki/%E9%BA%BB%E9%9B%80)
- [雀魂官网](https://mahjongsoul.com/)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [PettingZoo文档](https://pettingzoo.farama.org/)

## 许可证

本项目使用 MIT 许可证。

## 致谢

感谢所有为日本麻将规则整理和AI开发做出贡献的开发者。

---

**注意**: 本项目仍在积极开发中，API可能会有变动。