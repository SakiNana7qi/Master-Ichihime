# 麻将环境 (Mahjong Environment)

基于 PettingZoo 风格的多智能体立直麻将强化学习环境。

## 功能特性

- ✅ **完整的日本立直麻将规则**
- ✅ **多智能体支持**：4个玩家同时对局
- ✅ **Gymnasium/PettingZoo 兼容API**
- ✅ **部分可观察**：每个玩家只能看到自己的手牌
- ✅ **动作掩码**：自动提供合法动作列表
- ✅ **完整的游戏模拟**：包括吃、碰、杠、立直、和牌等所有动作
- ✅ **集成算点器**：自动计算番数、符数和点数

## 安装依赖

```bash
pip install numpy gymnasium
```

## 快速开始

### 基本使用

```python
from mahjong_environment import MahjongEnv

# 创建环境
env = MahjongEnv(render_mode="human", seed=42)

# 重置环境
obs, info = env.reset()

# 游戏循环
done = False
while not done:
    # 获取当前玩家
    current_agent = env.agent_selection
    
    # 获取合法动作
    action_mask = obs["action_mask"]
    legal_actions = [i for i, legal in enumerate(action_mask) if legal]
    
    # 选择动作（这里使用随机动作）
    import random
    action = random.choice(legal_actions)
    
    # 执行动作
    env.step(action)
    
    # 渲染
    env.render()
    
    # 检查是否结束
    done = env.terminations[current_agent]
    
    # 获取新的观测
    if not done:
        obs = env.observe(env.agent_selection)

# 查看结果
for agent in env.possible_agents:
    print(f"{agent}: 奖励 = {env.rewards[agent]}")
```

### 与自定义AI集成

```python
from mahjong_environment import MahjongEnv

class MyMahjongAgent:
    """你的麻将AI"""
    
    def select_action(self, observation, legal_actions):
        """根据观测选择动作"""
        # 实现你的策略
        # observation包含：手牌、牌河、副露、宝牌等信息
        # legal_actions是所有合法动作的列表
        
        # 示例：简单的规则策略
        action_mask = observation["action_mask"]
        
        # 如果可以自摸，就自摸
        if action_mask[108]:  # TSUMO = 108
            return 108
        
        # 如果可以荣和，就荣和
        if action_mask[109]:  # RON = 109
            return 109
        
        # 否则打出危险度最低的牌
        import random
        return random.choice(legal_actions)

# 使用自定义AI
env = MahjongEnv(render_mode="human")
agent = MyMahjongAgent()

obs, info = env.reset()

while not any(env.terminations.values()):
    action_mask = obs["action_mask"]
    legal_actions = [i for i, legal in enumerate(action_mask) if legal]
    
    action = agent.select_action(obs, legal_actions)
    env.step(action)
    
    if env.agent_selection:
        obs = env.observe(env.agent_selection)
```

## 观测空间

每个观测是一个字典，包含以下键：

| 键 | 形状 | 描述 |
|---|---|---|
| `hand` | (34,) | 自己的手牌，34种牌各有几张 |
| `drawn_tile` | (34,) | 刚摸到的牌（one-hot编码） |
| `rivers` | (4, 34) | 4个玩家的牌河 |
| `melds` | (4, 34) | 4个玩家的副露 |
| `riichi_status` | (4,) | 4个玩家的立直状态 |
| `scores` | (4,) | 4个玩家的分数（归一化） |
| `dora_indicators` | (5, 34) | 宝牌指示牌（最多5个） |
| `game_info` | (5,) | 场风、自风、本场数、立直棒、剩余牌数 |
| `phase_info` | (3,) | 是否轮到自己、是否打牌阶段、是否响应阶段 |
| `action_mask` | (112,) | 动作掩码，标记哪些动作合法 |

## 动作空间

动作空间是一个离散空间，包含112个可能的动作：

- **0-33**: 打出第i种牌（34种基本牌型）
- **34-67**: 打出第i种牌并立直
- **68-70**: 吃牌（左吃、中吃、右吃）
- **71**: 碰
- **72**: 明杠
- **73**: 暗杠
- **74-107**: 加杠（34种牌）
- **108**: 自摸和
- **109**: 荣和
- **110**: 跳过/不响应
- **111**: 九种九牌流局

**重要**：并非所有动作在所有时刻都是合法的。使用 `observation["action_mask"]` 来获取当前合法的动作。

## 奖励

- 和牌时：奖励 = 得分变化 / 1000（归一化）
- 放铳时：奖励 = 负的失分 / 1000
- 流局时：奖励 = 0（简化处理）

## 架构设计

```
mahjong_environment/
├── __init__.py                 # 包入口
├── mahjong_env.py             # 主环境类
├── game_state.py              # 游戏状态管理
├── player_state.py            # 玩家状态管理
├── utils/
│   ├── action_encoder.py      # 动作编码解码
│   ├── legal_actions_helper.py # 合法动作检测
│   ├── meld_helper.py         # 副露辅助工具
│   └── tile_utils.py          # 牌相关工具函数
├── test_env.py                # 测试脚本
└── README.md                  # 本文件
```

## 核心类说明

### MahjongEnv

主环境类，实现PettingZoo风格的API：

- `reset()`: 重置环境
- `step(action)`: 执行动作
- `observe(agent)`: 获取观测
- `render()`: 渲染当前状态

### GameState

游戏状态管理器，负责：

- 牌山管理（发牌、摸牌）
- 宝牌指示牌
- 场况信息（东几局、本场数）
- 状态机（发牌→打牌→响应→摸牌→...）

### PlayerState

玩家状态，包含：

- 手牌（私密）
- 牌河（公开）
- 副露（公开）
- 立直状态
- 分数

## 与算点器集成

环境自动集成了 `mahjong_scorer` 算点器，用于：

- 判断和牌形态
- 计算役种和番数
- 计算符数
- 计算最终得点

## 测试

运行测试脚本：

```bash
cd mahjong_environment
python test_env.py
```

测试包括：

1. 基本初始化
2. 重置功能
3. 观测空间验证
4. 动作编码解码
5. 牌工具函数
6. 随机动作执行

## 训练建议

### 特征工程

建议从观测中提取以下特征：

1. **手牌特征**：每种牌的数量、孤张、搭子、刻子
2. **进攻特征**：向听数、有效牌数量
3. **防守特征**：他家牌河、危险牌判断
4. **场况特征**：剩余牌数、宝牌数量、当前分数差距

### 奖励塑形

可以自定义奖励函数，例如：

```python
# 原始奖励（得点变化）
base_reward = env.rewards[agent]

# 添加额外奖励
if player_won:
    reward = base_reward + 10  # 和牌奖励
elif riichi_success:
    reward = base_reward + 1   # 立直成功奖励
elif avoid_houjuu:
    reward = base_reward + 0.5 # 避免放铳奖励
```

### 训练算法

推荐使用以下算法：

- **PPO** (Proximal Policy Optimization)
- **IMPALA** (Importance Weighted Actor-Learner Architecture)
- **Alpha Zero** 风格的自对弈

## 限制和简化

当前版本有以下简化：

1. **振听判断**：仅实现基本振听，未实现同巡振听
2. **流局听牌**：流局时暂不判断听牌费
3. **抢杠**：暂未实现抢杠功能
4. **多家和牌**：暂不支持多家同时荣和
5. **连庄规则**：简化处理，庄家和牌或流局听牌时连庄

## TODO

未来可能的改进：

- [ ] 完善振听判断
- [ ] 实现流局听牌费
- [ ] 实现抢杠
- [ ] 支持多家和牌
- [ ] 添加更详细的调试信息
- [ ] 优化性能（Cython加速）
- [ ] 添加replay功能
- [ ] 提供预训练模型

## 相关资源

- [麻将规则Wiki](https://ja.wikipedia.org/wiki/%E9%BA%BB%E9%9B%80%E3%81%AE%E5%BD%B9%E4%B8%80%E8%A6%A7)
- [雀魂官网](https://mahjongsoul.com/)
- [PettingZoo文档](https://pettingzoo.farama.org/)

## 许可证

本项目使用与主项目相同的许可证。

## 贡献

欢迎提交Issue和Pull Request！
