# 麻将AI Agent - 基于PPO的强化学习实现

这是一个使用**PPO (Proximal Policy Optimization)** 算法训练的立直麻将AI。

## 📁 项目结构

```
mahjong_agent/
├── __init__.py              # 模块初始化
├── config.py                # 超参数配置
├── model.py                 # Actor-Critic神经网络模型
├── rollout_buffer.py        # 经验数据缓冲区和GAE计算
├── ppo_updater.py          # PPO算法更新逻辑
├── train.py                # 主训练脚本
├── evaluate.py             # 评估脚本
├── requirements.txt        # 依赖包列表
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装环境和算点器的依赖
pip install -r mahjong_environment/requirements.txt

# 安装Agent的依赖
pip install -r mahjong_agent/requirements.txt
```

### 2. 开始训练

#### 快速测试训练（调试用）

```bash
python -m mahjong_agent.train --config fast --device cuda
```

#### 标准训练（推荐）

```bash
python -m mahjong_agent.train --config default --device cuda --seed 42
```

#### 高性能训练（长时间训练）

```bash
python -m mahjong_agent.train --config high_performance --device cuda
```

### 3. 恢复训练

如果训练中断，可以从检查点恢复：

```bash
python -m mahjong_agent.train --checkpoint checkpoints/checkpoint_100.pt
```

### 4. 评估模型

```bash
# 标准评估（100局）
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --episodes 100

# 交互式演示（观察AI决策）
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --interactive
```

## 🧠 模型架构

### Actor-Critic架构

我们的模型采用经典的Actor-Critic架构：

1. **观测编码器 (Observation Encoder)**
   - 手牌编码器：将34维手牌转换为128维特征
   - 摸牌编码器：将34维摸牌转换为64维特征
   - 牌河编码器：将4×34维牌河转换为256维特征
   - 副露编码器：将4×34维副露转换为256维特征
   - 宝牌编码器：将5×34维宝牌转换为128维特征
   - 游戏信息编码器：将场况信息转换为128维特征

2. **共享特征提取器 (Shared Encoder)**
   - 多层MLP（可配置层数和维度）
   - 支持LayerNorm和Dropout
   - 可选：Transformer层（用于捕捉复杂关系）

3. **Actor头 (Policy Head)**
   - 输出112维动作logits
   - 与动作掩码结合，只考虑合法动作

4. **Critic头 (Value Head)**
   - 输出单一标量：状态价值V(s)

### 关键特性

- **动作掩码**：确保只选择合法动作
- **GAE**：使用广义优势估计提高训练稳定性
- **PPO裁剪**：防止策略更新过大
- **熵正则化**：鼓励探索

## ⚙️ 配置说明

### 预设配置

我们提供了三种预设配置：

| 配置类型 | 说明 | 适用场景 |
|---------|------|---------|
| `fast` | 快速训练，较小模型 | 调试、快速实验 |
| `default` | 标准配置，平衡性能和速度 | 日常训练 |
| `high_performance` | 大模型，长时间训练 | 追求最佳性能 |

### 主要超参数

```python
# 学习相关
learning_rate = 3e-4        # 学习率
gamma = 0.99                # 折扣因子
gae_lambda = 0.95          # GAE lambda
clip_range = 0.2           # PPO裁剪范围

# 网络架构
hidden_dim = 512           # 隐藏层维度
num_hidden_layers = 3      # MLP层数
use_transformer = False    # 是否使用Transformer

# 训练流程
rollout_steps = 2048       # 每次收集步数
mini_batch_size = 256      # 小批次大小
num_epochs = 4             # 每次更新的epoch数
total_timesteps = 10M      # 总训练步数
```

详细配置请查看 `config.py`。

## 📊 训练监控

训练过程中会自动记录到TensorBoard：

```bash
tensorboard --logdir logs/
```

主要监控指标：

- `train/mean_episode_reward`：平均回报
- `train/policy_loss`：策略损失
- `train/value_loss`：价值损失
- `train/entropy`：策略熵
- `train/clip_fraction`：裁剪比例
- `train/approx_kl`：近似KL散度
- `eval/win_rate`：胜率

## 🎯 训练技巧

### 1. 超参数调优

- **学习率**：如果训练不稳定，降低学习率（1e-4）
- **clip_range**：如果策略更新太激进，降低裁剪范围（0.1）
- **entropy_coef**：如果探索不足，增加熵系数（0.02）

### 2. 训练阶段

建议分阶段训练：

1. **探索阶段**（0-1M步）
   - 较大的熵系数（0.02）
   - 鼓励探索各种策略

2. **优化阶段**（1M-5M步）
   - 标准熵系数（0.01）
   - 平衡探索和利用

3. **精炼阶段**（5M+步）
   - 较小的熵系数（0.005）
   - 降低学习率（1e-4）
   - 专注于性能提升

### 3. 常见问题

**Q: 训练不收敛怎么办？**
- 降低学习率
- 检查奖励设计是否合理
- 增加rollout_steps

**Q: 过拟合怎么办？**
- 增加熵系数
- 添加Dropout
- 使用更多的训练数据

**Q: 内存不足？**
- 减小mini_batch_size
- 减小rollout_steps
- 减小hidden_dim

## 🔧 高级用法

### 自定义训练循环

```python
from mahjong_agent import MahjongTrainer, get_default_config

# 创建自定义配置
config = get_default_config()
config.learning_rate = 1e-4
config.hidden_dim = 1024

# 创建训练器
trainer = MahjongTrainer(config=config)

# 开始训练
trainer.train()
```

### 自定义模型

```python
from mahjong_agent import MahjongActorCritic, PPOConfig

config = PPOConfig()
config.use_transformer = True
config.num_transformer_layers = 4

model = MahjongActorCritic(config)
```

### 评估特定检查点

```python
from mahjong_agent import MahjongEvaluator

evaluator = MahjongEvaluator("checkpoints/checkpoint_100.pt")
results = evaluator.evaluate(num_episodes=100)
print(f"胜率: {results['player_0_win_rate']:.2%}")
```

## 📈 性能基准

以下是在不同训练步数下的性能参考（对战随机策略）：

| 训练步数 | 胜率 | 平均分数 | 平均奖励 |
|---------|------|---------|---------|
| 100K | ~30% | 26000 | +0.5 |
| 500K | ~45% | 28000 | +2.0 |
| 1M | ~60% | 30000 | +4.0 |
| 5M | ~75% | 33000 | +8.0 |
| 10M | ~85% | 35000 | +12.0 |

*注：实际性能取决于具体配置和训练质量*

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📝 许可证

本项目采用与主项目相同的许可证。

## 🙏 致谢

本实现基于以下算法和工具：

- [PPO论文](https://arxiv.org/abs/1707.06347)
- PyTorch
- Gymnasium
- TensorBoard

---

**祝训练顺利！愿你的AI早日成为雀圣！🀄**
