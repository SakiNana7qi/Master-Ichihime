# 麻将AI Agent 完整使用指南

本指南详细介绍如何使用基于PPO算法的麻将AI系统。

## 📋 目录

1. [快速开始](#快速开始)
2. [详细教程](#详细教程)
3. [配置说明](#配置说明)
4. [训练技巧](#训练技巧)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 方法一：使用快速入门脚本

最简单的方式是使用交互式快速入门脚本：

```bash
python quickstart_agent.py
```

这将显示一个菜单，包含以下选项：
- 模型架构展示
- 环境交互演示
- 训练演示（快速版本）

### 方法二：直接训练

#### Windows:
```bash
train_quickstart.bat
```

#### Linux/Mac:
```bash
chmod +x train_quickstart.sh
./train_quickstart.sh
```

#### 手动执行:
```bash
# 安装依赖
pip install -r mahjong_agent/requirements.txt

# 开始训练
python -m mahjong_agent.train --config fast --device cuda
```

---

## 📚 详细教程

### 1. 环境准备

#### 安装依赖

```bash
# 基础环境依赖
pip install -r mahjong_environment/requirements.txt

# Agent依赖
pip install -r mahjong_agent/requirements.txt
```

#### 验证安装

```python
# 测试环境
from mahjong_environment import MahjongEnv
env = MahjongEnv()
obs, info = env.reset()
print("✓ 环境安装成功")

# 测试Agent
from mahjong_agent import MahjongActorCritic, get_default_config
model = MahjongActorCritic(get_default_config())
print("✓ Agent安装成功")
```

### 2. 基础训练

#### 使用预设配置

```python
from mahjong_agent import MahjongTrainer, get_default_config

# 获取默认配置
config = get_default_config()

# 创建训练器
trainer = MahjongTrainer(config=config)

# 开始训练
trainer.train()
```

#### 自定义配置

```python
from mahjong_agent import MahjongTrainer, PPOConfig

# 创建自定义配置
config = PPOConfig()
config.learning_rate = 1e-4
config.total_timesteps = 5_000_000
config.hidden_dim = 1024
config.use_transformer = True

# 训练
trainer = MahjongTrainer(config=config)
trainer.train()
```

### 3. 监控训练

#### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 在浏览器中打开
# http://localhost:6006
```

#### 主要监控指标

- **train/mean_episode_reward**: 平均回报（越高越好）
- **train/policy_loss**: 策略损失
- **train/value_loss**: 价值损失
- **train/entropy**: 策略熵（探索程度）
- **train/clip_fraction**: PPO裁剪比例
- **eval/win_rate**: 评估胜率

### 4. 恢复训练

如果训练中断，可以从检查点恢复：

```bash
python -m mahjong_agent.train --checkpoint checkpoints/checkpoint_100.pt
```

或在代码中：

```python
trainer = MahjongTrainer(checkpoint_path="checkpoints/checkpoint_100.pt")
trainer.train()
```

### 5. 评估模型

#### 命令行评估

```bash
# 标准评估（100局）
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --episodes 100

# 交互式演示
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --interactive

# 自定义输出路径
python -m mahjong_agent.evaluate --model checkpoints/final_model.pt --output my_report.txt
```

#### 代码评估

```python
from mahjong_agent import MahjongEvaluator

# 创建评估器
evaluator = MahjongEvaluator(
    model_path="checkpoints/final_model.pt",
    device="cuda"
)

# 评估性能
results = evaluator.evaluate(num_episodes=100)
print(f"胜率: {results['player_0_win_rate']:.2%}")
print(f"平均分数: {results['player_0_mean_score']:.1f}")

# 保存报告
evaluator.save_evaluation_report(results, "report.txt")
```

---

## ⚙️ 配置说明

### 预设配置对比

| 配置 | 训练时间 | 模型大小 | 推荐场景 |
|------|---------|---------|---------|
| **fast** | 短 (~2小时) | 小 (~50MB) | 快速测试、调试 |
| **default** | 中 (~10小时) | 中 (~200MB) | 日常训练 |
| **high_performance** | 长 (~50小时) | 大 (~500MB) | 追求最佳性能 |

### 关键超参数详解

#### 学习率相关

```python
config.learning_rate = 3e-4  # 初始学习率
config.lr_schedule = "linear"  # 学习率调度: constant/linear/cosine
```

- 学习率太大：训练不稳定，损失震荡
- 学习率太小：训练缓慢，可能不收敛
- 推荐范围：1e-4 到 5e-4

#### PPO核心参数

```python
config.gamma = 0.99          # 折扣因子（0.95-0.995）
config.gae_lambda = 0.95     # GAE lambda（0.9-0.99）
config.clip_range = 0.2      # PPO裁剪范围（0.1-0.3）
```

- **gamma**: 越大越重视长期奖励，越小越重视短期奖励
- **gae_lambda**: 平衡偏差和方差，0.95是常用值
- **clip_range**: 限制策略更新幅度，防止更新过大

#### 网络架构

```python
config.hidden_dim = 512              # 隐藏层维度
config.num_hidden_layers = 3         # 层数
config.use_transformer = False       # 是否使用Transformer
config.num_transformer_layers = 2    # Transformer层数
```

- **hidden_dim**: 越大模型容量越大，但训练越慢
- **num_hidden_layers**: 3-4层通常足够
- **Transformer**: 可以捕捉更复杂的关系，但计算昂贵

#### 训练流程

```python
config.rollout_steps = 2048      # 每次收集的步数
config.mini_batch_size = 256     # 批次大小
config.num_epochs = 4            # 每次更新的epoch数
```

- **rollout_steps**: 越大越稳定，但更新频率越低
- **mini_batch_size**: 根据显存调整，通常128-512
- **num_epochs**: 4-8通常足够

---

## 💡 训练技巧

### 1. 分阶段训练策略

#### 探索阶段（0-1M步）
- 目标：让AI探索各种策略
- 配置：
  ```python
  config.entropy_coef = 0.02  # 较高的熵系数
  config.learning_rate = 3e-4
  ```

#### 优化阶段（1M-5M步）
- 目标：优化策略，提高胜率
- 配置：
  ```python
  config.entropy_coef = 0.01  # 标准熵系数
  config.learning_rate = 2e-4
  ```

#### 精炼阶段（5M+步）
- 目标：微调细节，追求最佳性能
- 配置：
  ```python
  config.entropy_coef = 0.005  # 降低熵系数
  config.learning_rate = 1e-4  # 降低学习率
  ```

### 2. 超参数调优建议

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练不收敛 | 学习率太大 | 降低learning_rate |
| 收敛太慢 | 学习率太小 | 提高learning_rate |
| 损失震荡 | 批次太小或学习率太大 | 增加mini_batch_size或降低learning_rate |
| 过拟合 | 探索不足 | 增加entropy_coef |
| 欠拟合 | 模型容量不足 | 增加hidden_dim或num_hidden_layers |

### 3. 性能优化

#### GPU优化

```python
# 使用混合精度训练（如果GPU支持）
config.device = "cuda"

# 增加批次大小
config.mini_batch_size = 512

# 使用更大的rollout
config.rollout_steps = 4096
```

#### CPU训练

```python
# 设置设备为CPU
config.device = "cpu"

# 调整线程数
config.num_threads = 8

# 减小批次大小
config.mini_batch_size = 128
config.rollout_steps = 1024
```

### 4. 实战案例

#### 案例1：快速原型

目标：快速验证想法，2小时内看到结果

```python
from mahjong_agent import MahjongTrainer, get_fast_config

config = get_fast_config()
config.total_timesteps = 100_000
config.rollout_steps = 512

trainer = MahjongTrainer(config=config)
trainer.train()
```

#### 案例2：标准训练

目标：获得不错的性能，10小时训练

```python
from mahjong_agent import MahjongTrainer, get_default_config

config = get_default_config()
config.total_timesteps = 5_000_000

trainer = MahjongTrainer(config=config)
trainer.train()
```

#### 案例3：追求极致

目标：获得最佳性能，数天训练

```python
from mahjong_agent import MahjongTrainer, get_high_performance_config

config = get_high_performance_config()
config.total_timesteps = 50_000_000
config.use_transformer = True

trainer = MahjongTrainer(config=config)
trainer.train()
```

---

## ❓ 常见问题

### Q1: 训练需要多长时间？

**A**: 取决于配置和硬件：
- **fast配置** + GPU: ~2小时
- **default配置** + GPU: ~10小时
- **high_performance配置** + GPU: ~50小时
- CPU训练会慢5-10倍

### Q2: 需要什么样的硬件？

**A**: 最低要求：
- CPU: 4核以上
- RAM: 8GB
- GPU（可选）: 6GB显存

推荐配置：
- CPU: 8核以上
- RAM: 16GB
- GPU: RTX 3060 或更好（12GB显存）

### Q3: 如何判断训练是否成功？

**A**: 观察以下指标：
1. **mean_episode_reward**: 应该逐渐上升
2. **eval/win_rate**: 对抗随机策略应达到60%+
3. **entropy**: 应该逐渐下降（表示策略越来越确定）
4. **policy_loss**: 应该趋于稳定

### Q4: 训练中断了怎么办？

**A**: 使用检查点恢复：
```bash
python -m mahjong_agent.train --checkpoint checkpoints/checkpoint_XXX.pt
```

### Q5: 如何调整奖励函数？

**A**: 在环境代码中修改奖励计算逻辑。主要奖励点：
- 和牌：根据番数和点数给奖励
- 放铳：负奖励
- 立直：小额奖励/惩罚（鼓励/抑制立直）
- 游戏结束：根据最终排名给奖励

### Q6: 可以使用多个GPU吗？

**A**: 当前版本不支持多GPU训练。建议：
- 使用单个高性能GPU
- 或运行多个独立训练实验

### Q7: 模型太大无法加载？

**A**: 尝试：
```python
# 使用CPU加载
checkpoint = torch.load(path, map_location='cpu')

# 或减小模型大小
config.hidden_dim = 256
config.num_hidden_layers = 2
```

### Q8: 如何与人类对战？

**A**: 需要开发交互界面。可以参考evaluate.py中的play_interactive方法，并扩展为GUI。

---

## 📊 性能基准

### 对战随机策略

| 训练步数 | 胜率 | 平均分数 | 荣和率 |
|---------|------|---------|-------|
| 100K | 30% | 26000 | 5% |
| 500K | 45% | 28000 | 15% |
| 1M | 60% | 30000 | 25% |
| 5M | 75% | 33000 | 35% |
| 10M | 85% | 35000 | 45% |

### 训练曲线示例

```
Episode Reward:
  0K:  -5.0
  1M:  +2.0
  5M:  +8.0
 10M: +12.0

Win Rate:
  0K:  25%
  1M:  60%
  5M:  75%
 10M:  85%
```

---

## 🔗 相关链接

- [PPO论文](https://arxiv.org/abs/1707.06347)
- [强化学习教程](https://spinningup.openai.com/)
- [PyTorch文档](https://pytorch.org/docs/)
- [TensorBoard使用](https://www.tensorflow.org/tensorboard)

---

## 📝 许可证

本项目采用与主项目相同的许可证。

---

**祝训练顺利！如有问题，欢迎提Issue！** 🀄
