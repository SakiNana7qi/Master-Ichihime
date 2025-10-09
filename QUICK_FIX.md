# 🔧 快速修复：CPU 只用一个核心的问题

## 问题现象

- 训练时只有 CPU0 (或某个核心) 100% 使用率
- 其他 35 个核心使用率 <3%
- 训练速度很慢

## 根本原因

1. **多进程环境未启用** - 默认只用单环境
2. **psutil 未安装** - 无法设置 CPU 亲和度
3. **环境变量未设置** - NumPy/科学计算库只用 1 个线程

## ✅ 立即修复（4 步）

### 第 1 步：停止当前训练

按 `Ctrl+C` 停止正在运行的训练。

### 第 2 步：安装/确认依赖

```bash
pip install psutil tqdm
```

### 第 3 步：运行诊断

```bash
python diagnose_cpu.py
```

查看输出，确认：
- psutil 已安装 ✓
- 检测到你的 72 个 CPU ✓

### 第 4 步：重新启动训练

```bash
# 最简单的方法（会自动根据CPU核心数配置）
python train_multithread.py
```

或者手动设置：

```bash
# Linux/Mac
export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36
export OMP_WAIT_POLICY=PASSIVE

python -m mahjong_agent.train --config multithread --device cuda
```

## 验证是否成功

### 72 核 CPU (你的配置)

启动后应该看到：

```
使用设备: cuda
PyTorch 线程数: 72 (inter-op: ...)
环境配置: num_envs=32, pin_cpu_affinity=True
使用多进程环境模式：32 个并行环境
[信息] 为 32 个子进程分配 CPU 亲和度 (系统总共 72 个逻辑核心)
  进程  0 (PID xxxxx) -> CPU [0, 1]
  进程  1 (PID xxxxx) -> CPU [2, 3]
  ...（32 个进程，每个分配 2 个核心）
  进程 31 (PID xxxxx) -> CPU [62, 63]
```

打开任务管理器，应该看到：
- **多个 CPU 核心** 都在 40-80% 使用率
- **总体 CPU 使用率** > 50%
- **约 33 个 Python 进程**（1 主 + 32 子）
- **使用率分布均匀**，不是只有一个核心 100%

## 🎯 性能提升（72 核 CPU）

修复前：
- num_envs: 16
- CPU 使用: ~3% (只有 CPU0 是 100%)
- FPS: ~237
- 每个 update: ~276 秒

修复后（预期）：
- num_envs: 32（自动根据 72 核调整）
- CPU 使用: ~60% (分布在所有核心)
- FPS: ~800-1200
- 每个 update: ~60-80 秒

**速度提升约 8-12 倍！**

## ❌ 如果还是不行

### 检查 1: 确认配置

```bash
# 确保使用了 multithread 配置
python -c "
from mahjong_agent.config_multithread import get_multithread_config
config = get_multithread_config()
print(f'num_envs: {config.num_envs}')
print(f'num_threads: {getattr(config, \"num_threads\", \"未设置\")}')
"
```

应该输出：
```
num_envs: 8
num_threads: 32
```

### 检查 2: 环境变量

```bash
# 检查是否被限制为 1
echo $OMP_NUM_THREADS
echo $MKL_NUM_THREADS

# 如果输出是 1，需要重新设置或 unset
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
```

### 检查 3: 测试多进程

```bash
python test_multiprocess.py
```

如果测试失败，查看错误信息。

## 📚 详细文档

完整指南见：`MULTITHREAD_GUIDE.md`

## 🆘 还有问题？

1. 运行 `python diagnose_cpu.py` 获取完整诊断
2. 运行 `python test_multiprocess.py` 测试多进程
3. 检查是否有 `psutil` 警告
4. 确认 `--config multithread` 参数正确

---

**记住最重要的 3 点：**
1. ✅ 安装 `psutil`
2. ✅ 使用 `--config multithread`
3. ✅ 设置环境变量 `OMP_NUM_THREADS=36`


