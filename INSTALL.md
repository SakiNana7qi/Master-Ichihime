# 安装指南

本文档提供详细的安装步骤和环境配置说明。

## 系统要求

### 最低要求
- **操作系统**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.8 或更高版本
- **内存**: 8GB RAM
- **存储**: 5GB 可用空间

### 推荐配置（训练AI）
- **CPU**: 8核以上
- **内存**: 16GB RAM
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA支持)
- **存储**: 20GB 可用空间（用于日志和检查点）

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/Master-Ichihime.git
cd Master-Ichihime
```

### 2. 创建虚拟环境（推荐）

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

#### 基础功能（算点器 + 环境）
```bash
pip install numpy gymnasium
```

#### 完整功能（包含AI训练）

**方式一：使用requirements文件**
```bash
# 安装环境依赖
pip install -r mahjong_environment/requirements.txt

# 安装Agent依赖
pip install -r mahjong_agent/requirements.txt
```

**方式二：手动安装**
```bash
# 基础依赖
pip install numpy>=1.24.0

# 强化学习环境
pip install gymnasium>=0.29.0

# 深度学习框架
pip install torch>=2.0.0 torchvision>=0.15.0

# 训练工具
pip install tensorboard>=2.13.0 tqdm>=4.65.0

# 其他工具
pip install matplotlib>=3.7.0
```

---

## GPU支持（CUDA）

如果您有NVIDIA GPU并想加速训练：

### 检查CUDA版本
```bash
nvidia-smi
```

### 安装对应版本的PyTorch

访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择合适的版本。

例如，CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

或者CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 验证CUDA
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## 验证安装

### 测试算点器
```python
from mahjong_scorer.main_scorer import MainScorer
scorer = MainScorer()
print("✓ 算点器安装成功")
```

### 测试环境
```python
from mahjong_environment import MahjongEnv
env = MahjongEnv()
obs, info = env.reset()
print("✓ 环境安装成功")
```

### 测试Agent
```python
from mahjong_agent import MahjongActorCritic, get_default_config
model = MahjongActorCritic(get_default_config())
print("✓ Agent安装成功")
```

### 完整测试脚本
```bash
python -c "
from mahjong_scorer.main_scorer import MainScorer
from mahjong_environment import MahjongEnv
from mahjong_agent import MahjongActorCritic, get_default_config

print('测试算点器...', end=' ')
scorer = MainScorer()
print('✓')

print('测试环境...', end=' ')
env = MahjongEnv()
env.reset()
print('✓')

print('测试Agent...', end=' ')
model = MahjongActorCritic(get_default_config())
print('✓')

print('\n所有组件安装成功！')
"
```

---

## 常见问题

### Q1: 导入错误 "No module named 'xxx'"

**A**: 确保已激活虚拟环境并安装了所有依赖：
```bash
pip install -r mahjong_agent/requirements.txt
```

### Q2: CUDA not available

**A**: 
1. 检查是否安装了NVIDIA驱动：`nvidia-smi`
2. 安装对应CUDA版本的PyTorch
3. 如果没有GPU，使用CPU训练：`--device cpu`

### Q3: ImportError相关错误

**A**: 确保从项目根目录运行脚本，或正确设置PYTHONPATH：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%      # Windows
```

### Q4: 内存不足

**A**: 调整训练参数：
```python
config.mini_batch_size = 128  # 减小批次
config.rollout_steps = 1024   # 减少rollout步数
```

### Q5: 依赖冲突

**A**: 使用虚拟环境隔离依赖：
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
fresh_env\Scripts\activate     # Windows
pip install -r mahjong_agent/requirements.txt
```

---

## 卸载

### 删除虚拟环境
```bash
# 停用虚拟环境
deactivate

# 删除虚拟环境文件夹
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows
```

### 清理训练文件
```bash
rm -rf checkpoints/ logs/ *.pt *.pth
```

---

## 开发环境设置

如果您想参与开发：

### 安装开发依赖
```bash
pip install pytest black flake8 mypy
```

### 代码格式化
```bash
black mahjong_agent/
```

### 代码检查
```bash
flake8 mahjong_agent/
mypy mahjong_agent/
```

### 运行测试
```bash
pytest mahjong_environment/test_env.py
```

---

## 更新

### 更新代码
```bash
git pull origin main
```

### 更新依赖
```bash
pip install --upgrade -r mahjong_agent/requirements.txt
```

---

## 技术支持

如遇到问题：
1. 查看 [常见问题](#常见问题)
2. 查看 [Issue列表](https://github.com/yourusername/Master-Ichihime/issues)
3. 提交新Issue

---

**安装完成后，查看 [README.md](README.md) 开始使用！** 🀄
