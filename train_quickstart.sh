#!/bin/bash
# 快速训练脚本 - Linux/Mac

echo "======================================"
echo "  麻将AI训练 - 快速开始"
echo "======================================"
echo ""

# 检查Python
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

echo "1. 检查依赖..."
python -c "import torch; import numpy; import gymnasium" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖包..."
    pip install -r mahjong_agent/requirements.txt
fi

echo ""
echo "2. 开始训练（快速配置）..."
python -m mahjong_agent.train --config fast --device cuda --seed 42

echo ""
echo "======================================"
echo "  训练完成！"
echo "======================================"
echo ""
echo "检查点保存在: ./checkpoints/"
echo "日志保存在: ./logs/"
echo ""
echo "使用 'tensorboard --logdir logs/' 查看训练曲线"
echo ""
