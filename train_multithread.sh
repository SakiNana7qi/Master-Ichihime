#!/bin/bash
# 多线程训练启动脚本 (Linux/Mac)

echo "=========================================="
echo "  麻将 AI 多线程训练启动脚本"
echo "=========================================="

# 1. 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

echo "Python 版本: $(python --version)"

# 2. 设置环境变量以启用多线程
echo ""
echo "设置环境变量..."

# 获取 CPU 核心数
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "32")
echo "检测到 CPU 核心数: $CPU_COUNT"

# 设置 OpenMP、MKL、OpenBLAS 线程数
export OMP_NUM_THREADS=$CPU_COUNT
export MKL_NUM_THREADS=$CPU_COUNT
export OPENBLAS_NUM_THREADS=$CPU_COUNT
export NUMEXPR_NUM_THREADS=$CPU_COUNT
export VECLIB_MAXIMUM_THREADS=$CPU_COUNT

# 禁用 OpenMP 等待策略（减少 CPU 空转）
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0

# 设置 NumPy/BLAS 线程亲和度
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "环境变量配置完成:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OMP_WAIT_POLICY=$OMP_WAIT_POLICY"

# 3. 检查 psutil
echo ""
echo "检查依赖..."
python -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: psutil 未安装，无法设置 CPU 亲和度"
    echo "建议运行: pip install psutil"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 4. 启动训练
echo ""
echo "=========================================="
echo "  开始训练"
echo "=========================================="
echo ""

# 使用多线程配置
python -m mahjong_agent.train \
    --config multithread \
    --device cuda \
    --seed 42

echo ""
echo "训练完成或已退出"


