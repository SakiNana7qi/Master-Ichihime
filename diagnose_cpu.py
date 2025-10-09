#!/usr/bin/env python
"""
CPU 和多线程配置诊断脚本
用于检查系统配置是否正确，诊断 CPU 利用率问题
"""

import os
import sys
import platform
import multiprocessing as mp

print("=" * 80)
print("系统和多线程配置诊断")
print("=" * 80)

# 1. 系统信息
print("\n【系统信息】")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python 版本: {sys.version.split()[0]}")
print(f"CPU 架构: {platform.machine()}")

# 2. CPU 信息
print("\n【CPU 信息】")
try:
    import psutil
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    print(f"✓ psutil 已安装")
    print(f"  逻辑 CPU 数: {cpu_count_logical}")
    print(f"  物理 CPU 数: {cpu_count_physical}")
    
    # CPU 频率
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"  CPU 频率: {cpu_freq.current:.0f} MHz (最大: {cpu_freq.max:.0f} MHz)")
    
    # 当前 CPU 使用率
    print(f"\n  当前总体 CPU 使用率: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"  各核心使用率:", psutil.cpu_percent(interval=1, percpu=True))
    
except ImportError:
    print("✗ psutil 未安装")
    print("  请运行: pip install psutil")
    print("  没有 psutil，无法设置 CPU 亲和度！")
    cpu_count_logical = mp.cpu_count()
    cpu_count_physical = cpu_count_logical // 2  # 估算
    print(f"  multiprocessing.cpu_count(): {cpu_count_logical}")

# 3. PyTorch 配置
print("\n【PyTorch 配置】")
try:
    import torch
    print(f"✓ PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 线程配置
    print(f"\n  当前 PyTorch CPU 线程数: {torch.get_num_threads()}")
    print(f"  Inter-op 线程数: {torch.get_num_interop_threads()}")
    
    # 测试设置线程数
    torch.set_num_threads(cpu_count_logical)
    print(f"  测试设置后的线程数: {torch.get_num_threads()}")
    
except ImportError:
    print("✗ PyTorch 未安装")

# 4. NumPy 配置
print("\n【NumPy 配置】")
try:
    import numpy as np
    print(f"✓ NumPy 版本: {np.__version__}")
    
    # 检查 BLAS 库
    try:
        config = np.__config__.show()
        print("  NumPy 配置:", config if config else "无详细信息")
    except:
        print("  无法获取 NumPy 详细配置")
        
except ImportError:
    print("✗ NumPy 未安装")

# 5. 环境变量检查
print("\n【环境变量】")
env_vars = [
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
]

for var in env_vars:
    value = os.environ.get(var, '未设置')
    print(f"  {var}: {value}")

# 6. 多进程测试
print("\n【多进程测试】")
print(f"multiprocessing.cpu_count(): {mp.cpu_count()}")
print(f"multiprocessing start method: {mp.get_start_method()}")

def worker_test(i):
    """测试工作进程"""
    import os
    pid = os.getpid()
    return f"Worker {i} PID: {pid}"

try:
    with mp.Pool(processes=4) as pool:
        results = pool.map(worker_test, range(4))
    print("✓ 多进程创建成功:")
    for r in results:
        print(f"  {r}")
except Exception as e:
    print(f"✗ 多进程创建失败: {e}")

# 7. 建议配置
print("\n" + "=" * 80)
print("【推荐配置】")
print("=" * 80)

if cpu_count_logical >= 32:
    print(f"\n✓ 你有 {cpu_count_logical} 个逻辑 CPU，非常适合多进程训练！")
    print(f"\n推荐配置:")
    print(f"  num_envs = 16  # 使用 16 个并行环境")
    print(f"  num_threads = 32  # PyTorch 使用 32 个线程")
    print(f"  pin_cpu_affinity = True  # 启用 CPU 亲和度")
    print(f"\n命令:")
    print(f"  python -m mahjong_agent.train --config multithread --device cuda")
else:
    print(f"\n你有 {cpu_count_logical} 个逻辑 CPU")
    num_envs = max(2, cpu_count_logical // 4)
    print(f"\n推荐配置:")
    print(f"  num_envs = {num_envs}")
    print(f"  num_threads = {cpu_count_logical}")
    print(f"  pin_cpu_affinity = True")

# 8. 潜在问题检查
print("\n" + "=" * 80)
print("【潜在问题检查】")
print("=" * 80)

issues = []

try:
    import psutil
except ImportError:
    issues.append("❌ psutil 未安装 - 无法设置 CPU 亲和度")
    issues.append("   解决方法: pip install psutil")

for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    value = os.environ.get(var)
    if value and int(value) == 1:
        issues.append(f"⚠️  {var}={value} - 这会限制 NumPy/科学计算库只使用 1 个线程")
        issues.append(f"   解决方法: unset {var} 或设置为更大的值")

if issues:
    print("\n发现以下问题:")
    for issue in issues:
        print(issue)
else:
    print("\n✓ 未发现明显问题")

print("\n" + "=" * 80)

