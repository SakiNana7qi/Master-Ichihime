#!/usr/bin/env python
"""
多线程训练启动脚本 (跨平台)

功能:
1. 自动检测 CPU 核心数
2. 设置环境变量以启用多线程
3. 检查依赖并给出建议
4. 启动训练
"""

import os
import sys
import platform
import subprocess
import multiprocessing as mp

def print_header(text):
    """打印标题"""
    print("=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_section(text):
    """打印小节"""
    print(f"\n{text}")
    print("-" * 40)

def setup_environment(limit_cores: int | None = None):
    """设置环境变量"""
    print_header("麻将 AI 多线程训练启动脚本")
    
    # 检测 CPU 核心数
    detected = mp.cpu_count()
    cpu_count = detected if (not limit_cores or limit_cores <= 0) else min(limit_cores, detected)
    print(f"\n检测到 CPU 核心数: {cpu_count}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python 版本: {sys.version.split()[0]}")
    
    # 设置环境变量
    print_section("设置环境变量")
    
    env_vars = {
        'OMP_NUM_THREADS': str(cpu_count),
        'MKL_NUM_THREADS': str(cpu_count),
        'OPENBLAS_NUM_THREADS': str(cpu_count),
        'NUMEXPR_NUM_THREADS': str(cpu_count),
        'VECLIB_MAXIMUM_THREADS': str(cpu_count),
        'OMP_WAIT_POLICY': 'PASSIVE',
        'KMP_BLOCKTIME': '0',
        'OMP_PROC_BIND': 'spread',
        'OMP_PLACES': 'threads',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    return cpu_count

def check_dependencies():
    """检查依赖"""
    print_section("检查依赖")
    
    issues = []
    
    # 检查 PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        issues.append("❌ PyTorch 未安装")
        issues.append("   pip install torch")
    
    # 检查 psutil
    try:
        import psutil
        print(f"✓ psutil {psutil.__version__}")
    except ImportError:
        issues.append("⚠️  psutil 未安装 - 无法优化 CPU 亲和度")
        issues.append("   pip install psutil")
    
    # 检查环境模块
    try:
        from mahjong_environment import MahjongEnv
        print(f"✓ mahjong_environment")
    except ImportError:
        issues.append("❌ mahjong_environment 未安装")
    
    # 检查训练模块
    try:
        from mahjong_agent.train import MahjongTrainer
        print(f"✓ mahjong_agent")
    except ImportError as e:
        issues.append(f"❌ mahjong_agent 导入失败: {e}")
    
    if issues:
        print("\n发现问题:")
        for issue in issues:
            print(issue)
        
        response = input("\n是否继续? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    return len(issues) == 0

def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-cores', type=int, default=0, help='限制可用CPU核心数（0表示不限制）')
    parser.add_argument('--num-envs', type=int, default=0, help='并行环境数（0表示根据CPU自动设置）')
    parser.add_argument('--cores-per-proc', type=int, default=0, help='每个子进程绑定的核心数（0表示自动）')
    parser.add_argument('--profile', action='store_true', help='输出训练阶段耗时剖析')
    args = parser.parse_args()

    # 设置环境
    cpu_count = setup_environment(limit_cores=args.limit_cores)
    
    # 检查依赖
    all_ok = check_dependencies()
    
    # 启动训练
    print_section("启动训练")
    
    print("\n配置:")
    print(f"  config: multithread")
    print(f"  device: cuda")
    # 如果限制核心，则默认环境数为核心数的一半（每环境约2个核心）
    default_envs = 16 if args.limit_cores == 0 else max(4, min(32, (args.limit_cores // max(1, args.cores_per_proc)) if args.cores_per_proc > 0 else (args.limit_cores // 2)))
    desired_envs = default_envs if args.num_envs <= 0 else args.num_envs
    print(f"  num_envs: {desired_envs}")
    print(f"  num_threads: {cpu_count}")
    print(f"  pin_cpu_affinity: True")
    print()
    
    # 导入并启动
    try:
        from mahjong_agent.config_multithread import get_multithread_config
        from mahjong_agent.train import MahjongTrainer
        
        # 获取配置
        config = get_multithread_config()
        config.num_envs = min(desired_envs, max(2, cpu_count // 2))
        config.num_threads = cpu_count
        # 将核心限制下发到训练与亲和度
        config.cpu_core_limit = 0 if args.limit_cores <= 0 else args.limit_cores
        config.cores_per_proc = None if args.cores_per_proc <= 0 else args.cores_per_proc
        config.pin_cpu_affinity = True
        config.device = "cuda"
        config.seed = 42
        config.profile_timing = args.profile
        
        print(f"实际使用配置:")
        print(f"  num_envs: {config.num_envs}")
        print(f"  num_threads: {config.num_threads}")
        print(f"  cpu_core_limit: {config.cpu_core_limit}")
        print(f"  cores_per_proc: {config.cores_per_proc}")
        print(f"  profile_timing: {config.profile_timing}")
        print()
        
        print_header("开始训练")
        print()
        
        # 创建训练器
        trainer = MahjongTrainer(config=config, checkpoint_path=None)
        
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n训练完成")

if __name__ == "__main__":
    main()


