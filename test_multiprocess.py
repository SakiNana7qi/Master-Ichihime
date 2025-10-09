#!/usr/bin/env python
"""
测试多进程环境是否正常工作

这个脚本会：
1. 创建多个环境进程
2. 检查 CPU 亲和度设置
3. 监控 CPU 使用率
4. 验证并行采样
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_multiprocessing():
    """测试基本的多进程功能"""
    print("=" * 80)
    print("测试 1: 基本多进程功能")
    print("=" * 80)
    
    def worker(i):
        pid = os.getpid()
        time.sleep(0.1)
        return f"Worker {i}: PID {pid}"
    
    num_workers = min(8, mp.cpu_count())
    print(f"\n创建 {num_workers} 个工作进程...")
    
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker, range(num_workers))
        
        print("✓ 成功:")
        for r in results:
            print(f"  {r}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def test_vec_env():
    """测试向量化环境"""
    print("\n" + "=" * 80)
    print("测试 2: SubprocVecEnv")
    print("=" * 80)
    
    try:
        from mahjong_agent.vec_env import SubprocVecEnv
        
        num_envs = 4
        print(f"\n创建 {num_envs} 个并行环境...")
        
        vec_env = SubprocVecEnv(
            num_envs=num_envs,
            base_seed=42,
            pin_cpu_affinity=True
        )
        
        print("✓ 环境创建成功")
        
        # 测试 reset
        print("\n测试 reset...")
        obs_list, agents = vec_env.reset()
        print(f"✓ Reset 成功，获得 {len(obs_list)} 个观测")
        
        # 测试 step
        print("\n测试 step (执行 10 步)...")
        for i in range(10):
            # 为每个环境选择随机合法动作
            actions = []
            for obs in obs_list:
                mask = obs.get("action_mask", None)
                if mask is not None and mask.sum() > 0:
                    import numpy as np
                    legal_actions = np.where(mask == 1)[0]
                    action = np.random.choice(legal_actions)
                else:
                    action = 110  # SKIP
                actions.append(int(action))
            
            results = vec_env.step(actions)
            
            # 更新观测
            new_obs_list = []
            for j, (next_obs, next_agent, reward, done, mask) in enumerate(results):
                if next_obs is None or next_agent is None:
                    # 环境结束，重置
                    next_obs, _ = vec_env.reset_one(j)
                new_obs_list.append(next_obs)
            obs_list = new_obs_list
            
            if (i + 1) % 5 == 0:
                print(f"  步骤 {i+1} 完成")
        
        print("✓ Step 测试成功")
        
        # 关闭环境
        vec_env.close()
        print("✓ 环境关闭成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cpu_affinity():
    """测试 CPU 亲和度"""
    print("\n" + "=" * 80)
    print("测试 3: CPU 亲和度")
    print("=" * 80)
    
    try:
        import psutil
        print("✓ psutil 可用")
        
        # 获取当前进程的亲和度
        proc = psutil.Process()
        affinity = proc.cpu_affinity()
        print(f"\n主进程 CPU 亲和度: {affinity}")
        print(f"可用 CPU 核心数: {len(affinity)}")
        
        return True
        
    except ImportError:
        print("✗ psutil 未安装")
        print("  无法设置 CPU 亲和度，所有进程会竞争同一个核心")
        print("  建议: pip install psutil")
        return False
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False

def test_cpu_usage():
    """监控 CPU 使用率"""
    print("\n" + "=" * 80)
    print("测试 4: CPU 使用率监控")
    print("=" * 80)
    
    try:
        import psutil
        
        print("\n监控 5 秒钟的 CPU 使用率...")
        print("(在另一个终端运行训练脚本以看到效果)")
        
        for i in range(5):
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            avg_cpu = sum(cpu_percent) / len(cpu_percent)
            max_cpu = max(cpu_percent)
            
            # 统计高负载核心数
            busy_cores = sum(1 for x in cpu_percent if x > 50)
            
            print(f"\n第 {i+1} 秒:")
            print(f"  平均 CPU 使用率: {avg_cpu:.1f}%")
            print(f"  最高单核使用率: {max_cpu:.1f}%")
            print(f"  高负载核心数 (>50%): {busy_cores}/{len(cpu_percent)}")
            
            # 显示前 8 个核心的使用率
            print(f"  前 8 个核心: {[f'{x:.0f}%' for x in cpu_percent[:8]]}")
        
        return True
        
    except ImportError:
        print("✗ psutil 未安装，跳过")
        return False

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("多进程环境测试套件")
    print("=" * 80)
    
    results = {}
    
    # 测试 1: 基本多进程
    results['basic_mp'] = test_basic_multiprocessing()
    
    # 测试 2: 向量化环境
    results['vec_env'] = test_vec_env()
    
    # 测试 3: CPU 亲和度
    results['cpu_affinity'] = test_cpu_affinity()
    
    # 测试 4: CPU 使用率
    results['cpu_usage'] = test_cpu_usage()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！多进程环境配置正确。")
        print("\n可以开始训练:")
        print("  python train_multithread.py")
        print("或:")
        print("  python -m mahjong_agent.train --config multithread --device cuda")
    else:
        print("⚠️  部分测试失败")
        print("\n建议:")
        if not results.get('cpu_affinity'):
            print("  1. 安装 psutil: pip install psutil")
        if not results.get('vec_env'):
            print("  2. 检查环境模块是否正确安装")
        print("  3. 运行诊断脚本: python diagnose_cpu.py")
    
    print("=" * 80)

if __name__ == "__main__":
    main()


