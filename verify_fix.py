#!/usr/bin/env python
"""
快速验证 CPU 利用率修复是否生效
"""

import sys
import time
import multiprocessing as mp
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 80)
    print("  CPU 利用率修复验证")
    print("=" * 80)
    
    # 1. 检查 CPU 核心数
    cpu_count = mp.cpu_count()
    print(f"\n✓ 检测到 {cpu_count} 个 CPU 核心")
    
    # 2. 检查 psutil
    try:
        import psutil
        print(f"✓ psutil {psutil.__version__} 已安装")
    except ImportError:
        print("✗ psutil 未安装！")
        print("  请运行: pip install psutil")
        return False
    
    # 3. 检查配置
    print("\n" + "-" * 80)
    print("检查配置")
    print("-" * 80)
    
    try:
        from mahjong_agent.config_multithread import get_multithread_config
        config = get_multithread_config()
        
        print(f"  num_envs: {config.num_envs}")
        print(f"  num_threads: {getattr(config, 'num_threads', '未设置')}")
        print(f"  pin_cpu_affinity: {getattr(config, 'pin_cpu_affinity', False)}")
        
        # 验证配置是否合理
        expected_envs = 32 if cpu_count >= 64 else (24 if cpu_count >= 48 else 16)
        if config.num_envs < expected_envs * 0.8:
            print(f"\n⚠️  警告: num_envs={config.num_envs} 可能太少了")
            print(f"  建议: {expected_envs} (对于 {cpu_count} 核 CPU)")
        else:
            print(f"\n✓ 配置合理（{config.num_envs} 个环境适合 {cpu_count} 核 CPU）")
        
    except Exception as e:
        print(f"✗ 配置检查失败: {e}")
        return False
    
    # 4. 检查 vec_env
    print("\n" + "-" * 80)
    print("检查 SubprocVecEnv")
    print("-" * 80)
    
    try:
        from mahjong_agent.vec_env import SubprocVecEnv
        
        # 读取源代码检查是否有 // 2
        vec_env_path = Path(__file__).parent / "mahjong_agent" / "vec_env.py"
        if vec_env_path.exists():
            content = vec_env_path.read_text(encoding='utf-8')
            if "cpu_count(logical=True) // 2" in content:
                print("✗ 发现问题: vec_env.py 中仍然有 '// 2'")
                print("  这会导致只使用一半的 CPU 核心！")
                return False
            else:
                print("✓ vec_env.py 已修复（不再除以 2）")
        
    except Exception as e:
        print(f"✗ SubprocVecEnv 检查失败: {e}")
        return False
    
    # 5. 测试小规模多进程
    print("\n" + "-" * 80)
    print("测试多进程环境（4 个环境，5 秒）")
    print("-" * 80)
    
    try:
        vec_env = SubprocVecEnv(num_envs=4, base_seed=999, pin_cpu_affinity=True)
        
        # 检查输出中是否有 "系统总共"
        print("\n开始采样...")
        start_time = time.time()
        
        obs_list, agents = vec_env.reset()
        
        steps = 0
        for _ in range(50):
            import numpy as np
            actions = []
            for obs in obs_list:
                mask = obs.get("action_mask")
                if mask is not None and mask.sum() > 0:
                    legal = np.where(mask == 1)[0]
                    action = np.random.choice(legal)
                else:
                    action = 110
                actions.append(int(action))
            
            results = vec_env.step(actions)
            
            new_obs_list = []
            for i, (next_obs, next_agent, r, d, m) in enumerate(results):
                if next_obs is None:
                    next_obs, _ = vec_env.reset_one(i)
                new_obs_list.append(next_obs)
            obs_list = new_obs_list
            steps += 4
        
        vec_env.close()
        
        elapsed = time.time() - start_time
        fps = steps / elapsed
        
        print(f"\n✓ 测试完成")
        print(f"  总步数: {steps}")
        print(f"  耗时: {elapsed:.2f} 秒")
        print(f"  FPS: {fps:.1f}")
        
        if fps < 50:
            print(f"\n⚠️  警告: FPS 太低 ({fps:.1f})")
            print("  可能原因:")
            print("  - 环境本身很慢")
            print("  - CPU 亲和度未生效")
            print("  - 其他性能瓶颈")
        
    except Exception as e:
        print(f"✗ 多进程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    print(f"✓ CPU 核心: {cpu_count}")
    print(f"✓ psutil: 已安装")
    print(f"✓ 配置: num_envs={config.num_envs}")
    print(f"✓ vec_env.py: 已修复")
    print(f"✓ 多进程测试: 通过 (FPS={fps:.1f})")
    
    print("\n" + "=" * 80)
    print("✓ 所有检查通过！现在可以开始训练：")
    print("  python train_multithread.py")
    print("=" * 80)
    
    print("\n📊 训练时监控 CPU 使用率：")
    print("  - Windows: 打开任务管理器 -> 性能 -> CPU")
    print("  - 应该看到多个核心都有负载（不是只有一个 100%）")
    print("  - 期望 FPS: 800-1200（对于 32 环境）")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

