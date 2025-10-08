#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的训练启动脚本 - 避免编码问题
"""

import sys
import subprocess
import os


def check_dependencies():
    """检查依赖"""
    print("=" * 80)
    print(" " * 25 + "麻将AI训练 - 快速开始")
    print("=" * 80)
    print()

    print("检查依赖...")
    try:
        import torch
        import numpy
        import gymnasium

        print("✓ 依赖已安装")
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("正在安装依赖包...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "mahjong_agent/requirements.txt",
            ]
        )
    print()


def start_training():
    """开始训练"""
    print("开始训练（快速配置）...")
    print()

    # 运行训练命令
    cmd = [
        sys.executable,
        "-m",
        "mahjong_agent.train",
        "--config",
        "fast",
        "--device",
        "cuda",
        "--seed",
        "42",
    ]

    try:
        subprocess.run(cmd)

        print()
        print("=" * 80)
        print(" " * 30 + "训练完成！")
        print("=" * 80)
        print()
        print("检查点保存在: ./checkpoints/")
        print("日志保存在: ./logs/")
        print()
        print("使用以下命令查看训练曲线:")
        print("  tensorboard --logdir logs/")
        print()

    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    check_dependencies()
    start_training()

    # 不自动暂停，让用户自己决定
