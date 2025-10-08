#!/usr/bin/env python
"""Quick test"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mahjong_agent import MahjongTrainer, get_fast_config

print("Starting quick training test...")

config = get_fast_config()
config.rollout_steps = 128  # Very small for testing
config.total_timesteps = 500  # Just 500 steps
config.log_interval = 1
config.verbose = True

print(f"Config: {config.rollout_steps} rollout steps, {config.total_timesteps} total")

try:
    trainer = MahjongTrainer(config=config)
    print("Trainer initialized, starting training...")
    trainer.train()
    print("SUCCESS! Training completed without errors!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
