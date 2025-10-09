# mahjong_agent/train.py
"""
主训练脚本 - 实现完整的PPO训练循环
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mahjong_environment import MahjongEnv
from mahjong_agent.vec_env import SubprocVecEnv
from mahjong_agent.model import MahjongActorCritic
from mahjong_agent.rollout_buffer import RolloutBuffer
from mahjong_agent.ppo_updater import PPOUpdater
from mahjong_agent.config import PPOConfig, get_default_config


class MahjongTrainer:
    """
    麻将AI训练器
    实现完整的PPO训练流程
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        初始化训练器

        Args:
            config: PPO配置，如果为None则使用默认配置
            checkpoint_path: 检查点路径，用于恢复训练
        """
        self.config = config or get_default_config()

        # 设置设备
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"使用设备: {self.device}")

        # CUDA 数值精度优化（加速矩阵乘）
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                print("已启用 TF32/高精度矩阵乘以提升吞吐")
            except Exception:
                pass

        # 设置 PyTorch 线程数和并行策略
        # 若指定 cpu_core_limit，则将 PyTorch 线程数限制到该值
        core_limit = getattr(self.config, "cpu_core_limit", None)
        default_threads = os.cpu_count() or 1
        num_threads = getattr(self.config, "num_threads", default_threads)
        if isinstance(core_limit, int) and core_limit > 0:
            num_threads = min(num_threads, core_limit)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(min(4, num_threads // 4))  # 减少线程竞争
        print(f"PyTorch 线程数: {num_threads} (inter-op: {torch.get_num_interop_threads()})")

        # 设置随机种子
        self._set_seed(self.config.seed)

        # 创建环境（支持并行）
        self.num_envs = max(1, getattr(self.config, "num_envs", 1))
        pin_affinity = getattr(self.config, "pin_cpu_affinity", False)
        
        print(f"环境配置: num_envs={self.num_envs}, pin_cpu_affinity={pin_affinity}")
        
        if self.num_envs == 1:
            print("使用单环境模式")
            self.env = MahjongEnv(
                render_mode="human" if self.config.render_training else None,
                seed=self.config.seed,
            )
            self.vec_env = None
        else:
            # 多进程环境（仅用于数据采样阶段）
            print(f"使用多进程环境模式：{self.num_envs} 个并行环境")
            self.vec_env = SubprocVecEnv(
                self.num_envs,
                base_seed=self.config.seed,
                pin_cpu_affinity=pin_affinity,
                cpu_core_limit=getattr(self.config, "cpu_core_limit", None),
                cores_per_proc=getattr(self.config, "cores_per_proc", None),
            )
            self.env = None
            print(f"多进程环境初始化完成")

        # 创建模型
        self.model = MahjongActorCritic(self.config).to(self.device)
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 创建PPO更新器
        self.ppo_updater = PPOUpdater(self.model, self.config)

        # 创建Rollout缓冲区
        self.rollout_buffer = RolloutBuffer(self.config, self.device)

        # 训练统计
        self.global_step = 0
        self.rollout_count = 0
        self.episode_count = 0

        # 创建保存目录
        os.makedirs(self.config.save_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.log_dir)
        self.metrics_log_path = os.path.join(
            self.config.log_dir, "realtime_metrics.jsonl"
        )

        # 加载检查点
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print("训练器初始化完成！")

    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _numpy_obs_to_torch(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """将numpy观测转换为torch张量"""
        torch_obs = {}
        for key, value in obs.items():
            if key != "action_mask":  # action_mask在后面单独处理
                torch_obs[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
        return torch_obs

    def collect_rollouts(self) -> Dict[str, float]:
        """
        收集一轮经验数据

        Returns:
            统计信息字典
        """
        self.model.eval()  # 设为评估模式
        self.rollout_buffer.reset()

        # 统计信息
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        # 重置环境
        if self.vec_env is not None:
            obs_list, current_agents = self.vec_env.reset()
            # 为所有子环境初始化episode统计
            epi_rewards = [0.0 for _ in range(self.num_envs)]
            epi_lengths = [0 for _ in range(self.num_envs)]

            # 预分配批次缓冲，避免循环内重复分配
            import numpy as _np
            first_keys = [k for k in obs_list[0].keys() if k != "action_mask"]
            np_obs_batch = {
                k: _np.zeros((self.num_envs,) + _np.asarray(obs_list[0][k]).shape, dtype=_np.asarray(obs_list[0][k]).dtype)
                for k in first_keys
            }
            action_masks_np = _np.zeros((self.num_envs,) + _np.asarray(obs_list[0]["action_mask"]).shape, dtype=_np.asarray(obs_list[0]["action_mask"]).dtype)

            # 预分配GPU批次张量
            torch_obs_batch = {
                k: torch.zeros_like(torch.from_numpy(v)).to(self.device)
                for k, v in np_obs_batch.items()
            }
            torch_action_mask = torch.zeros_like(torch.from_numpy(action_masks_np)).to(self.device)
        else:
            obs, info = self.env.reset(seed=self.config.seed + self.global_step)

        import time as _t
        prof = getattr(self, "_prof", {"env_step": 0.0, "model_infer": 0.0, "ipc": 0.0})
        loop_start = _t.time()

        # 收集rollout_steps步的数据
        for step in range(self.config.rollout_steps):
            # 检查游戏是否结束
            if self.vec_env is None and self.env.agent_selection is None:
                # 保存episode统计
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)

                # 重置环境
                obs, info = self.env.reset()
                current_episode_reward = 0
                current_episode_length = 0
                self.episode_count += 1

            if self.vec_env is not None:
                # 并行分支：构造批次推理，并将每个子环境样本写入buffer
                # 填充预分配的批次缓冲
                for i in range(self.num_envs):
                    o = obs_list[i]
                    for k in np_obs_batch:
                        _np.copyto(np_obs_batch[k][i], _np.asarray(o[k]))
                    _np.copyto(action_masks_np[i], _np.asarray(o["action_mask"]))

                # 将CPU批次拷贝到GPU张量（非阻塞）
                for k in torch_obs_batch:
                    torch_obs_batch[k].copy_(torch.from_numpy(np_obs_batch[k]), non_blocking=True)
                torch_action_mask.copy_(torch.from_numpy(action_masks_np), non_blocking=True)

                t0 = _t.time()
                # 使用 bfloat16 自动混合精度以提升前向速度（4090 友好）
                with torch.inference_mode():
                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if self.device.type == "cuda"
                        else torch.autocast(enabled=False)
                    )
                    with amp_ctx:
                        actions_t, log_probs_t, _, values_t = (
                            self.model.get_action_and_value(
                                torch_obs_batch, action_mask=torch_action_mask
                            )
                        )
                t1 = _t.time()
                # 将少量必要数据搬回CPU并转为可支持的精度
                actions = actions_t.detach().to(dtype=torch.int64, device="cpu").numpy()  # (N,)
                log_probs = log_probs_t.detach().to(dtype=torch.float32, device="cpu").numpy()  # (N,)
                values = values_t.detach().to(dtype=torch.float32, device="cpu").numpy()  # (N,)

                # 步进所有环境
                t2 = _t.time()
                results = self.vec_env.step([int(a) for a in actions])
                t3 = _t.time()

                # 聚合本步并行样本后，批量写入缓冲区
                rewards_np = _np.array([float(r[2]) for r in results], dtype=_np.float32)
                dones_np = _np.array([bool(r[3]) for r in results], dtype=_np.bool_)
                obs_batch_np = {k: np_obs_batch[k].copy() for k in np_obs_batch}
                self.rollout_buffer.add_batch(
                    obs_batch=obs_batch_np,
                    actions=actions.astype(_np.int64, copy=False),
                    log_probs=log_probs.astype(_np.float32, copy=False),
                    rewards=rewards_np,
                    values=values.astype(_np.float32, copy=False),
                    dones=dones_np.astype(_np.float32, copy=False),
                    action_masks=action_masks_np.astype(_np.float32, copy=False),
                )

                # 统计与下一步观测
                new_obs_list = [None] * self.num_envs
                for i in range(self.num_envs):
                    next_obs, next_agent, reward, done, next_mask = results[i]
                    epi_rewards[i] += float(reward)
                    epi_lengths[i] += 1
                    if next_obs is None or next_agent is None:
                        episode_rewards.append(epi_rewards[i])
                        episode_lengths.append(epi_lengths[i])
                        epi_rewards[i] = 0.0
                        epi_lengths[i] = 0
                        next_obs, _ = self.vec_env.reset_one(i)
                    new_obs_list[i] = next_obs

                obs_list = new_obs_list
                # 并行分支：本步已完成写入与推进，继续下一步
                self.global_step += self.num_envs
                # 记录时间（仅在开启profile时）
                if getattr(self.config, "profile_timing", False):
                    prof["model_infer"] += (t1 - t0)
                    prof["env_step"] += (t3 - t2)
                    prof["ipc"] += max(0.0, (t2 - t1))
                # 跳过单环境下方的存储与推进逻辑
                continue
            else:
                # 单环境
                current_agent = self.env.agent_selection
                action_mask = obs["action_mask"]
                torch_obs = self._numpy_obs_to_torch(obs)
                torch_action_mask = (
                    torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                )
                with torch.no_grad():
                    action, log_prob, _, value = self.model.get_action_and_value(
                        torch_obs, action_mask=torch_action_mask
                    )

            # 转换为Python标量（确保类型正确）
            action_np = int(action.cpu().item())
            log_prob_np = float(log_prob.cpu().item())
            value_np = float(value.squeeze().cpu().item())

            # 执行动作
            if self.vec_env is None:
                self.env.step(action_np)

            # 获取奖励和终止状态
            if self.vec_env is None:
                reward = self.env.rewards.get(current_agent, 0.0)
                done = self.env.terminations.get(current_agent, False)

            # 存储到缓冲区
            self.rollout_buffer.add(
                obs=obs,
                action=action_np,
                log_prob=log_prob_np,
                reward=reward,
                value=value_np,
                done=done,
                action_mask=action_mask,
            )

            # 更新统计
            current_episode_reward += reward
            current_episode_length += 1
            self.global_step += 1

            # 获取下一个观测
            if self.vec_env is None and self.env.agent_selection is not None:
                obs = self.env.observe(self.env.agent_selection)

            # 渲染（如果需要）
            if self.config.render_training and step % 50 == 0:
                self.env.render()

        # 计算最后一个状态的价值（用于bootstrap）
        if self.vec_env is None and self.env.agent_selection is not None:
            obs = self.env.observe(self.env.agent_selection)
            torch_obs = self._numpy_obs_to_torch(obs)
            with torch.no_grad():
                last_value = self.model.get_value(torch_obs).cpu().item()
            last_done = self.env.terminations.get(self.env.agent_selection, False)
        else:
            last_value = 0.0
            last_done = True

        # 计算优势和回报
        self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

        # 返回统计信息
        stats = {
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "num_episodes": len(episode_rewards),
        }

        # 输出剖析信息
        if getattr(self.config, "profile_timing", False) and self.vec_env is not None:
            total = max(1e-9, ( _t.time() - loop_start))
            infer = prof["model_infer"]
            envs = prof["env_step"]
            ipc = prof["ipc"]
            other = max(0.0, total - infer - envs - ipc)
            print(
                f"[Profile] infer={infer:.2f}s ({infer/total:.0%}) env_step={envs:.2f}s ({envs/total:.0%}) ipc={ipc:.2f}s ({ipc/total:.0%}) other={other:.2f}s ({other/total:.0%})",
                flush=True,
            )
            self._prof = {"env_step": 0.0, "model_infer": 0.0, "ipc": 0.0}

        return stats

    def train_step(self) -> Dict[str, float]:
        """
        执行一次训练步骤（收集数据 + 更新策略）

        Returns:
            训练统计信息
        """
        # 收集数据
        rollout_stats = self.collect_rollouts()

        # 更新策略
        self.model.train()  # 设为训练模式
        update_stats = self.ppo_updater.update(self.rollout_buffer)

        # 合并统计信息
        stats = {**rollout_stats, **update_stats}

        return stats

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 80)
        print(" " * 30 + "开始训练")
        print("=" * 80)
        total_updates = self.config.total_timesteps // self.config.rollout_steps
        print(f"总步数: {self.config.total_timesteps:,}")
        print(f"Rollout步数: {self.config.rollout_steps}")
        print(f"预计迭代次数: {total_updates}")
        print("=" * 80 + "\n")

        start_time = time.time()

        # 主训练循环（带进度条）
        with tqdm(total=total_updates, desc="Training", unit="update") as pbar:
            while self.global_step < self.config.total_timesteps:
                self.rollout_count += 1

                # 执行一次训练步骤
                stats = self.train_step()

                # 更新进度条状态
                pbar.set_postfix(
                    steps=f"{self.global_step:,}",
                    ret=f"{stats.get('mean_episode_reward', 0):.2f}",
                    pol=f"{stats.get('policy_loss', 0):.3f}",
                    val=f"{stats.get('value_loss', 0):.3f}",
                    ent=f"{stats.get('entropy', 0):.3f}",
                    kl=f"{stats.get('approx_kl', 0):.3f}",
                )
                pbar.update(1)

                # 记录日志
                if self.rollout_count % self.config.log_interval == 0:
                    self._log_stats(stats, start_time)

                # 保存模型
                if self.rollout_count % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.rollout_count}.pt")
                    print(f"[OK] Checkpoint saved: checkpoint_{self.rollout_count}.pt")

                # 评估
                if self.rollout_count % self.config.eval_interval == 0:
                    eval_stats = self.evaluate(num_episodes=5)
                    self._log_eval_stats(eval_stats)

        # 训练结束
        print("\n" + "=" * 80)
        print(" " * 30 + "训练完成")
        print("=" * 80)
        print(f"总步数: {self.global_step:,}")
        print(f"总时间: {(time.time() - start_time) / 3600:.2f} 小时")
        print("=" * 80 + "\n")

        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        print("[OK] Final model saved")

        self.writer.close()

    def _log_stats(self, stats: Dict[str, float], start_time: float):
        """记录训练统计信息"""
        elapsed_time = time.time() - start_time
        fps = self.global_step / elapsed_time

        # 控制台输出
        if self.config.verbose:
            print(f"\n【Rollout {self.rollout_count}】 步数: {self.global_step:,}")
            print(f"  平均回报: {stats.get('mean_episode_reward', 0):.3f}")
            print(f"  平均长度: {stats.get('mean_episode_length', 0):.1f}")
            print(f"  策略损失: {stats.get('policy_loss', 0):.4f}")
            print(f"  价值损失: {stats.get('value_loss', 0):.4f}")
            print(f"  熵: {stats.get('entropy', 0):.4f}")
            print(f"  裁剪比例: {stats.get('clip_fraction', 0):.3f}")
            print(f"  近似KL: {stats.get('approx_kl', 0):.4f}")
            print(f"  学习率: {stats.get('learning_rate', 0):.6f}")
            print(f"  FPS: {fps:.1f}")

        # TensorBoard
        for key, value in stats.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)
        self.writer.add_scalar("train/fps", fps, self.global_step)

        # 写入实时JSONL供GUI读取
        payload = {
            "type": "train",
            "time": time.time(),
            "global_step": int(self.global_step),
            "rollout": int(self.rollout_count),
            "fps": float(fps),
            "clip_range": float(getattr(self.config, "clip_range", 0.0)),
        }
        # 将数值型stat写入（忽略不可转float的值）
        for k, v in stats.items():
            try:
                payload[k] = float(v)
            except Exception:
                pass
        try:
            with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_eval_stats(self, stats: Dict[str, float]):
        """记录评估统计信息"""
        if self.config.verbose:
            print(f"\n【评估】")
            print(f"  平均回报: {stats.get('mean_reward', 0):.3f}")
            print(f"  平均长度: {stats.get('mean_length', 0):.1f}")
            print(f"  胜率: {stats.get('win_rate', 0):.2%}")

        for key, value in stats.items():
            self.writer.add_scalar(f"eval/{key}", value, self.global_step)

        # 写入实时JSONL供GUI读取
        payload = {
            "type": "eval",
            "time": time.time(),
            "global_step": int(self.global_step),
            "rollout": int(self.rollout_count),
        }
        for k, v in stats.items():
            try:
                payload[k] = float(v)
            except Exception:
                pass
        try:
            with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估当前策略

        Args:
            num_episodes: 评估的局数

        Returns:
            评估统计信息
        """
        self.model.eval()

        # 在并行环境模式下，self.env 为 None，这里创建临时单环境用于评估
        eval_env = self.env if self.env is not None else MahjongEnv(seed=self.config.seed)

        episode_rewards = []
        episode_lengths = []
        wins = 0

        for _ in range(num_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            max_steps = 2000  # 安全上限，避免环境异常导致死循环

            while not done and episode_length < max_steps:
                # 终局保护：有些环境在结束时不会立刻将 agent_selection 置为 None
                if getattr(eval_env, "game_state", None) is not None:
                    if getattr(eval_env.game_state, "phase", None) == "end":
                        break
                if eval_env.agent_selection is None:
                    break

                current_agent = eval_env.agent_selection
                action_mask = obs["action_mask"]

                # 使用确定性策略
                torch_obs = self._numpy_obs_to_torch(obs)
                torch_action_mask = (
                    torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    action, _, _, _ = self.model.get_action_and_value(
                        torch_obs, action_mask=torch_action_mask, deterministic=True
                    )

                eval_env.step(action.cpu().item())

                reward = eval_env.rewards.get(current_agent, 0.0)
                done = eval_env.terminations.get(current_agent, False)

                episode_reward += reward
                episode_length += 1

                if eval_env.agent_selection is not None:
                    obs = eval_env.observe(eval_env.agent_selection)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 检查是否获胜（player_0的奖励最高）
            if eval_env.rewards.get("player_0", 0) > 0:
                wins += 1

        return {
            "mean_reward": np.mean(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "win_rate": wins / num_episodes,
        }

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = os.path.join(self.config.save_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.ppo_updater.optimizer.state_dict(),
                "global_step": self.global_step,
                "rollout_count": self.rollout_count,
                "episode_count": self.episode_count,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ppo_updater.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.rollout_count = checkpoint["rollout_count"]
        self.episode_count = checkpoint["episode_count"]
        print(f"已加载检查点: {path}")
        print(f"  全局步数: {self.global_step}")
        print(f"  Rollout次数: {self.rollout_count}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="训练麻将AI")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast", "high_performance", "multithread"],
        help="配置类型",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="检查点路径（恢复训练）"
    )
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--cpu-core-limit", type=int, default=0, help="限制可用CPU核心数（0表示不限制）")
    parser.add_argument("--num-envs", type=int, default=0, help="并行环境数（0表示使用配置默认值）")
    parser.add_argument("--profile", action="store_true", help="输出训练阶段耗时剖析")

    args = parser.parse_args()

    # 加载配置
    if args.config == "default":
        from mahjong_agent.config import get_default_config

        config = get_default_config()
    elif args.config == "fast":
        from mahjong_agent.config import get_fast_config

        config = get_fast_config()
    elif args.config == "high_performance":
        from mahjong_agent.config import get_high_performance_config

        config = get_high_performance_config()
    elif args.config == "multithread":
        from mahjong_agent.config_multithread import get_multithread_config

        config = get_multithread_config()
        # 如命令行提供限制与环境数，则在下方统一覆盖
        config.pin_cpu_affinity = True

    # 覆盖配置
    config.device = args.device
    config.seed = args.seed
    # 覆盖 CPU 限制与并行环境数
    if args.cpu_core_limit and args.cpu_core_limit > 0:
        config.cpu_core_limit = args.cpu_core_limit
        # 将 PyTorch 线程数也限制到该值
        config.num_threads = min(getattr(config, "num_threads", os.cpu_count() or 1), args.cpu_core_limit)
    if args.num_envs and args.num_envs > 0:
        config.num_envs = args.num_envs
    if args.profile:
        config.profile_timing = True

    # 创建训练器并开始训练
    trainer = MahjongTrainer(config=config, checkpoint_path=args.checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
