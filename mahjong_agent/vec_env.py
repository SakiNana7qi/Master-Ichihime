#!/usr/bin/env python
"""
多进程环境管理（SubprocVecEnv）

用于并行运行多个 MahjongEnv 实例以提升CPU利用率和采样效率。
"""

import os
import multiprocessing as mp
import platform
from typing import List, Tuple, Optional


def _env_worker(remote, parent_remote, seed: Optional[int]):
    """环境工作进程函数"""
    parent_remote.close()

    # 设置子进程的线程数限制，避免过度竞争
    # 每个子进程只使用少量线程
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import numpy as np

    # 如果 NumPy 支持线程控制，限制为 1
    try:
        import numpy

        if hasattr(numpy, "__config__") and hasattr(numpy.__config__, "show"):
            pass  # NumPy 配置检查
    except:
        pass

    from mahjong_environment import MahjongEnv

    env = MahjongEnv(seed=seed)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                remote.send((obs, env.agent_selection))
            elif cmd == "step":
                action = data
                acting_agent = env.agent_selection
                env.step(action)

                # 奖励/终止针对执行动作的智能体
                reward = env.rewards.get(acting_agent, 0.0)
                done = env.terminations.get(acting_agent, False)

                if env.agent_selection is None:
                    remote.send((None, None, reward, done, None))
                else:
                    next_obs = env.observe(env.agent_selection)
                    remote.send(
                        (
                            next_obs,
                            env.agent_selection,
                            reward,
                            done,
                            next_obs.get("action_mask", None),
                        )
                    )
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except KeyboardInterrupt:
        pass


class SubprocVecEnv:
    def __init__(
        self, num_envs: int, base_seed: int = 42, pin_cpu_affinity: bool = False,
        cpu_core_limit: int | None = None, cores_per_proc: int | None = None,
    ):
        self.num_envs = num_envs
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes: List[mp.Process] = []
        self.pin_cpu_affinity = pin_cpu_affinity
        self.cpu_core_limit = cpu_core_limit
        self.cores_per_proc = cores_per_proc

        for idx, (work_remote, remote) in enumerate(
            zip(self.work_remotes, self.remotes)
        ):
            p = mp.Process(
                target=_env_worker, args=(work_remote, remote, base_seed + idx)
            )
            p.daemon = True
            p.start()
            work_remote.close()
            self.processes.append(p)

        # 可选：为子进程设置CPU亲和度（Windows/Linux），尽量分散到不同核
        if self.pin_cpu_affinity:
            self._pin_affinity()

    def _pin_affinity(self):
        try:
            import psutil  # type: ignore
        except ImportError as e:
            print(
                f"[警告] psutil 未安装，无法设置 CPU 亲和度。请安装: pip install psutil"
            )
            print(f"       这可能导致所有子进程运行在同一个 CPU 核心上，降低性能")
            return
        except Exception as e:
            print(f"[警告] psutil 导入失败: {e}")
            return

        try:
            # 使用限制后的逻辑核心数量（若提供）
            total_logical = psutil.cpu_count(logical=True)
            cpu_count = total_logical if (self.cpu_core_limit is None or self.cpu_core_limit <= 0) else min(self.cpu_core_limit, total_logical)
            if not cpu_count or cpu_count < 2:
                print(f"[警告] CPU 核心数不足: {cpu_count}")
                return

            note = "(受限)" if (self.cpu_core_limit is not None and self.cpu_core_limit > 0) else ""
            print(
                f"[信息] 为 {self.num_envs} 个子进程分配 CPU 亲和度 {note} (可用 {cpu_count}/{total_logical} 个逻辑核心)"
            )

            # 每个进程分配的核心数：优先使用外部指定，否则自动按比例分配
            cores_per_proc = self.cores_per_proc if (self.cores_per_proc and self.cores_per_proc > 0) else max(1, cpu_count // self.num_envs)

            for i, p in enumerate(self.processes):
                try:
                    proc = psutil.Process(p.pid)
                    # 分散到不同的核心上，使用循环分配
                    start = (i * cores_per_proc) % cpu_count
                    end = start + cores_per_proc
                    
                    # 处理跨边界的情况
                    if end <= cpu_count:
                        mask = list(range(start, end))
                    else:
                        # 跨边界：使用前面的核心
                        mask = list(range(start, cpu_count)) + list(range(0, end - cpu_count))

                    # 确保mask不为空
                    if not mask:
                        mask = [i % cpu_count]

                    proc.cpu_affinity(mask)
                    print(f"  进程 {i:2d} (PID {p.pid:5d}) -> CPU {mask}")
                except psutil.NoSuchProcess:
                    print(f"  [警告] 进程 {i} 不存在")
                except AttributeError:
                    print(f"  [警告] 当前平台不支持 CPU 亲和度设置")
                    break
                except Exception as e:
                    print(f"  [警告] 进程 {i} CPU 亲和度设置失败: {e}")
        except Exception as e:
            print(f"[警告] CPU 亲和度设置整体失败: {e}")

    def reset(self, seeds: Optional[List[int]] = None):
        obs_list = []
        current_agents = []
        for i, r in enumerate(self.remotes):
            r.send(("reset", None if seeds is None else seeds[i]))
        for r in self.remotes:
            obs, current_agent = r.recv()
            obs_list.append(obs)
            current_agents.append(current_agent)
        return obs_list, current_agents

    def reset_one(self, idx: int, seed: Optional[int] = None):
        """重置单个子环境，返回(obs, current_agent)"""
        self.remotes[idx].send(("reset", seed))
        return self.remotes[idx].recv()

    def step(self, actions: List[int]):
        # 并发发送
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))

        # 并发接收（先poll再recv，减少阻塞等待）
        results = [None] * self.num_envs
        pending = set(range(self.num_envs))
        # 简单轮询，避免单次阻塞在某个pipe
        while pending:
            done_now = []
            for i in list(pending):
                r = self.remotes[i]
                if r.poll(0.0):
                    results[i] = r.recv()
                    done_now.append(i)
            for i in done_now:
                pending.discard(i)
        # 每个元素: (next_obs, next_agent, reward, done, next_action_mask)
        return results

    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            r.send(("close", None))
        for p in self.processes:
            p.join(timeout=1)
        self.closed = True
