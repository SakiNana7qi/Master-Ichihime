#!/usr/bin/env python
"""
多进程环境管理（SubprocVecEnv）

用于并行运行多个 MahjongEnv 实例以提升CPU利用率和采样效率。
"""

import os
import multiprocessing as mp
from typing import List, Tuple, Optional


def _env_worker(remote, parent_remote, seed: Optional[int]):
    parent_remote.close()
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
    def __init__(self, num_envs: int, base_seed: int = 42):
        self.num_envs = num_envs
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes: List[mp.Process] = []

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

        # 并发接收
        results = [r.recv() for r in self.remotes]
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
