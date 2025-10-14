#!/usr/bin/env python
"""
多进程环境管理（SubprocVecEnv）

用于并行运行多个 MahjongEnv 实例以提升CPU利用率和采样效率。
"""

import os
import multiprocessing as mp
import platform
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

import numpy as np
try:
    from multiprocessing.shared_memory import SharedMemory
except Exception:
    SharedMemory = None  # 兼容性保护

# 仅用于类型标注的共享内存类型别名，避免运行时变量用于类型表达式
if TYPE_CHECKING:
    from multiprocessing.shared_memory import SharedMemory as SharedMemoryT
else:
    class SharedMemoryT:  # type: ignore
        pass


def _env_worker(remote, parent_remote, seed: Optional[int], use_shm: bool,
                shm_specs: Optional[Dict[str, Any]]):
    """环境工作进程函数"""
    parent_remote.close()

    # 设置子进程的线程数限制，避免过度竞争
    # 每个子进程只使用少量线程
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # 如果 NumPy 支持线程控制，限制为 1
    try:
        import numpy

        if hasattr(numpy, "__config__") and hasattr(numpy.__config__, "show"):
            pass  # NumPy 配置检查
    except:
        pass

    from mahjong_environment import MahjongEnv
    from mahjong_environment.utils.action_encoder import ActionEncoder

    env = MahjongEnv(seed=seed)

    # 共享内存映射（仅当启用时）
    i8_arr = None
    f32_arr = None
    layout = None
    if use_shm and shm_specs is not None and SharedMemory is not None:
        try:
            shm_i8 = SharedMemory(name=shm_specs["i8_name"])  # bytes
            shm_f32 = SharedMemory(name=shm_specs["f32_name"])  # bytes
            i8_size = int(shm_specs["i8_size"])  # elements (int8)
            f32_size = int(shm_specs["f32_size"])  # elements (float32)
            i8_arr = np.ndarray((i8_size,), dtype=np.int8, buffer=shm_i8.buf)
            f32_arr = np.ndarray((f32_size,), dtype=np.float32, buffer=shm_f32.buf)
            layout = shm_specs.get("layout", {})
        except Exception:
            i8_arr = None
            f32_arr = None
            layout = None

    def _write_obs_to_shm(obs: Dict[str, np.ndarray]):
        if i8_arr is None or f32_arr is None or layout is None:
            return False
        try:
            # int8 连续段
            off = 0
            hand = obs["hand"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+34] = hand; off += 34
            drawn = obs["drawn_tile"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+34] = drawn; off += 34
            rivers = obs["rivers"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+136] = rivers; off += 136
            melds = obs["melds"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+136] = melds; off += 136
            riichi = obs["riichi_status"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+4] = riichi; off += 4
            dora = obs["dora_indicators"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+170] = dora; off += 170
            phase = obs["phase_info"].astype(np.int8, copy=False).reshape(-1)
            i8_arr[off:off+3] = phase; off += 3
            mask = obs["action_mask"].astype(np.int8, copy=False).reshape(-1)
            act_dim = int(layout.get("action_dim", mask.shape[0]))
            i8_arr[off:off+act_dim] = mask[:act_dim]

            # float32 连续段
            foff = 0
            scores = obs["scores"].astype(np.float32, copy=False).reshape(-1)
            f32_arr[foff:foff+4] = scores; foff += 4
            ginfo = obs["game_info"].astype(np.float32, copy=False).reshape(-1)
            f32_arr[foff:foff+5] = ginfo; foff += 5
            return True
        except Exception:
            return False

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(seed=data)
                if use_shm and _write_obs_to_shm(obs):
                    remote.send((env.agent_selection,))
                else:
                    remote.send((obs, env.agent_selection))
            elif cmd == "step":
                action = data
                acting_agent = env.agent_selection
                env.step(action)

                # 奖励/终止针对执行动作的智能体
                reward = env.rewards.get(acting_agent, 0.0)
                done = env.terminations.get(acting_agent, False)

                if env.agent_selection is None:
                    if use_shm:
                        remote.send((None, None, reward, done))
                    else:
                        remote.send((None, None, reward, done, None))
                else:
                    next_obs = env.observe(env.agent_selection)
                    if use_shm and _write_obs_to_shm(next_obs):
                        remote.send((None, env.agent_selection, reward, done))
                    else:
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
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        # 主进程退出或管道被关闭时，子进程优雅退出
        try:
            remote.close()
        except Exception:
            pass
        pass


class SubprocVecEnv:
    def __init__(
        self, num_envs: int, base_seed: int = 42, pin_cpu_affinity: bool = False,
        cpu_core_limit: int | None = None, cores_per_proc: int | None = None,
        use_shared_memory: bool = False,
    ):
        self.num_envs = num_envs
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes: List[mp.Process] = []
        self.pin_cpu_affinity = pin_cpu_affinity
        self.cpu_core_limit = cpu_core_limit
        self.cores_per_proc = cores_per_proc
        self.use_shared_memory = bool(use_shared_memory and SharedMemory is not None)

        # 共享内存布局：
        # int8 段: hand(34) + drawn(34) + rivers(4*34=136) + melds(136) + riichi(4) + dora(5*34=170) + phase(3) + mask(112)
        # float32 段: scores(4) + game_info(5)
        self.action_dim = 112
        self.i8_size = 34 + 34 + 136 + 136 + 4 + 170 + 3 + self.action_dim
        self.f32_size = 4 + 5
        self.shm_i8: List[SharedMemoryT] = []
        self.shm_f32: List[SharedMemoryT] = []
        self._i8_views: List[np.ndarray] = []
        self._f32_views: List[np.ndarray] = []

        for idx, (work_remote, remote) in enumerate(
            zip(self.work_remotes, self.remotes)
        ):
            shm_specs = None
            if self.use_shared_memory:
                name_i8 = f"mj_i8_{os.getpid()}_{idx}"
                name_f32 = f"mj_f32_{os.getpid()}_{idx}"
                shm_i8 = SharedMemory(create=True, size=self.i8_size, name=name_i8)
                shm_f32 = SharedMemory(create=True, size=self.f32_size * 4, name=name_f32)
                self.shm_i8.append(shm_i8)
                self.shm_f32.append(shm_f32)
                self._i8_views.append(np.ndarray((self.i8_size,), dtype=np.int8, buffer=shm_i8.buf))
                self._f32_views.append(np.ndarray((self.f32_size,), dtype=np.float32, buffer=shm_f32.buf))
                shm_specs = {
                    "i8_name": name_i8,
                    "f32_name": name_f32,
                    "i8_size": self.i8_size,
                    "f32_size": self.f32_size,
                    "layout": {"action_dim": self.action_dim},
                }
            p = mp.Process(
                target=_env_worker, args=(work_remote, remote, base_seed + idx, self.use_shared_memory, shm_specs)
            )
            # 使用非守护进程，配合显式 close() 与 join() 以避免 Windows 下退出时的 BrokenPipe 噪声
            p.daemon = False
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
        for i, r in enumerate(self.remotes):
            msg = r.recv()
            if self.use_shared_memory:
                # 共享内存模式下，reset 返回 (current_agent,)
                current_agent = msg[0]
                current_agents.append(current_agent)
                # 从共享内存恢复观测为字典
                obs_list.append(self._read_obs_from_shm(i))
            else:
                obs, current_agent = msg
                obs_list.append(obs)
                current_agents.append(current_agent)
        return obs_list, current_agents

    def reset_one(self, idx: int, seed: Optional[int] = None):
        """重置单个子环境，返回(obs, current_agent)"""
        self.remotes[idx].send(("reset", seed))
        msg = self.remotes[idx].recv()
        if self.use_shared_memory:
            # 共享内存模式下，msg = (current_agent,)
            current_agent = msg[0]
            obs = self._read_obs_from_shm(idx)
            return (obs, current_agent)
        else:
            return msg

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
                    msg = r.recv()
                    if self.use_shared_memory:
                        # 共享内存：worker 在非终止步发送 (None, next_agent, reward, done)
                        # 在终止步发送 (None, None, reward, done)
                        if len(msg) == 4:
                            _placeholder, next_agent, reward, done = msg
                            if next_agent is None:
                                next_obs = None
                            else:
                                next_obs = self._read_obs_from_shm(i)
                            results[i] = (next_obs, next_agent, reward, done, None)
                        else:
                            # 兼容完整返回（非共享内存格式）
                            results[i] = msg
                    else:
                        results[i] = msg
                    done_now.append(i)
            for i in done_now:
                pending.discard(i)
        # 每个元素: (next_obs, next_agent, reward, done, next_action_mask)
        return results

    def _read_obs_from_shm(self, idx: int) -> Dict[str, np.ndarray]:
        # 将共享内存中的连续段还原为 dict（返回视图，避免复制）
        i8 = self._i8_views[idx]
        f32 = self._f32_views[idx]
        off = 0
        hand = i8[off:off+34]; off += 34
        drawn = i8[off:off+34]; off += 34
        rivers = i8[off:off+136].reshape(4, 34); off += 136
        melds = i8[off:off+136].reshape(4, 34); off += 136
        riichi = i8[off:off+4]; off += 4
        dora = i8[off:off+170].reshape(5, 34); off += 170
        phase = i8[off:off+3]; off += 3
        mask = i8[off:off+self.action_dim]

        foff = 0
        scores = f32[foff:foff+4]; foff += 4
        ginfo = f32[foff:foff+5]; foff += 5

        return {
            "hand": hand,
            "drawn_tile": drawn,
            "rivers": rivers,
            "melds": melds,
            "riichi_status": riichi,
            "scores": scores,
            "dora_indicators": dora,
            "game_info": ginfo,
            "phase_info": phase,
            "action_mask": mask,
        }

    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            r.send(("close", None))
        # 给予子进程更充足的时间退出
        for p in self.processes:
            p.join(timeout=5)
        # 若仍有存活的子进程，强制终止
        for p in self.processes:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        # 释放共享内存
        if self.use_shared_memory:
            try:
                for shm in self.shm_i8:
                    shm.close(); shm.unlink()
                for shm in self.shm_f32:
                    shm.close(); shm.unlink()
            except Exception:
                pass
        self.closed = True
