# mahjong_agent/evaluate.py
"""
评估脚本 - 评估训练好的麻将AI
支持与随机策略、固定策略对战，并提供详细的性能分析
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mahjong_environment import MahjongEnv
from mahjong_agent.model import MahjongActorCritic
from mahjong_agent.config import PPOConfig


class MahjongEvaluator:
    """麻将AI评估器"""

    def __init__(
        self, model_path: str, config: Optional[PPOConfig] = None, device: str = "cuda"
    ):
        """
        初始化评估器

        Args:
            model_path: 模型权重路径
            config: PPO配置
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载模型（兼容 PyTorch 2.6 的 weights_only 安全默认）
        # 1) 首选显式允许 PPOConfig 的反序列化，并使用 weights_only=False 读取完整检查点
        # 2) 回退到 weights_only=True 仅加载 state_dict
        try:
            # 允许 PPOConfig 作为安全全局类型
            try:
                torch.serialization.add_safe_globals([PPOConfig])
            except Exception:
                pass

            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if config is None:
                config = checkpoint.get("config", PPOConfig())
            loaded_step = checkpoint.get("global_step", "N/A")
        except Exception:
            # 仅加载权重
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            checkpoint = {"model_state_dict": state_dict}
            if config is None:
                config = PPOConfig()
            loaded_step = "N/A"

        # 兼容 torch.compile 保存的 _orig_mod.* 键名：去前缀
        if isinstance(state_dict, dict):
            try:
                if any(isinstance(k, str) and k.startswith("_orig_mod.") for k in state_dict.keys()):
                    stripped = {}
                    for k, v in state_dict.items():
                        if isinstance(k, str) and k.startswith("_orig_mod."):
                            stripped[k[len("_orig_mod."):]] = v
                        else:
                            stripped[k] = v
                    state_dict = stripped
            except Exception:
                pass

        self.config = config
        self.model = MahjongActorCritic(config).to(self.device)
        # 宽松加载以兼容不同版本/编译状态
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        print(f"已加载模型: {model_path}")
        print(f"训练步数: {loaded_step}")

        # 创建环境
        self.env = MahjongEnv(render_mode=None)

    def _numpy_obs_to_torch(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """将numpy观测转换为torch张量"""
        torch_obs = {}
        for key, value in obs.items():
            if key != "action_mask":
                torch_obs[key] = torch.from_numpy(value).unsqueeze(0).to(self.device)
        return torch_obs

    def evaluate(
        self,
        num_episodes: int = 100,
        deterministic: bool = True,
        verbose: bool = True,
        max_steps: int = 2000,  # 0 表示不限制步数，仅依赖环境的终局信号
    ) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            num_episodes: 评估局数
            deterministic: 是否使用确定性策略
            verbose: 是否显示进度

        Returns:
            评估统计信息
        """
        # 统计信息（不再统计win概率，改为马点/终局积分表现）
        episode_rewards = {f"player_{i}": [] for i in range(4)}
        episode_lengths = []
        final_scores = {f"player_{i}": [] for i in range(4)}
        # 胜负类型统计
        ron_count = 0
        tsumo_count = 0
        draw_count = 0
        # 马点（uma）结果：默认使用四人场 [15, 5, -5, -15]，单位千点
        uma_table = [15, 5, -5, -15]
        base_points = 25000  # 起始分
        oka = 0  # 雀魂通常不加oka，这里默认0（如需变更可在此调整）
        # 纯Uma(排名) 与 合成马点(分差+Uma)
        player_uma_rank = {f"player_{i}": [] for i in range(4)}
        player_uma_combined = {f"player_{i}": [] for i in range(4)}
        # 名次分布统计（每位玩家）
        rank_counts = {f"player_{i}": {1: 0, 2: 0, 3: 0, 4: 0} for i in range(4)}
        # 每局总分校验
        score_sums = []
        # 东家出现次数统计
        east_counts = {f"player_{i}": 0 for i in range(4)}

        def _seat_priority(w):
            # 将座风转换为优先级：E(0) < S(1) < W(2) < N(3)
            if isinstance(w, str):
                m = {
                    "E": 0, "S": 1, "W": 2, "N": 3,
                    "东": 0, "南": 1, "西": 2, "北": 3,
                    "east": 0, "south": 1, "west": 2, "north": 3,
                }
                return m.get(w, 9)
            try:
                wi = int(w)
                return wi if 0 <= wi <= 3 else 9
            except Exception:
                return 9

        iterator = (
            tqdm(range(num_episodes), desc="评估中") if verbose else range(num_episodes)
        )

        for episode in iterator:
            obs, info = self.env.reset(seed=episode)
            try:
                # 评估/交互关闭近似掩码，启用精确合法性（允许吃碰杠/响应）
                self.env.game_state.fast_mask = False
            except Exception:
                pass
            episode_length = 0
            episode_reward_acc = {f"player_{i}": 0.0 for i in range(4)}

            warned_zero_mask = False
            last_tiles_remaining = None

            while True:
                # 终局/异常状态保护
                if getattr(self.env, "game_state", None) is not None:
                    if getattr(self.env.game_state, "phase", None) == "end":
                        break
                if self.env.agent_selection is None:
                    break
                # 确保每步开始时获取当前智能体的最新观测
                try:
                    obs = self.env.observe(self.env.agent_selection)
                except Exception:
                    pass
                if max_steps > 0 and episode_length >= max_steps:
                    if verbose:
                        print(f"[警告] 单局步数超过上限 {max_steps}，提前结束该局。")
                        # 打印关键状态快照
                        try:
                            gs = self.env.game_state
                            print(f"[状态] phase={getattr(gs,'phase',None)} cur={getattr(gs,'current_player',None)} tiles_remaining={getattr(gs,'tiles_remaining',None)} pending={len(getattr(gs,'pending_responses',[]))} last_discard={getattr(gs,'last_discard',None)}")
                        except Exception:
                            pass
                    break

                # 若环境标记任一智能体终止，则认为该局已结束
                if any(self.env.terminations.values()):
                    break

                current_agent = self.env.agent_selection
                action_mask = obs["action_mask"]
                if np.asarray(action_mask).sum() <= 0:
                    # 尝试一次自救：若在draw阶段或无drawn_tile，则触发一次观察促使摸牌
                    try:
                        if getattr(self.env, "game_state", None) is not None:
                            gs = self.env.game_state
                            if gs.phase == "draw" and self.env.agent_selection == current_agent:
                                # 再获取一次观测（环境 observe 内已实现 draw→discard 推进）
                                obs = self.env.observe(current_agent)
                                action_mask = obs.get("action_mask", action_mask)
                    except Exception:
                        pass
                    if np.asarray(action_mask).sum() <= 0:
                        if verbose and not warned_zero_mask:
                            print("[警告] action_mask 全为0，提前结束该局以避免卡死。")
                            warned_zero_mask = True
                        break

                # 为 player_0 使用模型；其他玩家使用随机合法动作
                if current_agent == "player_0":
                    # 获取动作（确定性路径：直接前向 + 掩码 + argmax）
                    torch_obs = self._numpy_obs_to_torch(obs)
                    torch_action_mask = (
                        torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                    )
                    with torch.no_grad():
                        logits, _value = self.model.forward(torch_obs)
                        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
                        logits = logits.clamp(min=-50.0, max=50.0)
                        masked_logits = logits.masked_fill(torch_action_mask == 0, -1e9)
                        action_t = masked_logits.argmax(dim=-1)
                    action_int = int(action_t.cpu().item())
                else:
                    legal = np.where(np.asarray(action_mask) > 0)[0]
                    if legal.size == 0:
                        # 极端保护：无合法动作则选择0
                        action_int = 0
                    else:
                        action_int = int(np.random.choice(legal))

                # 执行动作
                self.env.step(action_int)

                # 累计奖励
                for agent in self.env.possible_agents:
                    episode_reward_acc[agent] += self.env.rewards.get(agent, 0.0)

                episode_length += 1

                # 诊断打印（每200步输出一次状态，便于定位卡点）
                if verbose and (episode_length % 400 == 0):
                    try:
                        gs = self.env.game_state
                        print(f"[诊断] step={episode_length} phase={gs.phase} cur={gs.current_player} tiles_remaining={gs.tiles_remaining} pending={len(gs.pending_responses)}")
                    except Exception:
                        pass

                # 获取下一个观测
                if self.env.agent_selection is not None:
                    obs = self.env.observe(self.env.agent_selection)

            # 记录统计信息
            episode_lengths.append(episode_length)

            for agent in self.env.possible_agents:
                episode_rewards[agent].append(episode_reward_acc[agent])
                player_idx = int(agent.split("_")[1])
                final_scores[agent].append(
                    self.env.game_state.players[player_idx].score
                )

            # 计算该局马点（基于终局分与排名）
            scores_now = [self.env.game_state.players[i].score for i in range(4)]
            # 座风优先级（用于同分破平）
            seat_pri = [_seat_priority(self.env.game_state.players[i].seat_wind) for i in range(4)]
            # 东家统计
            for i in range(4):
                if seat_pri[i] == 0:
                    east_counts[f"player_{i}"] += 1
            # 排名：先比分数降序，同分按东南西北
            order = sorted(range(4), key=lambda x: (-scores_now[x], seat_pri[x]))
            # 名次分布统计
            for r, pid in enumerate(order, start=1):
                rank_counts[f"player_{pid}"][r] += 1
            # 分差换算千点
            diff_k = [(s - base_points) / 1000.0 for s in scores_now]
            # 分配uma
            rank_to_uma = {order[r]: uma_table[r] for r in range(4)}
            # oka 分配（此处默认0，若设置>0通常全部给第一名，按需更改）
            rank_to_oka = {order[0]: oka, order[1]: 0, order[2]: 0, order[3]: 0}
            for pid in range(4):
                # 纯Uma(排名)：不叠加分差/oka
                uma_rank = rank_to_uma[pid]
                # 合成马点(分差+Uma+Oka)
                uma_combined = diff_k[pid] + rank_to_uma[pid] + rank_to_oka[pid]
                player_uma_rank[f"player_{pid}"].append(uma_rank)
                player_uma_combined[f"player_{pid}"].append(uma_combined)
            # 记录总分和（用于校验是否恒等于 4*base_points）
            score_sums.append(sum(scores_now))

            # 记录胜负类型
            try:
                rr = self.env.game_state.round_result
                if rr is not None:
                    if rr.result_type == "ron":
                        ron_count += 1
                    elif rr.result_type == "tsumo":
                        tsumo_count += 1
                    elif rr.result_type in ("draw", "abort"):
                        draw_count += 1
            except Exception:
                pass

        # 计算统计结果
        results = {
            "num_episodes": num_episodes,
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
        }

        # 每个玩家的统计
        for agent in self.env.possible_agents:
            prefix = f"{agent}_"
            results[prefix + "mean_reward"] = np.mean(episode_rewards[agent])
            results[prefix + "std_reward"] = np.std(episode_rewards[agent])
            results[prefix + "mean_score"] = np.mean(final_scores[agent])
            results[prefix + "std_score"] = np.std(final_scores[agent])
            # 追加马点统计：纯Uma(排名) 与 合成马点(分差+Uma)
            results[prefix + "mean_uma_rank"] = np.mean(player_uma_rank[agent])
            results[prefix + "std_uma_rank"] = np.std(player_uma_rank[agent])
            results[prefix + "mean_uma_combined"] = np.mean(player_uma_combined[agent])
            results[prefix + "std_uma_combined"] = np.std(player_uma_combined[agent])
            # 兼容字段：mean_uma 默认为合成马点
            results[prefix + "mean_uma"] = results[prefix + "mean_uma_combined"]
            results[prefix + "std_uma"] = results[prefix + "std_uma_combined"]
            # 名次分布
            rc = rank_counts[agent]
            total = sum(rc.values()) if sum(rc.values()) > 0 else 1
            results[prefix + "rank1_rate"] = rc[1] / total
            results[prefix + "rank2_rate"] = rc[2] / total
            results[prefix + "rank3_rate"] = rc[3] / total
            results[prefix + "rank4_rate"] = rc[4] / total
            # 东家占比
            results[prefix + "east_rate"] = east_counts[agent] / max(1, num_episodes)

        # 总分和校验
        score_sums_arr = np.array(score_sums, dtype=np.float64)
        # 考虑立直棒：每棒1000点暂留在场上；理想情况下总分=4*25000 - 1000*riichi_sticks
        try:
            sticks = float(getattr(self.env.game_state, 'riichi_sticks', 0))
        except Exception:
            sticks = 0.0
        target_sum = base_points * 4 - 1000.0 * sticks
        results["score_sum_mean"] = float(score_sums_arr.mean()) if score_sums_arr.size else 0.0
        results["score_sum_std"] = float(score_sums_arr.std()) if score_sums_arr.size else 0.0
        results["score_sum_ok_rate"] = float(np.mean(score_sums_arr == target_sum)) if score_sums_arr.size else 0.0

        # player_0（主要评估对象）的综合性能（以合成马点为主）
        results["player_0_performance"] = results["player_0_mean_uma_combined"]

        # 胜负类型比例
        total_eps = max(1, num_episodes)
        results["ron_rate"] = ron_count / total_eps
        results["tsumo_rate"] = tsumo_count / total_eps
        results["draw_rate"] = draw_count / total_eps

        return results

    def evaluate_full_match(self, mode: str = "east", verbose: bool = True) -> Dict[str, float]:
        """
        整场评估：东风/半庄。按多局累积分差后一次性计算Uma。
        说明：当前环境按“单局”重置，这里采用每局得分相对25000的“分差累积”近似整场计分。
        Args:
            mode: "east" (东风战，约4局) 或 "hanchan" (半庄，约8局)
        Returns:
            包含最终整场分数与Uma的统计
        """
        uma_table = [15, 5, -5, -15]
        base_points = 25000
        oka = 0
        # 设定整场局数（不含连庄近似；如需严格连庄需环境支持）
        total_rounds = 4 if mode == "east" else 8
        # 累积分差（起始分25000，累计每局相对分差）
        cum_scores = np.array([base_points] * 4, dtype=np.int64)

        for rnd in range(total_rounds):
            # 每局独立评估一把并取终局分
            obs, info = self.env.reset(seed=10000 + rnd)
            while True:
                if getattr(self.env, "game_state", None) is not None:
                    if getattr(self.env.game_state, "phase", None) == "end":
                        break
                if self.env.agent_selection is None:
                    break
                try:
                    obs = self.env.observe(self.env.agent_selection)
                except Exception:
                    pass
                action_mask = obs.get("action_mask", None)
                if action_mask is None or np.asarray(action_mask).sum() <= 0:
                    break
                # 确定性行动
                torch_obs = self._numpy_obs_to_torch(obs)
                torch_action_mask = (
                    torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                )
                with torch.no_grad():
                    logits, _value = self.model.forward(torch_obs)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9).clamp(-50.0, 50.0)
                    action = logits.masked_fill(torch_action_mask == 0, -1e9).argmax(dim=-1)
                self.env.step(int(action.cpu().item()))
                if self.env.agent_selection is not None:
                    try:
                        obs = self.env.observe(self.env.agent_selection)
                    except Exception:
                        pass
            # 回合结束：获取本局终局分
            round_scores = [self.env.game_state.players[i].score for i in range(4)]
            # 累积分差
            for i in range(4):
                cum_scores[i] += (round_scores[i] - base_points)

        # 整场Uma（按最终分数排名）
        # 使用当前局面座风作为同分破平（近似）
        def _seat_priority(w):
            if isinstance(w, str):
                m = {
                    "E": 0, "S": 1, "W": 2, "N": 3,
                    "东": 0, "南": 1, "西": 2, "北": 3,
                    "east": 0, "south": 1, "west": 2, "north": 3,
                }
                return m.get(w, 9)
            try:
                wi = int(w)
                return wi if 0 <= wi <= 3 else 9
            except Exception:
                return 9
        seat_pri = [_seat_priority(self.env.game_state.players[i].seat_wind) for i in range(4)]
        order = sorted(range(4), key=lambda x: (-int(cum_scores[x]), seat_pri[x]))
        rank_to_uma = {order[r]: uma_table[r] for r in range(4)}
        rank_to_oka = {order[0]: oka, order[1]: 0, order[2]: 0, order[3]: 0}
        diff_k_final = [(s - base_points) / 1000.0 for s in cum_scores]
        uma_rank = [rank_to_uma[i] for i in range(4)]
        uma_combined = [diff_k_final[i] + rank_to_uma[i] + rank_to_oka[i] for i in range(4)]

        results = {
            "match_mode": mode,
            "final_scores": {f"player_{i}": int(cum_scores[i]) for i in range(4)},
            "uma_rank": {f"player_{i}": float(uma_rank[i]) for i in range(4)},
            "uma_combined": {f"player_{i}": float(uma_combined[i]) for i in range(4)},
            "player_0_performance": float(uma_combined[0]),
        }
        return results

    def play_interactive(self, render: bool = True, max_steps: int = 0):
        """
        交互式游戏（人类可以观察AI如何决策）

        Args:
            render: 是否渲染游戏状态
        """
        print("\n" + "=" * 80)
        print(" " * 30 + "交互式演示")
        print("=" * 80 + "\n")

        obs, info = self.env.reset()
        try:
            self.env.game_state.fast_mask = False
        except Exception:
            pass
        step = 0

        warned_zero_mask = False
        while True:
            # 终局/异常保护
            if getattr(self.env, "game_state", None) is not None:
                if getattr(self.env.game_state, "phase", None) == "end":
                    break
            if self.env.agent_selection is None:
                break
            # 确保每步开始时获取当前智能体的最新观测
            try:
                obs = self.env.observe(self.env.agent_selection)
            except Exception:
                pass
            if max_steps > 0 and step >= max_steps:
                print(f"[警告] 交互演示步数超过上限 {max_steps}，提前结束。")
                break

            # 若环境标记任一智能体终止，则认为该局已结束
            if any(self.env.terminations.values()):
                break

            current_agent = self.env.agent_selection
            action_mask = obs["action_mask"]
            if np.asarray(action_mask).sum() <= 0:
                if not warned_zero_mask:
                    print("[警告] action_mask 全为0，提前结束以避免卡死。")
                    warned_zero_mask = True
                break

            # 获取AI动作（交互：使用采样概率展示，但避免 log_prob 计算卡顿，改为 softmax 概率）
            torch_obs = self._numpy_obs_to_torch(obs)
            torch_action_mask = (
                torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                logits, value = self.model.forward(torch_obs)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
                logits = logits.clamp(min=-50.0, max=50.0)
                masked_logits = logits.masked_fill(torch_action_mask == 0, -1e9)
                if masked_logits.dtype.is_floating_point:
                    probs = torch.softmax(masked_logits, dim=-1)
                else:
                    probs = None
                # 采样或贪心，这里与训练一致可取采样；为稳定可使用贪心
                action = masked_logits.argmax(dim=-1)

            # 解码动作
            from mahjong_environment.utils.action_encoder import ActionEncoder

            action_type, params = ActionEncoder.decode_action(action.cpu().item())

            # 显示决策信息
            print(f"\n【步骤 {step}】 当前玩家: {current_agent}")
            print(f"  动作: {action_type} {params}")
            print(f"  价值估计: {value.cpu().item():.3f}")
            try:
                sel_prob = probs[0, action.item()].cpu().item() if probs is not None else None
            except Exception:
                sel_prob = None
            if sel_prob is not None:
                print(f"  动作概率: {sel_prob:.3f}")

            # 输出所有玩家手牌与副露（仅交互演示用于人工校对）
            try:
                from mahjong_environment.utils.tile_utils import format_hand as _format_hand
                gs = self.env.game_state
                for i, agent in enumerate(self.env.possible_agents):
                    player = gs.players[i]
                    hand_tiles = player.get_all_tiles()
                    tile_cnt = len(hand_tiles)
                    has_drawn = bool(player.drawn_tile)
                    print(f"  {agent} 手牌({tile_cnt}张, 摸牌={'是' if has_drawn else '否'}): {_format_hand(hand_tiles)}", end=' ')
                    try:
                        if player.open_melds:
                            meld_strs = []
                            for m in player.open_melds:
                                meld_strs.append(f"{m.meld_type}:{_format_hand(m.tiles)}")
                            print(f"  {agent} 副露: {' | '.join(meld_strs)}")
                        else:
                            print(f"  {agent} 副露: 无")
                    except Exception:
                        pass
            except Exception:
                pass

            # 执行动作
            self.env.step(action.cpu().item())
            step += 1

            # 渲染
            if render and step % 10 == 0:
                self.env.render()

            # 获取下一个观测
            if self.env.agent_selection is not None:
                obs = self.env.observe(self.env.agent_selection)

        # 显示最终结果
        print("\n" + "=" * 80)
        print("游戏结束！")
        print("=" * 80)
        self.env.render()

        print("\n最终分数:")
        for i, agent in enumerate(self.env.possible_agents):
            player = self.env.game_state.players[i]
            reward = self.env.rewards[agent]
            print(f"  {agent}: {player.score} 点 (奖励: {reward:+.2f})")

        # 若有役种信息，打印役种
        try:
            rr = getattr(self.env.game_state, 'round_result', None)
            if rr is not None and getattr(rr, 'yaku_names', None):
                print("\n役种:")
                for name in rr.yaku_names:
                    print(f"  - {name}")
        except Exception:
            pass

    def benchmark_vs_random(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        与随机策略对战的基准测试

        Args:
            num_episodes: 测试局数

        Returns:
            基准测试结果
        """
        print(f"\n基于终局马点的评估（{num_episodes}局）...")
        results = self.evaluate(num_episodes=num_episodes, verbose=True)

        print("\n基准测试结果:")
        print(f"  player_0 纯Uma(排名): {results['player_0_mean_uma_rank']:.2f} 千点")
        print(f"  player_0 合成马点(分差+Uma): {results['player_0_mean_uma_combined']:.2f} 千点")
        print(f"  player_0 平均分数: {results['player_0_mean_score']:.1f}")
        # 追加摘要：名次分布、东家占比与总分校验
        try:
            print(
                f"  player_0 名次分布: 1位={results['player_0_rank1_rate']:.2%}, 2位={results['player_0_rank2_rate']:.2%}, 3位={results['player_0_rank3_rate']:.2%}, 4位={results['player_0_rank4_rate']:.2%}"
            )
            print(f"  player_0 东家占比: {results['player_0_east_rate']:.2%}")
            print(
                f"  总分和校验: 均值={results['score_sum_mean']:.1f}, 方差={results['score_sum_std']:.1f}, 等于100000比例={results['score_sum_ok_rate']:.2%}"
            )
        except Exception:
            pass

        return results

    def save_evaluation_report(self, results: Dict[str, float], output_path: str):
        """
        保存评估报告

        Args:
            results: 评估结果
            output_path: 输出路径
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 30 + "麻将AI评估报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"评估局数: {results['num_episodes']}\n")
            f.write(
                f"平均局长: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}\n\n"
            )

            f.write("=" * 80 + "\n")
            f.write("各玩家性能:\n")
            f.write("=" * 80 + "\n\n")

            for i in range(4):
                agent = f"player_{i}"
                f.write(f"{agent}:\n")
                if f"{agent}_mean_reward" in results:
                    f.write(
                        f"  平均奖励: {results[f'{agent}_mean_reward']:.3f} ± {results[f'{agent}_std_reward']:.3f}\n"
                    )
                if f"{agent}_mean_score" in results:
                    f.write(
                        f"  平均分数: {results[f'{agent}_mean_score']:.1f} ± {results[f'{agent}_std_score']:.1f}\n"
                    )
                # 纯Uma(排名)
                f.write(
                    f"  纯Uma(排名): {results[f'{agent}_mean_uma_rank']:.2f} ± {results[f'{agent}_std_uma_rank']:.2f} 千点\n"
                )
                # 合成马点(分差+Uma)
                f.write(
                    f"  合成马点(分差+Uma): {results[f'{agent}_mean_uma_combined']:.2f} ± {results[f'{agent}_std_uma_combined']:.2f} 千点\n"
                )
                # 名次分布与东家占比
                f.write(
                    f"  名次分布: 1位={results[f'{agent}_rank1_rate']:.2%}, 2位={results[f'{agent}_rank2_rate']:.2%}, 3位={results[f'{agent}_rank3_rate']:.2%}, 4位={results[f'{agent}_rank4_rate']:.2%}\n"
                )
                f.write(f"  东家占比: {results[f'{agent}_east_rate']:.2%}\n")
                # 可选胜率
                if f"{agent}_win_rate" in results:
                    f.write(f"  胜率: {results[f'{agent}_win_rate']:.2%}\n")
                f.write("\n")

            # 可选的胜利类型统计（若存在）
            has_types = any(k in results for k in ("ron_rate", "tsumo_rate", "draw_rate"))
            if has_types:
                f.write("=" * 80 + "\n")
                f.write("胜利类型统计:\n")
                f.write("=" * 80 + "\n\n")
                if "ron_rate" in results:
                    f.write(f"荣和率: {results['ron_rate']:.2%}\n")
                if "tsumo_rate" in results:
                    f.write(f"自摸率: {results['tsumo_rate']:.2%}\n")
                if "draw_rate" in results:
                    f.write(f"流局率: {results['draw_rate']:.2%}\n")
            # 总分和校验
            f.write("=" * 80 + "\n")
            f.write("总分和校验:\n")
            f.write("=" * 80 + "\n\n")
            f.write(
                f"均值: {results['score_sum_mean']:.1f}  方差: {results['score_sum_std']:.1f}  等于100000比例: {results['score_sum_ok_rate']:.2%}\n"
            )

        print(f"\n评估报告已保存至: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="评估麻将AI")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    # 支持 --episodes 与 --episode 两个别名
    parser.add_argument("--episodes", "--episode", type=int, default=100, help="评估局数")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--interactive", action="store_true", help="交互式演示")
    parser.add_argument(
        "--output", type=str, default="evaluation_report.txt", help="报告输出路径"
    )
    parser.add_argument("--full-match", choices=["east", "hanchan"], default=None, help="整场评估模式：east(东风)/hanchan(半庄)")

    args = parser.parse_args()

    # 创建评估器
    evaluator = MahjongEvaluator(model_path=args.model, device=args.device)

    if args.interactive:
        # 交互式演示
        evaluator.play_interactive(render=True)
    elif args.full_match is not None:
        # 整场评估
        res = evaluator.evaluate_full_match(mode=args.full_match, verbose=True)
        print("\n整场评估结果:")
        print(f"  模式: {res['match_mode']}")
        print(f"  最终分数: {res['final_scores']}")
        print(f"  纯Uma(排名): {res['uma_rank']}")
        print(f"  合成马点(分差+Uma): {res['uma_combined']}")
    else:
        # 标准评估
        results = evaluator.benchmark_vs_random(num_episodes=args.episodes)

        # 保存报告
        evaluator.save_evaluation_report(results, args.output)


if __name__ == "__main__":
    main()
