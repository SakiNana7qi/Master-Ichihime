# mahjong_environment/gui_demo.py
"""
麻将环境GUI演示
使用tkinter创建图形界面展示随机智能体对局
"""

import sys
import os
import random
import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import threading
import time

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_environment import MahjongEnv
from mahjong_environment.utils.action_encoder import ActionEncoder
from mahjong_environment.utils.tile_utils import tile_to_unicode


class MahjongGUI:
    """麻将游戏GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("雀魂立直麻将 - 随机智能体演示")
        self.root.geometry("1200x800")

        # 游戏状态
        self.env = None
        self.obs = None
        self.is_running = False
        self.auto_play = False
        self.step_count = 0

        # 创建界面
        self.create_widgets()

        # 初始化环境
        self.reset_game()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # 顶部控制区
        self.create_control_panel(main_frame)

        # 中间游戏显示区
        self.create_game_display(main_frame)

        # 底部日志区
        self.create_log_panel(main_frame)

    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 游戏信息
        self.info_label = ttk.Label(
            control_frame,
            text="步数: 0 | 剩余牌数: 0 | 当前玩家: -",
            font=("Arial", 10),
        )
        self.info_label.grid(row=0, column=0, columnspan=4, pady=(0, 10))

        # 按钮
        ttk.Button(control_frame, text="重置游戏", command=self.reset_game).grid(
            row=1, column=0, padx=5
        )
        ttk.Button(control_frame, text="下一步", command=self.step_game).grid(
            row=1, column=1, padx=5
        )

        self.auto_button = ttk.Button(
            control_frame, text="自动运行", command=self.toggle_auto_play
        )
        self.auto_button.grid(row=1, column=2, padx=5)

        ttk.Button(control_frame, text="清空日志", command=self.clear_log).grid(
            row=1, column=3, padx=5
        )

        # 速度控制
        ttk.Label(control_frame, text="速度:").grid(row=2, column=0, pady=(10, 0))
        self.speed_var = tk.IntVar(value=500)
        speed_scale = ttk.Scale(
            control_frame,
            from_=100,
            to=2000,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
        )
        speed_scale.grid(
            row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0), padx=5
        )

    def create_game_display(self, parent):
        """创建游戏显示区"""
        game_frame = ttk.LabelFrame(parent, text="游戏状态", padding="10")
        game_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        game_frame.columnconfigure(0, weight=1)
        game_frame.rowconfigure(0, weight=1)

        # 使用Canvas来绘制游戏状态
        canvas_frame = ttk.Frame(game_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # 创建带滚动条的Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="white", height=400)
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )

        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.canvas.configure(yscrollcommand=scrollbar.set)

    def create_log_panel(self, parent):
        """创建日志面板"""
        log_frame = ttk.LabelFrame(parent, text="游戏日志", padding="10")
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=8, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

    def log(self, message):
        """添加日志"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)

    def reset_game(self):
        """重置游戏"""
        self.auto_play = False
        self.is_running = False
        self.step_count = 0

        # 创建新环境
        seed = random.randint(0, 10000)
        self.env = MahjongEnv(render_mode=None, seed=seed)
        self.obs, info = self.env.reset(seed=seed)

        self.log(f"[系统] 游戏重置 (种子: {seed})")
        self.update_display()

    def step_game(self):
        """执行一步游戏"""
        if self.env is None or self.env.agent_selection is None:
            self.log("[警告] 游戏已结束或未初始化")
            return

        if self.env.terminations[self.env.agent_selection]:
            self.log("[系统] 游戏已结束")
            self.display_final_result()
            return

        # 获取合法动作
        action_mask = self.obs["action_mask"]
        legal_actions = np.where(action_mask == 1)[0]

        if len(legal_actions) == 0:
            self.log("[错误] 没有合法动作")
            return

        # 随机选择动作
        action = random.choice(legal_actions)
        action_type, params = ActionEncoder.decode_action(action)

        current_agent = self.env.agent_selection

        # 执行动作
        self.env.step(action)
        self.step_count += 1

        # 记录日志
        self.log(f"[第{self.step_count}步] {current_agent} 执行 {action_type} {params}")

        # 检查是否结束
        if self.env.agent_selection is not None:
            self.obs = self.env.observe(self.env.agent_selection)

        # 更新显示
        self.update_display()

        # 检查游戏是否结束
        if any(self.env.terminations.values()):
            self.log("[系统] 游戏结束！")
            self.display_final_result()
            self.auto_play = False

    def toggle_auto_play(self):
        """切换自动运行"""
        self.auto_play = not self.auto_play

        if self.auto_play:
            self.auto_button.config(text="停止")
            self.log("[系统] 开始自动运行")
            self.auto_play_thread()
        else:
            self.auto_button.config(text="自动运行")
            self.log("[系统] 停止自动运行")

    def auto_play_thread(self):
        """自动运行线程"""
        if not self.auto_play:
            return

        self.step_game()

        if self.auto_play and not any(self.env.terminations.values()):
            delay = self.speed_var.get()
            self.root.after(delay, self.auto_play_thread)

    def update_display(self):
        """更新游戏显示"""
        if self.env is None:
            return

        # 清空Canvas
        self.canvas.delete("all")

        # 更新信息标签
        tiles_remaining = self.env.game_state.tiles_remaining
        current_player = (
            self.env.agent_selection if self.env.agent_selection else "结束"
        )
        self.info_label.config(
            text=f"步数: {self.step_count} | 剩余牌数: {tiles_remaining} | 当前玩家: {current_player}"
        )

        y_offset = 20

        # 显示场况信息
        wind_names = {"east": "东", "south": "南", "west": "北", "north": "北"}
        game_info = (
            f"【{wind_names[self.env.game_state.round_wind]}场】 "
            f"本场: {self.env.game_state.honba} "
            f"立直棒: {self.env.game_state.riichi_sticks}"
        )
        self.canvas.create_text(
            600, y_offset, text=game_info, font=("Arial", 12, "bold"), anchor=tk.N
        )
        y_offset += 30

        # 显示宝牌
        dora_text = "宝牌指示牌: " + " ".join(
            [tile_to_unicode(d) for d in self.env.game_state.dora_indicators]
        )
        self.canvas.create_text(
            600, y_offset, text=dora_text, font=("Arial", 10), anchor=tk.N
        )
        y_offset += 30

        # 显示四个玩家
        for i, player in enumerate(self.env.game_state.players):
            y_offset = self.draw_player(player, y_offset, i)

        # 更新Canvas滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def draw_player(self, player, y_start, player_id):
        """绘制单个玩家信息"""
        x_margin = 20
        y = y_start

        # 玩家标题
        wind_names = {"east": "东", "south": "南", "west": "西", "north": "北"}
        is_current = self.env.agent_selection == f"player_{player_id}"
        is_dealer = player_id == self.env.game_state.dealer

        title = f"玩家{player_id} [{wind_names[player.seat_wind]}]"
        if is_dealer:
            title += " [庄家]"
        if player.is_riichi:
            title += " [立直]"

        color = "red" if is_current else "black"
        self.canvas.create_text(
            x_margin,
            y,
            text=title,
            font=("Arial", 11, "bold"),
            anchor=tk.W,
            fill=color,
        )

        # 分数
        self.canvas.create_text(
            x_margin + 200,
            y,
            text=f"分数: {player.score}",
            font=("Arial", 10),
            anchor=tk.W,
        )
        y += 25

        # 手牌（只在调试模式或是当前玩家时显示）
        if True:  # 这里设为True以便演示，实际可以根据需要调整
            hand_tiles = player.get_all_tiles()
            hand_text = "手牌: " + " ".join([tile_to_unicode(t) for t in hand_tiles])
            self.canvas.create_text(
                x_margin + 20, y, text=hand_text, font=("Arial", 10), anchor=tk.W
            )
            y += 25

        # 牌河（显示最近10张）
        if player.river:
            river_display = player.river[-10:]
            if len(player.river) > 10:
                river_text = "牌河: ... " + " ".join(
                    [tile_to_unicode(t) for t in river_display]
                )
            else:
                river_text = "牌河: " + " ".join(
                    [tile_to_unicode(t) for t in river_display]
                )
            self.canvas.create_text(
                x_margin + 20, y, text=river_text, font=("Arial", 9), anchor=tk.W
            )
            y += 25

        # 副露
        if player.open_melds:
            melds_text = "副露: " + " | ".join(
                [
                    f"{m.meld_type}[{' '.join([tile_to_unicode(t) for t in m.tiles])}]"
                    for m in player.open_melds
                ]
            )
            self.canvas.create_text(
                x_margin + 20, y, text=melds_text, font=("Arial", 9), anchor=tk.W
            )
            y += 25

        # 分隔线
        self.canvas.create_line(x_margin, y + 5, 1150, y + 5, fill="gray", dash=(4, 2))
        y += 20

        return y

    def display_final_result(self):
        """显示最终结果"""
        self.log("\n" + "=" * 60)
        self.log("最终结果:")
        self.log("=" * 60)

        for i, agent in enumerate(self.env.possible_agents):
            player = self.env.game_state.players[i]
            reward = self.env.rewards[agent]

            status = "[+]" if reward > 0 else "[-]" if reward < 0 else "[=]"
            self.log(f"{status} {agent}: 分数={player.score}, 奖励={reward:+.2f}")

        if self.env.game_state.round_result:
            result = self.env.game_state.round_result
            self.log(f"\n结果类型: {result.result_type}")

            if result.winner is not None:
                self.log(f"和牌者: player_{result.winner}")
                self.log(f"番数: {result.han} 番")
                self.log(f"符数: {result.fu} 符")
                self.log(f"得点: {result.points} 点")

                if result.loser is not None:
                    self.log(f"放铳者: player_{result.loser}")

        self.log("=" * 60 + "\n")


def main():
    """主函数"""
    root = tk.Tk()
    app = MahjongGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
