#!/usr/bin/env python
"""
PyQt5 实时训练监控界面

读取 mahjong_agent/train.py 持续写入的 logs/realtime_metrics.jsonl 文件，
实时展示关键训练指标与曲线。
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


class MetricsReader(QtCore.QThread):
    """后台线程：持续读取JSONL文件并发出新数据信号"""

    new_record = QtCore.pyqtSignal(dict)

    def __init__(self, file_path: str, poll_interval: float = 0.5):
        super().__init__()
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._stop = False
        self._pos = 0

    def stop(self):
        self._stop = True

    def run(self):
        while not self._stop:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    f.seek(self._pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            self.new_record.emit(data)
                        except Exception:
                            pass
                    self._pos = f.tell()
            except FileNotFoundError:
                pass

            time.sleep(self.poll_interval)


class MonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, metrics_path: str):
        super().__init__()
        self.setWindowTitle("Mahjong AI 训练监控")
        self.resize(1100, 700)

        # 中央widget
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 顶部信息栏
        self.info_label = QtWidgets.QLabel("等待训练数据...")
        layout.addWidget(self.info_label)

        # 图表区域
        plots_container = QtWidgets.QWidget()
        plots_layout = QtWidgets.QGridLayout(plots_container)
        layout.addWidget(plots_container)

        pg.setConfigOptions(antialias=True)

        # 训练曲线：回报、策略损失、价值损失、熵、KL
        self.plot_widgets: Dict[str, pg.PlotWidget] = {}
        self.plot_data: Dict[str, List[float]] = {}
        self.plot_steps: List[int] = []

        metrics = [
            ("mean_episode_reward", "平均回报"),
            ("policy_loss", "策略损失"),
            ("value_loss", "价值损失"),
            ("entropy", "熵"),
            ("approx_kl", "近似KL"),
            ("learning_rate", "学习率"),
            ("clip_range", "裁剪范围"),
            ("fps", "FPS"),
            ("win_rate", "评估胜率"),
        ]

        for i, (key, title) in enumerate(metrics):
            pw = pg.PlotWidget()
            pw.setTitle(title)
            pw.showGrid(x=True, y=True)
            self.plot_widgets[key] = pw
            self.plot_data[key] = []
            row, col = divmod(i, 3)
            plots_layout.addWidget(pw, row, col)

        # 底部控制栏
        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)

        self.path_label = QtWidgets.QLabel(metrics_path)
        toolbar.addWidget(QtWidgets.QLabel("监控文件: "))
        toolbar.addWidget(self.path_label)
        toolbar.addStretch(1)

        # 文件读取线程
        self.reader = MetricsReader(metrics_path, poll_interval=0.5)
        self.reader.new_record.connect(self.on_new_record)
        self.reader.start()

        # 曲线对象
        self.curves: Dict[str, pg.PlotDataItem] = {}
        for key in self.plot_widgets:
            self.curves[key] = self.plot_widgets[key].plot(pen=pg.mkPen(width=2))

    def closeEvent(self, event):
        try:
            self.reader.stop()
            self.reader.wait(1000)
        finally:
            event.accept()

    @QtCore.pyqtSlot(dict)
    def on_new_record(self, rec: dict):
        step = int(rec.get("global_step", 0))
        self.plot_steps.append(step)

        # 更新顶栏信息
        fps = rec.get("fps", rec.get("FPS", 0.0))
        info = f"Step: {step:,} | Rollout: {rec.get('rollout', 0)} | FPS: {fps:.1f}"
        self.info_label.setText(info)

        # 更新各曲线（训练与评估记录都写入，对应键不存在则补0）
        for key in self.plot_data:
            val = rec.get(key, 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            self.plot_data[key].append(val)
            self.curves[key].setData(self.plot_steps, self.plot_data[key])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mahjong AI 训练GUI监控")
    parser.add_argument(
        "--log",
        type=str,
        default=os.path.join("logs", "realtime_metrics.jsonl"),
        help="指标文件路径",
    )
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = MonitorWindow(args.log)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
