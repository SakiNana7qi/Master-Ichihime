@echo off
chcp 65001 >nul
REM Kuai su xun lian jiao ben - Windows

echo ======================================
echo   Mahjong AI Training - Quick Start
echo ======================================
echo.

echo 1. Checking dependencies...
python -c "import torch; import numpy; import gymnasium" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r mahjong_agent\requirements.txt
)

echo.
echo 2. Starting training (fast config)...
python -m mahjong_agent.train --config fast --device cuda --seed 42

echo.
echo ======================================
echo   Training Complete!
echo ======================================
echo.
echo Checkpoints saved in: .\checkpoints\
echo Logs saved in: .\logs\
echo.
echo Use 'tensorboard --logdir logs/' to view training curves
echo.
pause
