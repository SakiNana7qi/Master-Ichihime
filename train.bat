@echo off
REM Simple training script - no encoding issues

echo ======================================
echo   Mahjong AI Training
echo ======================================
echo.

echo Checking dependencies...
python -c "import torch, numpy, gymnasium" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r mahjong_agent\requirements.txt
    echo.
)

echo Starting training...
echo.
python -m mahjong_agent.train --config fast --device cuda --seed 42

echo.
echo ======================================
echo   Training Complete
echo ======================================
echo.
echo Checkpoints: .\checkpoints\
echo Logs: .\logs\
echo.
echo View training: tensorboard --logdir logs/
echo.
pause
