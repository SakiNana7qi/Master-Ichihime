@echo off
chcp 936 >nul
REM ¿ìËÙÑµÁ·½Å±¾ - Windows

echo ======================================
echo   Âéœ«AIÑµÁ· - ¿ìËÙ¿ªÊŒ
echo ======================================
echo.

echo 1. Œì²éÒÀÀµ...
python -c "import torch; import numpy; import gymnasium" 2>nul
if errorlevel 1 (
    echo °²×°ÒÀÀµ°ü...
    pip install -r mahjong_agent\requirements.txt
)

echo.
echo 2. ¿ªÊŒÑµÁ·£š¿ìËÙÅäÖã©...
python -m mahjong_agent.train --config fast --device cuda --seed 42

echo.
echo ======================================
echo   ÑµÁ·Íê³É£¡
echo ======================================
echo.
echo Œì²é±£µãŽæÔÚ: .\checkpoints\
echo ÈÕÖŸÏÀñµÔÚ: .\logs\
echo.
echo Ê¹ÓÃ 'tensorboard --logdir logs/' ²é¿ŽÑµÁ·ÇúÏß
echo.
pause
