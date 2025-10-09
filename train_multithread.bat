@echo off
REM 多线程训练启动脚本 (Windows)

echo ==========================================
echo   麻将 AI 多线程训练启动脚本 (Windows)
echo ==========================================
echo.

REM 1. 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    pause
    exit /b 1
)

echo Python 版本:
python --version

REM 2. 设置环境变量
echo.
echo 设置环境变量...

REM 获取 CPU 核心数 (Windows)
set CPU_COUNT=%NUMBER_OF_PROCESSORS%
if "%CPU_COUNT%"=="" set CPU_COUNT=32

echo 检测到 CPU 核心数: %CPU_COUNT%

REM 设置线程数
set OMP_NUM_THREADS=%CPU_COUNT%
set MKL_NUM_THREADS=%CPU_COUNT%
set OPENBLAS_NUM_THREADS=%CPU_COUNT%
set NUMEXPR_NUM_THREADS=%CPU_COUNT%

REM 优化设置
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0
set OMP_PROC_BIND=spread
set OMP_PLACES=threads

echo 环境变量配置完成:
echo   OMP_NUM_THREADS=%OMP_NUM_THREADS%
echo   MKL_NUM_THREADS=%MKL_NUM_THREADS%
echo   OMP_WAIT_POLICY=%OMP_WAIT_POLICY%

REM 3. 检查 psutil
echo.
echo 检查依赖...
python -c "import psutil" 2>nul
if errorlevel 1 (
    echo 警告: psutil 未安装，无法设置 CPU 亲和度
    echo 建议运行: pip install psutil
    echo.
    set /p CONTINUE=是否继续? (Y/N): 
    if /i not "%CONTINUE%"=="Y" exit /b 1
)

REM 4. 启动训练
echo.
echo ==========================================
echo   开始训练
echo ==========================================
echo.

python -m mahjong_agent.train --config multithread --device cuda --seed 42

echo.
echo 训练完成或已退出
pause


