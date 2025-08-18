@echo off
echo Starting 4D Digital Pheromone Multi-Agent System Experiment
echo ============================================================

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Check for CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo NVIDIA GPU driver not found
    exit /b 1
)

REM Activate Anaconda environment
call conda activate digital_pheromone

REM Create necessary directories
if not exist "results" mkdir results
if not exist "results\figures" mkdir results\figures
if not exist "logs" mkdir logs

REM Check system resources for high-performance setup (RTX A6000 + 30GB RAM)
python -c "import psutil, torch; ram_avail=psutil.virtual_memory().available/1024**3; gpu_mem=torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0; print(f'Available RAM: {ram_avail:.1f}GB, GPU Memory: {gpu_mem:.1f}GB'); print('High-performance configuration enabled') if ram_avail > 20 and gpu_mem > 20 else print('WARNING: Insufficient resources for optimal performance')" 2>nul

REM Set environment variables for high-performance RTX A6000 + 30GB RAM
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,garbage_collection_threshold:0.9,expandable_segments:True
set CUDA_LAUNCH_BLOCKING=0
set RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0
set RAY_memory_usage_threshold=0.9
set RAY_memory_monitor_refresh_ms=1000
set OMP_NUM_THREADS=16
set RAY_MEMORY_LIMIT=25769803776

REM Run the experiment with high-performance configuration (shorter test)
echo Starting high-performance experiment on RTX A6000...
python -m src.experiments.run_experiment --config config/experiment_config.yaml

echo Experiment completed!
pause
