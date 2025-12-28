@echo off
REM Simple wrapper script for running ONNX benchmarks on Windows
REM Edit the variables below to match your setup

REM Configuration
set BENCHMARK_URL=http://localhost:5173
set MODELS_JSON=./public/models/models.json
set BACKEND=wasm
set WARMUP=1
set MEASURE=100
set OUTPUT_FORMAT=csv

echo ==========================================
echo ONNX Runtime Web Benchmark Runner
echo ==========================================
echo.
echo Configuration:
echo   URL: %BENCHMARK_URL%
echo   Models: %MODELS_JSON%
echo   Backend: %BACKEND%
echo   Warmup: %WARMUP% runs
echo   Measure: %MEASURE% runs
echo   Format: %OUTPUT_FORMAT%
echo.
echo ==========================================
echo.

REM Check if Python script exists
if not exist run_benchmark_cli.py (
    echo ERROR: run_benchmark_cli.py not found!
    echo Make sure you're in the correct directory.
    pause
    exit /b 1
)

REM Check if models.json exists
if not exist %MODELS_JSON% (
    echo ERROR: %MODELS_JSON% not found!
    echo Please update MODELS_JSON path in this script.
    pause
    exit /b 1
)

REM Run the benchmark
python run_benchmark_cli.py ^
    --url %BENCHMARK_URL% ^
    --models %MODELS_JSON% ^
    --backend %BACKEND% ^
    --warmup %WARMUP% ^
    --measure %MEASURE% ^
    --format %OUTPUT_FORMAT%

echo.
echo ==========================================
echo Benchmark complete!
echo Check the benchmark_results\ directory for output
echo ==========================================
pause
