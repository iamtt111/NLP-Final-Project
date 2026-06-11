@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ====================================
echo   NLP QA System - Quick Start
echo ====================================
echo.

:: 1. Create venv if not exists
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Make sure Python is installed.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists, skipping.
)

:: Activate venv
call venv\Scripts\activate.bat

:: 2. Install dependencies
echo [2/4] Installing dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 3. Check .env
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [WARNING] .env created from .env.example.
        echo           Please edit .env and fill in your API keys, then re-run.
        notepad .env
        pause
        exit /b 0
    ) else (
        echo [ERROR] .env.example not found.
        pause
        exit /b 1
    )
) else (
    echo [3/4] .env found.
)

:: 4. Build index (auto-detect corpus changes or force rebuild)
set "NEED_REBUILD=0"

:: Check --force-rebuild flag
for %%a in (%*) do (
    if "%%a"=="--force-rebuild" set "NEED_REBUILD=1"
)

if "!NEED_REBUILD!"=="1" (
    echo [3/4] Force rebuild requested.
) else if not exist "data\bm25_cache.pkl" (
    echo [3/4] No BM25 index found, building...
    set "NEED_REBUILD=1"
) else (
    :: Use Python to compare corpus mtime vs cache mtime (reliable on Windows)
    python -c "import os,sys,pathlib;c=max((f.stat().st_mtime for f in pathlib.Path('corpus').rglob('*') if f.is_file()),default=0);k=os.path.getmtime('data/bm25_cache.pkl');sys.exit(0 if c>k else 1)" 2>nul
    if !errorlevel! equ 0 (
        echo [3/4] Corpus has changed, rebuilding index...
        set "NEED_REBUILD=1"
    ) else (
        echo [3/4] BM25 index is up to date, skipping.
    )
)

if "!NEED_REBUILD!"=="1" (
    python build_corpus.py --corpus ./corpus
    if errorlevel 1 (
        echo [ERROR] Failed to build corpus index.
        pause
        exit /b 1
    )
)

:: 5. Select input CSV
echo.
echo [4/4] Select input CSV:
set "CSV_COUNT=0"
for %%f in (input\*.csv) do (
    set /a CSV_COUNT+=1
    set "CSV_!CSV_COUNT!=%%f"
    echo   !CSV_COUNT!. %%~nxf
)

if "!CSV_COUNT!"=="0" (
    echo [ERROR] No CSV files found in input\ folder.
    pause
    exit /b 1
)

if "!CSV_COUNT!"=="1" (
    set "SELECTED_CSV=!CSV_1!"
    echo   Auto-selected: !SELECTED_CSV!
) else (
    set /p "CSV_CHOICE=Enter number (1-!CSV_COUNT!): "
    call set "SELECTED_CSV=%%CSV_!CSV_CHOICE!%%"
    if "!SELECTED_CSV!"=="" (
        echo [ERROR] Invalid selection.
        pause
        exit /b 1
    )
)

:: 6. Build output filename: {input_file_name}_{timestamp}.csv
for %%f in ("!SELECTED_CSV!") do set "INPUT_NAME=%%~nf"
for /f "tokens=*" %%t in ('powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set "TIMESTAMP=%%t"
set "OUTPUT_FILE=output\!INPUT_NAME!_!TIMESTAMP!.csv"

:: 6. Run QA
echo.
echo Running QA pipeline...
echo   Input:  !SELECTED_CSV!
echo   Output: !OUTPUT_FILE!
echo.
python qa.py --input "!SELECTED_CSV!" --output "!OUTPUT_FILE!" --workers 4
if errorlevel 1 (
    echo [ERROR] QA pipeline failed.
    pause
    exit /b 1
)

:: 7. Open output
echo.
echo ====================================
echo   Done! Opening !OUTPUT_FILE!
echo ====================================
start "" "!OUTPUT_FILE!"
pause
