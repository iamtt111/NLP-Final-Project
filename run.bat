@echo off
chcp 65001 >nul
setlocal

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

:: 4. Build index if not exists
if not exist "data\bm25_cache.pkl" (
    echo [3/4] Building BM25 index from corpus...
    python build_corpus.py --corpus ./corpus
    if errorlevel 1 (
        echo [ERROR] Failed to build corpus index.
        pause
        exit /b 1
    )
) else (
    echo [3/4] BM25 index already exists, skipping. (Delete data\bm25_cache.pkl to rebuild)
)

:: 5. Run QA
echo [4/4] Running QA pipeline...
echo.
python qa.py --input input/questions.csv --output output/answers.csv --workers 4
if errorlevel 1 (
    echo [ERROR] QA pipeline failed.
    pause
    exit /b 1
)

echo.
echo ====================================
echo   Done! Check output\answers.csv
echo ====================================
pause
