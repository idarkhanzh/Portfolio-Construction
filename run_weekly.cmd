@echo off
REM Weekly rebalance: build the portfolio, then publish it to Quantphemes.
REM Point Windows Task Scheduler at this file. Logs land in logs\run.log.

cd /d "%~dp0"
if not exist logs mkdir logs

REM --- credentials -----------------------------------------------------
REM Prefer setting these as Windows user environment variables (setx) so the
REM API key is not sitting in this file. Uncomment only as a last resort.
REM set QP_API_KEY=qc_live_...
set QP_PORTFOLIO_ID=d4ddf27b-9871-40a7-8063-5ff69ebd6fbb

echo. >> logs\run.log
echo ==== %DATE% %TIME% ==== >> logs\run.log

python portfolio_engine.py >> logs\run.log 2>&1
if errorlevel 1 (
    echo ENGINE FAILED - nothing published >> logs\run.log
    exit /b 1
)

python quantphemes_publish.py --verify >> logs\run.log 2>&1
if errorlevel 1 (
    echo PUBLISH FAILED - CSV written but targets not sent >> logs\run.log
    exit /b 1
)

echo OK >> logs\run.log
