@echo off
rem Usage: run_agent.bat C:\path\to\your\project [extra cli args]
rem Example: run_agent.bat C:\work\myapp chat
setlocal
set WS=%~1
if "%WS%"=="" (echo Usage: run_agent.bat ^<workspace^> [chat^|ask "task"^|status^|index^|verify-model] & exit /b 1)
shift
python "%~dp0cli.py" --workspace "%WS%" %1 %2 %3 %4 %5 %6 %7 %8 %9
