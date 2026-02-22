@echo off
REM Parse Analysis Tool - Windows 批处理脚本
REM 用法: parse_analysis.bat <input_file> [output_file]

if "%~1"=="" (
    echo 用法: parse_analysis.bat input_file [output_file]
    echo.
    echo 示例:
    echo   parse_analysis.bat analysis.txt
    echo   parse_analysis.bat analysis.txt output.json
    exit /b 1
)

set INPUT_FILE=%~1
set OUTPUT_FILE=%~2

if "%OUTPUT_FILE%"=="" (
    python tools\parse_analysis.py "%INPUT_FILE%"
) else (
    python tools\parse_analysis.py "%INPUT_FILE%" -o "%OUTPUT_FILE%"
)
