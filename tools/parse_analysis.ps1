#!/bin/bash
# Parse Analysis Tool - PowerShell 脚本
# 用法: ./parse_analysis.ps1 <input_file> [output_file]

param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputFile = ""
)

if ($OutputFile -eq "") {
    python tools/parse_analysis.py $InputFile
} else {
    python tools/parse_analysis.py $InputFile -o $OutputFile
}
