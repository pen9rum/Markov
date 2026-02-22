# 批量实验工具 - PowerShell版本

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcDir = Join-Path $ScriptDir "..\src"
$ToolsDir = Join-Path $ScriptDir "."

Set-Location $SrcDir

python (Join-Path $ToolsDir "batch_experiment.py")

if ($LASTEXITCODE -ne 0) {
    Write-Host "实验执行失败" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n按任意键退出..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
