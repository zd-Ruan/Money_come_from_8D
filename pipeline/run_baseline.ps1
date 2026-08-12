$ErrorActionPreference = "Stop"
$pipelineRoot = $PSScriptRoot
$workspaceRoot = Split-Path $pipelineRoot -Parent
$env:PYTHONPATH = Join-Path $pipelineRoot "src"
Set-Location $workspaceRoot
& "C:\Exception\quant\python.exe" -m quant_pipeline.cli run
exit $LASTEXITCODE
