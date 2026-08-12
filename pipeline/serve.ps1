$ErrorActionPreference = "Stop"
$pipelineRoot = $PSScriptRoot
$repositoryRoot = Split-Path $pipelineRoot -Parent
$launchRoot = Split-Path $repositoryRoot -Parent
$env:PYTHONPATH = Join-Path $pipelineRoot "src"
Set-Location $launchRoot
& "C:\Exception\quant\python.exe" -m quant_pipeline.cli serve --port 8765
exit $LASTEXITCODE
