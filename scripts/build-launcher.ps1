$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $compiler /nologo /target:winexe /reference:System.Windows.Forms.dll "/out:$projectRoot\FastMatch.exe" "$PSScriptRoot\Launcher.cs"
if ($LASTEXITCODE -ne 0) { throw 'EXE build failed' }
Write-Host "Built $projectRoot\FastMatch.exe"
