param([Parameter(Mandatory=$true)][string]$Manifest)
$ErrorActionPreference = 'Stop'
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$root = Split-Path -Parent $manifestPath
$metadata = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ([System.IO.Path]::GetFileName($metadata.archive) -ne $metadata.archive) {
    throw 'Archive must be a filename'
}
$output = Join-Path $root $metadata.archive
$temporary = "$output.assembling"
try {
    $destination = [System.IO.File]::Create($temporary)
    try {
        foreach ($part in $metadata.parts) {
            if ([System.IO.Path]::GetFileName($part.name) -ne $part.name) { throw 'Part must be a filename' }
            $path = Join-Path $root $part.name
            if ((Get-Item -LiteralPath $path).Length -ne $part.size -or
                (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash -ne $part.sha256) {
                throw "Corrupt part: $($part.name)"
            }
            $source = [System.IO.File]::OpenRead($path)
            try { $source.CopyTo($destination) } finally { $source.Dispose() }
        }
    } finally { $destination.Dispose() }
    if ((Get-Item -LiteralPath $temporary).Length -ne $metadata.size -or
        (Get-FileHash -Algorithm SHA256 -LiteralPath $temporary).Hash -ne $metadata.sha256) {
        throw 'Assembled archive checksum mismatch'
    }
    Move-Item -Force -LiteralPath $temporary -Destination $output
    Write-Output $output
} catch {
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $temporary
    throw
}
