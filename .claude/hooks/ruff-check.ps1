$json = [Console]::In.ReadToEnd() | ConvertFrom-Json
$file_path = $json.tool_input.file_path
if (-not $file_path) { exit 0 }
if ($file_path -notmatch '\.py$') { exit 0 }
if (-not (Test-Path $file_path)) { exit 0 }

$result = & python -m ruff check $file_path 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Output "ruff found issues in ${file_path}:"
    Write-Output $result.Trim()
}
exit 0
