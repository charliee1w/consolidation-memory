$json = [Console]::In.ReadToEnd() | ConvertFrom-Json
$command = $json.tool_input.command
if ($command -match 'git\s+push') {
    $lastTag = & git -C "D:\consolidation-memory" describe --tags --abbrev=0 2>$null
    if ($lastTag) {
        $count = & git -C "D:\consolidation-memory" rev-list --count "$lastTag..HEAD" 2>$null
        Write-Output "Pushed to remote. $count commits since last release ($lastTag). Evaluate whether a new release is warranted: new features = minor bump, bug fixes = patch bump. Skip for docs/test/refactor-only changes."
    }
}
exit 0
