# setup_full_access.ps1
# Configures Codex (desktop/CLI) for FULL ACCESS + auto-run without approval prompts.
#   - sandbox_mode  = "danger-full-access"  -> network + full filesystem access
#   - approval_policy = "never"             -> no approval prompts
# Usage (in a normal terminal, NOT inside the Codex sandbox):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_full_access.ps1
# After running: fully restart the Codex desktop app, then start a NEW thread.

$ErrorActionPreference = 'Stop'
$config = Join-Path $env:USERPROFILE '.codex\config.toml'

if (-not (Test-Path -LiteralPath $config)) {
  Write-Host "[ERROR] config.toml not found: $config"
  exit 1
}

$stamp = Get-Date -Format 'yyyyMMddHHmmss'
$backup = "$config.bak.$stamp"
Copy-Item -LiteralPath $config -Destination $backup -Force
Write-Host "backup: $backup"

$text = Get-Content -LiteralPath $config -Raw
if ($null -eq $text) { $text = '' }

# 1) sandbox_mode -> danger-full-access
if ($text -match '(?m)^sandbox_mode\s*=') {
  $text = $text -replace '(?m)^sandbox_mode\s*=.*$', 'sandbox_mode = "danger-full-access"'
} else {
  $text = $text.TrimEnd() + "`r`nsandbox_mode = `"danger-full-access`"`r`n"
}

# 2) approval_policy -> never (no prompts; combined with full access = auto-run)
if ($text -match '(?m)^approval_policy\s*=') {
  $text = $text -replace '(?m)^approval_policy\s*=.*$', 'approval_policy = "never"'
} else {
  $text = $text.TrimEnd() + "`r`napproval_policy = `"never`"`r`n"
}

[System.IO.File]::WriteAllText($config, $text, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "config updated: sandbox_mode=danger-full-access, approval_policy=never"
Write-Host "Now fully restart the Codex desktop app and start a NEW thread."
Write-Host "Security note: this is the most permissive mode; commands can modify anything on this machine."
