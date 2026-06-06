param(
    [int[]]$Ports = @(8000, 5173)
)

$ErrorActionPreference = "Stop"

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    $processIds = @()

    if ($connections) {
        $processIds += $connections | Select-Object -ExpandProperty OwningProcess -Unique
    }

    if (-not $processIds) {
        $netstatLines = netstat -ano | Select-String -Pattern "LISTENING\s+(\d+)$" | Where-Object {
            $_.Line -match "[:.]$port\s+"
        }
        $processIds += $netstatLines | ForEach-Object {
            if ($_.Line -match "LISTENING\s+(\d+)$") {
                [int]$Matches[1]
            }
        } | Select-Object -Unique
    }

    if (-not $processIds) {
        Write-Host "Port $port is not listening."
        continue
    }

    foreach ($processId in $processIds) {
        try {
            $process = Get-Process -Id $processId -ErrorAction Stop
            Stop-Process -Id $processId -Force
            Write-Host "Stopped $($process.ProcessName) on port $port (PID $processId)."
        } catch {
            Write-Warning "Could not stop process ${processId} on port ${port}: $($_.Exception.Message)"
        }
    }
}
