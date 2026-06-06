param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipInstall,
    [switch]$NoBrowser,
    [switch]$Hidden
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$BackendVenv = Join-Path $BackendDir ".venv"
$VenvPython = Join-Path $BackendVenv "Scripts\python.exe"
$BackendLogDir = Join-Path $BackendDir "logs"
$FrontendLogDir = Join-Path $FrontendDir "logs"

function Test-PortListening {
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($null -ne $connection) {
        return $true
    }

    $netstatLine = netstat -ano | Select-String -Pattern "LISTENING\s+\d+$" | Where-Object {
        $_.Line -match "[:.]$Port\s+"
    } | Select-Object -First 1
    return $null -ne $netstatLine
}

function Get-PythonCommand {
    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pythonLauncher) {
        return @{
            File = $pythonLauncher.Source
            Args = @("-3")
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            File = $python.Source
            Args = @()
        }
    }

    throw "Python was not found. Install Python 3.11+ and retry."
}

Write-Host "Emotion Chatbot local startup"
Write-Host "Root: $RootDir"

if (-not (Test-Path $BackendDir)) {
    throw "Backend directory not found: $BackendDir"
}

if (-not (Test-Path $FrontendDir)) {
    throw "Frontend directory not found: $FrontendDir"
}

$envFile = Join-Path $BackendDir ".env"
$envExample = Join-Path $BackendDir ".env.example"
if (-not (Test-Path $envFile)) {
    Copy-Item -Path $envExample -Destination $envFile
    Write-Warning "Created backend\.env from backend\.env.example. Fill GLM_API_KEY before using chat."
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating backend virtual environment..."
    $pythonCommand = Get-PythonCommand
    & $pythonCommand.File @($pythonCommand.Args) -m venv $BackendVenv
}

if (-not $SkipInstall) {
    Write-Host "Checking backend dependencies..."
    $dependencyCheck = "import importlib.util, sys; modules=['fastapi','uvicorn','httpx','torch','transformers','pandas','sklearn']; missing=[m for m in modules if importlib.util.find_spec(m) is None]; print(','.join(missing)); sys.exit(1 if missing else 0)"
    $missingDependencies = & $VenvPython -c $dependencyCheck 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($missingDependencies) {
            Write-Host "Missing backend packages: $missingDependencies"
        }
        Write-Host "Installing backend dependencies..."
        & $VenvPython -m pip install -r (Join-Path $BackendDir "requirements.txt")
    }

    if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
        Write-Host "Installing frontend dependencies..."
        Push-Location $FrontendDir
        npm install
        Pop-Location
    }
}

if (Test-PortListening -Port $BackendPort) {
    Write-Host "Backend port $BackendPort is already in use. Reusing existing backend."
} else {
    Write-Host "Starting backend on http://127.0.0.1:$BackendPort ..."
    New-Item -ItemType Directory -Path $BackendLogDir -Force | Out-Null
    if ($Hidden) {
        $backendOutLog = Join-Path $BackendLogDir "server.out.log"
        $backendErrLog = Join-Path $BackendLogDir "server.err.log"
        $backendCommand = "cd /d `"$BackendDir`" && `"$VenvPython`" -m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort 1>> `"$backendOutLog`" 2>> `"$backendErrLog`""
        Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $backendCommand) -WindowStyle Hidden
    } else {
        $backendCommand = "title Emotion Chatbot Backend && cd /d `"$BackendDir`" && `"$VenvPython`" -m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort"
        Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $backendCommand) -WorkingDirectory $BackendDir
    }
}

if (Test-PortListening -Port $FrontendPort) {
    Write-Host "Frontend port $FrontendPort is already in use. Reusing existing frontend."
} else {
    Write-Host "Starting frontend on http://127.0.0.1:$FrontendPort ..."
    New-Item -ItemType Directory -Path $FrontendLogDir -Force | Out-Null
    if ($Hidden) {
        $frontendOutLog = Join-Path $FrontendLogDir "server.out.log"
        $frontendErrLog = Join-Path $FrontendLogDir "server.err.log"
        $frontendCommand = "cd /d `"$FrontendDir`" && npm.cmd run dev -- --host 127.0.0.1 --port $FrontendPort 1>> `"$frontendOutLog`" 2>> `"$frontendErrLog`""
        Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $frontendCommand) -WindowStyle Hidden
    } else {
        $frontendCommand = "title Emotion Chatbot Frontend && cd /d `"$FrontendDir`" && npm.cmd run dev -- --host 127.0.0.1 --port $FrontendPort"
        Start-Process -FilePath "cmd.exe" -ArgumentList @("/k", $frontendCommand) -WorkingDirectory $FrontendDir
    }
}

$frontendUrl = "http://127.0.0.1:$FrontendPort/"
$backendUrl = "http://127.0.0.1:$BackendPort/"

Write-Host ""
Write-Host "Frontend: $frontendUrl"
Write-Host "Backend:  $backendUrl"
if ($Hidden) {
    Write-Host "Logs:"
    Write-Host "  backend\logs\server.out.log"
    Write-Host "  backend\logs\server.err.log"
    Write-Host "  frontend\logs\server.out.log"
    Write-Host "  frontend\logs\server.err.log"
}
Write-Host "Use stop-local.bat or close the backend/frontend command windows to stop the services."

if (-not $NoBrowser) {
    Start-Sleep -Seconds 3
    Start-Process $frontendUrl
}
