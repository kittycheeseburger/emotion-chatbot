@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0backend\.env" (
  copy "%~dp0backend\.env.example" "%~dp0backend\.env" >nul
  echo Created backend\.env from backend\.env.example.
)

if not exist "%~dp0backend\.venv\Scripts\python.exe" (
  echo Creating backend virtual environment...
  py -3 -m venv "%~dp0backend\.venv" 2>nul || python -m venv "%~dp0backend\.venv"
)

echo Checking backend dependencies...
"%~dp0backend\.venv\Scripts\python.exe" -c "import importlib.util, sys; modules=['fastapi','uvicorn','httpx','torch','transformers','pandas','sklearn']; sys.exit(1 if any(importlib.util.find_spec(m) is None for m in modules) else 0)" 2>nul
if errorlevel 1 (
  echo Installing backend dependencies...
  "%~dp0backend\.venv\Scripts\python.exe" -m pip install -r "%~dp0backend\requirements.txt"
)

if not exist "%~dp0frontend\node_modules" (
  echo Installing frontend dependencies...
  cd /d "%~dp0frontend"
  call npm install
  cd /d "%~dp0"
)

start "Emotion Chatbot Backend" cmd /k ""%~dp0start-backend.bat""
start "Emotion Chatbot Frontend" cmd /k ""%~dp0start-frontend.bat""
timeout /t 3 >nul
start http://127.0.0.1:5173/

echo Frontend: http://127.0.0.1:5173/
echo Backend:  http://127.0.0.1:8000/
pause
