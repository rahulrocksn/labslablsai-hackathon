@echo off
REM setup_windows.bat
REM Script to set up the environment and run the Mirror app on a Vultr Windows Server.

REM ----------------------------------------------------------------------
REM Auto-Elevate to Admin if not running as Admin
REM ----------------------------------------------------------------------
fsutil dirty query %systemdrive% >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrative Privileges...
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /b
)

REM ----------------------------------------------------------------------
REM Script content starts here (running as Admin)
REM ----------------------------------------------------------------------
cd /d "%~dp0"

REM 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.
    echo opening python download page...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 2. Install Dependencies
echo Installing Python requirements...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found! Please ensure you are in the project directory.
    pause
    exit /b 1
)

REM 3. Fix for MediaPipe
pip install mediapipe --upgrade

REM 4. Configure Firewall (Requires Admin) for Port 80 (HTTP)
echo Configuring firewall...
netsh advfirewall firewall add rule name="Streamlit HTTP" dir=in action=allow protocol=TCP localport=80

REM 5. Get IP Address from .env or Detect
set PUBLIC_IP=
if exist .env (
    echo Reading IP from .env file...
    for /f "tokens=1,2 delims==" %%A in (.env) do (
        if "%%A"=="VULTR_PUBLIC_IP" set PUBLIC_IP=%%B
    )
)

if "%PUBLIC_IP%"=="" (
    echo Detecting Public IP Address (Auto)...
    for /f "tokens=*" %%a in ('curl -s ifconfig.me') do set PUBLIC_IP=%%a
)

if "%PUBLIC_IP%"=="" (
    set PUBLIC_IP=localhost
)

REM Trim whitespace from IP (PowerShell logic to trim if needed, but simple batch replacement usually suffices)
set PUBLIC_IP=%PUBLIC_IP: =%

REM 6. Run the application on Port 80
echo.
echo ========================================================
echo Setup complete! Running Streamlit app...
echo Access your app at: http://%PUBLIC_IP%/
echo ========================================================
echo.

REM Run Streamlit on Port 80 (HTTP Default)
streamlit run app.py --server.port 80 --server.address 0.0.0.0

pause
