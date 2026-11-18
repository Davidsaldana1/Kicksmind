@echo off
setlocal
cd /d "%~dp0"
echo ================================
echo KicksMind V3 - Generar EXE
echo ================================
py -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo Necesitas PyInstaller: py -m pip install pyinstaller
    pause
    exit /b 1
)
if not exist "modelo_kicksmind_v3.joblib" (
    echo Falta el modelo V3. Ejecuta antes: py entrenar.py
    pause
    exit /b 1
)
py -m PyInstaller --noconfirm --clean --windowed --onedir ^
  --name KicksMind --add-data "modelo_kicksmind_v3.joblib;." ^
  --exclude-module matplotlib app.py
if errorlevel 1 (
  echo Hubo un problema generando el .exe.
  pause
  exit /b 1
)
echo EXE creado en dist\KicksMind\KicksMind.exe
echo Comparte TODO dist\KicksMind, no solamente el .exe.
pause
