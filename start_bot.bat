@echo off
REM =================================================================
REM 🚀 SCRIPT DE INICIO PARA TRADING BOT - CMD
REM =================================================================
REM Este script automatiza el inicio del bot de trading en Windows.
REM
REM Uso:
REM 1. Abre un Símbolo del sistema (CMD)
REM 2. Navega a la carpeta del proyecto: cd C:\\Ruta\\A\\Tu\\Proyecto
REM 3. Ejecuta este script: start_bot.bat
REM =================================================================

ECHO =================================================
ECHO   🚀 INICIANDO ENTORNO DEL BOT DE TRADING TCN
ECHO =================================================
ECHO.

REM --- PASO 1: VERIFICAR ENTORNO VIRTUAL ---
IF NOT EXIST ".\\venv\\Scripts\\activate.bat" (
    ECHO ❌ ERROR: Entorno virtual no encontrado.
    ECHO    Por favor, asegúrate de haber creado el entorno con 'python -m venv .venv'
    GOTO :EOF
)
ECHO ✅ Entorno virtual encontrado.
ECHO.

REM --- PASO 2: ACTIVAR ENTORNO VIRTUAL ---
ECHO 🔧 Activando entorno virtual...
CALL .\\venv\\Scripts\\activate.bat
ECHO ✅ Entorno virtual activado.
ECHO.

REM --- PASO 3: VERIFICAR SCRIPT PRINCIPAL ---
IF NOT EXIST "run_trading_manager.py" (
    ECHO ❌ ERROR: Script principal 'run_trading_manager.py' no encontrado.
    GOTO :EOF
)
ECHO ✅ Script principal encontrado.
ECHO.

REM --- PASO 4: EJECUTAR EL BOT ---
ECHO =================================================
ECHO   ▶️  EJECUTANDO EL BOT DE TRADING...
ECHO   🛑 (Presiona Ctrl+C para detener el bot)
ECHO =================================================
ECHO.

python run_trading_manager.py

ECHO.
ECHO =================================================
ECHO   👋 SESIÓN DE TRADING FINALIZADA
ECHO =================================================

REM Desactivar el entorno (opcional)
CALL .\\venv\\Scripts\\deactivate.bat
:EOF 