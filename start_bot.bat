@echo off
REM 🚀 BINANCE TRADING BOT - INICIO RÁPIDO (.BAT)
REM =============================================

echo 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
echo 🚀 BINANCE TRADING BOT - PROFESSIONAL
echo 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
echo ⚡ Trailing Stop Automático: HABILITADO
echo.

REM Cambiar al directorio del proyecto
cd /d "C:\Users\emman\Desktop\BinanceBot"

REM Verificar que estamos en el directorio correcto
if not exist "run_trading_manager.py" (
    echo ❌ ERROR: Archivo run_trading_manager.py no encontrado
    echo    Verifica que estés en el directorio correcto
    pause
    exit /b 1
)

echo 📂 Directorio: %CD%
echo 🚀 Iniciando Trading Manager...
echo 💡 Usa Ctrl+C para detener
echo.

REM Ejecutar el trading manager
python run_trading_manager.py

echo.
echo 🎯 Trading Manager finalizado
pause 