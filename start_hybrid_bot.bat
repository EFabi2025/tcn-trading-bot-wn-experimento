@echo off
REM =================================================================
REM 🚀 Script de Inicio para Trading Bot con Motor Híbrido (Windows)
REM =================================================================
REM USO: Doble click en este archivo o ejecutar desde cmd
REM
REM CARACTERÍSTICAS:
REM   ✅ Motor de Features Híbridas optimizado
REM   ✅ Mejora de confianza hasta +12% en predicciones
REM   ✅ Fallback automático al motor original
REM   ✅ Calidad de features: 0.84/1.0
REM =================================================================

cls
echo 🚀 INICIANDO BOT DE TRADING CON MOTOR HÍBRIDO
echo ============================================================
echo 🔧 Motor de Features: HÍBRIDO OPTIMIZADO
echo 🎯 Umbrales: CONFIGURABLES VÍA .env
echo 🛡️ Seguridad: FALLBACK AUTOMÁTICO
echo ============================================================
echo.

REM --- 1. Verificar Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python no está instalado o no está en PATH
    echo    Por favor, instala Python desde https://python.org
    pause
    exit /b 1
)
echo ✅ Python detectado

REM --- 2. Activar Entorno Virtual ---
echo 📦 Activando entorno virtual...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✅ Entorno virtual 'venv' activado
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✅ Entorno virtual '.venv' activado
) else (
    echo ❌ ERROR: No se encontró entorno virtual
    echo    Crea uno con: python -m venv venv
    pause
    exit /b 1
)

REM --- 3. Verificar Configuración ---
echo 📋 Verificando configuración...
if not exist ".env" (
    echo ❌ ERROR: No se encontró el archivo .env
    echo    Por favor, crea un archivo .env basado en config_example.env
    echo    Consulta CAMPOS_ENV_REQUERIDOS.md para más información
    pause
    exit /b 1
)
echo ✅ Archivo .env encontrado

REM --- 4. Ejecutar Bot con Motor Híbrido ---
echo.
echo ▶️  Ejecutando bot con motor híbrido...
echo 📊 Características activas:
echo    ✅ Features híbridas limpias y optimizadas
echo    ✅ Mejora de confianza en predicciones
echo    ✅ Gestión avanzada de riesgo
echo    ✅ Notificaciones Discord inteligentes
echo    (Presiona Ctrl+C para detener en cualquier momento)
echo ------------------------------------------------------------

REM Intentar primero el motor híbrido
if exist "start_hybrid_trading.py" (
    echo 🔧 Usando motor híbrido optimizado...
    python start_hybrid_trading.py
) else if exist "run_trading_manager.py" (
    echo ⚠️  Usando motor original como fallback...
    python run_trading_manager.py
) else (
    echo ❌ ERROR: No se encontró ningún script de trading
    pause
    exit /b 1
)

REM --- 5. Mensaje de Finalización ---
echo.
echo ------------------------------------------------------------
echo 🏁 Sesión de trading con motor híbrido finalizada.
echo 📊 Gracias por usar el Bot TCN con Features Optimizadas
echo.
pause 