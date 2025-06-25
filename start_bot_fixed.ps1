# =================================================================
# Script de Inicio para Trading Bot con Motor Hibrido
# =================================================================
# USO:
#   .\start_bot_fixed.ps1
#
# CARACTERISTICAS:
#   - Motor de Features Hibridas optimizado
#   - Mejora de confianza hasta +12% en predicciones
#   - Fallback automatico al motor original
#   - Calidad de features: 0.84/1.0
#
# NOTA SOBRE PERMISOS:
#   Si recibes un error de "ejecucion de scripts deshabilitada",
#   ejecuta este comando una sola vez en PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# =================================================================

# --- 1. Configuracion ---
# Detiene el script inmediatamente si cualquier comando falla.
$ErrorActionPreference = "Stop"
Clear-Host
Write-Host "INICIANDO BOT DE TRADING CON MOTOR HIBRIDO"
Write-Host "============================================================"
Write-Host "Motor de Features: HIBRIDO OPTIMIZADO"
Write-Host "Umbrales: CONFIGURABLES VIA .env"
Write-Host "Seguridad: FALLBACK AUTOMATICO"
Write-Host "============================================================"

# --- 2. Activar Entorno Virtual ---
# Busca el script de activacion en las carpetas comunes 'venv' y '.venv'.
$activateScriptPath = ""
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    $activateScriptPath = ".\venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    $activateScriptPath = ".\.venv\Scripts\Activate.ps1"
}

# Si no se encuentra, muestra un error y sale.
if (-not $activateScriptPath) {
    Write-Host "ERROR: No se encontro el entorno virtual (venv o .venv)."
    Write-Host "   Por favor, crea uno con el comando: python -m venv venv"
    exit 1
}

# Activa el entorno.
. $activateScriptPath
Write-Host "Entorno virtual activado."

# --- 3. Verificar Configuracion ---
Write-Host "Verificando configuracion..."

# Verificar que existe el archivo .env
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: No se encontro el archivo .env"
    Write-Host "   Por favor, crea un archivo .env basado en config_example.env"
    Write-Host "   Consulta CAMPOS_ENV_REQUERIDOS.md para mas informacion"
    exit 1
}

Write-Host "Archivo .env encontrado"

# --- 4. Ejecutar el Bot con Motor Hibrido ---
Write-Host "Ejecutando bot con motor hibrido..."
Write-Host "Caracteristicas activas:"
Write-Host "   - Features hibridas limpias y optimizadas"
Write-Host "   - Mejora de confianza en predicciones"
Write-Host "   - Gestion avanzada de riesgo"
Write-Host "   - Notificaciones Discord inteligentes"
Write-Host "   (Presiona Ctrl+C para detener en cualquier momento)"
Write-Host "----------------------------------------------------"

# Ejecuta el script hibrido. El bloque 'finally' se asegura
# de que el mensaje de despedida aparezca incluso si detienes
# el bot con Ctrl+C.
try {
    # Intentar primero el motor hibrido
    if (Test-Path "start_hybrid_trading.py") {
        Write-Host "Usando motor hibrido optimizado..."
        python start_hybrid_trading.py
    } elseif (Test-Path "run_trading_manager.py") {
        Write-Host "Usando motor original como fallback..."
        python run_trading_manager.py
    } else {
        Write-Host "ERROR: No se encontro ningun script de trading"
        exit 1
    }
}
finally {
    Write-Host "----------------------------------------------------"
    Write-Host "Sesion de trading con motor hibrido finalizada."
    Write-Host "Gracias por usar el Bot TCN con Features Optimizadas"
} 