# =================================================================
# 🚀 Script de Inicio Simple y Robusto para Trading Bot
# =================================================================
# USO:
#   .\start_bot.ps1
#
# NOTA SOBRE PERMISOS:
#   Si recibes un error de "ejecución de scripts deshabilitada",
#   ejecuta este comando una sola vez en PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# =================================================================

# --- 1. Configuración ---
# Detiene el script inmediatamente si cualquier comando falla.
$ErrorActionPreference = "Stop"
Clear-Host
Write-Host "🚀 Iniciando Bot de Trading..."

# --- 2. Activar Entorno Virtual ---
# Busca el script de activación en las carpetas comunes 'venv' y '.venv'.
$activateScriptPath = ""
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    $activateScriptPath = ".\venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    $activateScriptPath = ".\.venv\Scripts\Activate.ps1"
}

# Si no se encuentra, muestra un error y sale.
if (-not $activateScriptPath) {
    Write-Host "❌ ERROR: No se encontró el entorno virtual ('venv' o '.venv')."
    Write-Host "   Por favor, crea uno con el comando: python -m venv venv"
    exit 1
}

# Activa el entorno.
. $activateScriptPath
Write-Host "✅ Entorno virtual activado."

# --- 3. Ejecutar el Bot ---
Write-Host "▶️  Ejecutando el bot de trading..."
Write-Host "   (Presiona Ctrl+C para detener en cualquier momento)"
Write-Host "----------------------------------------------------"

# Ejecuta el script de Python. El bloque 'finally' se asegura
# de que el mensaje de despedida aparezca incluso si detienes
# el bot con Ctrl+C.
try {
    python run_trading_manager.py
}
finally {
    Write-Host "----------------------------------------------------"
    Write-Host "👋 Sesión de trading finalizada."
} 