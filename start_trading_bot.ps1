# 🚀 BINANCE TRADING BOT - INICIO RÁPIDO
# ========================================
# Script PowerShell para ejecutar el trading manager desde cualquier ubicación

param(
    [switch]$Test,          # -Test para modo testing
    [switch]$Background,    # -Background para ejecutar en segundo plano
    [switch]$Status,        # -Status para ver estado del bot
    [switch]$Stop          # -Stop para detener el bot
)

# Configuración
$ProjectPath = "C:\Users\emman\Desktop\BinanceBot"
$MainScript = "run_trading_manager.py"
$TestScript = "test_trailing_stop_automatico.py"

# Colores para output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# Banner
Write-ColorOutput Green @"
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
🚀 BINANCE TRADING BOT - PROFESSIONAL
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
📍 Proyecto: $ProjectPath
⚡ Trailing Stop Automático: HABILITADO
"@

# Verificar que el directorio existe
if (-not (Test-Path $ProjectPath)) {
    Write-ColorOutput Red "❌ ERROR: Directorio del proyecto no encontrado"
    Write-Host "   Ruta esperada: $ProjectPath"
    exit 1
}

# Cambiar al directorio del proyecto
try {
    Set-Location $ProjectPath
    Write-ColorOutput Yellow "📂 Directorio cambiado a: $ProjectPath"
} catch {
    Write-ColorOutput Red "❌ Error cambiando directorio: $_"
    exit 1
}

# Verificar que Python está disponible
try {
    $pythonVersion = python --version 2>&1
    Write-ColorOutput Green "✅ Python disponible: $pythonVersion"
} catch {
    Write-ColorOutput Red "❌ Python no encontrado. Instala Python o agrégalo al PATH"
    exit 1
}

# Ejecutar según parámetros
if ($Test) {
    Write-ColorOutput Cyan "🧪 Ejecutando modo TEST..."
    python $TestScript
} 
elseif ($Status) {
    Write-ColorOutput Cyan "📊 Verificando estado del bot..."
    # Verificar si hay procesos Python ejecutándose
    $pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
    if ($pythonProcesses) {
        Write-ColorOutput Green "✅ Bot ejecutándose - Procesos Python activos: $($pythonProcesses.Count)"
        $pythonProcesses | Format-Table Id, ProcessName, StartTime -AutoSize
    } else {
        Write-ColorOutput Yellow "⚠️ No se detectaron procesos Python activos"
    }
}
elseif ($Stop) {
    Write-ColorOutput Yellow "🛑 Deteniendo bot..."
    $pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
    if ($pythonProcesses) {
        $pythonProcesses | Stop-Process -Force
        Write-ColorOutput Green "✅ Procesos Python detenidos"
    } else {
        Write-ColorOutput Yellow "⚠️ No hay procesos Python para detener"
    }
}
elseif ($Background) {
    Write-ColorOutput Cyan "🚀 Ejecutando en segundo plano..."
    Start-Process -FilePath "python" -ArgumentList $MainScript -WindowStyle Hidden
    Write-ColorOutput Green "✅ Bot iniciado en segundo plano"
}
else {
    Write-ColorOutput Cyan "🚀 Ejecutando Trading Manager..."
    Write-ColorOutput Yellow "   💡 Usa Ctrl+C para detener"
    Write-ColorOutput Yellow "   📱 Revisa Discord para notificaciones"
    Write-Host ""
    
    # Ejecutar el script principal
    try {
        python $MainScript
    } catch {
        Write-ColorOutput Red "❌ Error ejecutando el trading manager: $_"
        exit 1
    }
}

Write-ColorOutput Green "🎯 Operación completada" 