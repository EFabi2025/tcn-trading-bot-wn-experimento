#!/usr/bin/env python3
"""
🔥 ACTIVAR TRADING REAL
Script para cambiar la configuración de simulación a trading real
"""

import os
from datetime import datetime

def crear_env_trading_real():
    """🔥 Crear archivo .env para trading real"""

    print("🔥 CONFIGURANDO TRADING REAL")
    print("=" * 50)

    # Leer configuración actual
    current_config = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    current_config[key] = value
    except FileNotFoundError:
        print("⚠️ Archivo .env no encontrado. Creando desde plantilla...")
        # Usar valores de placeholder en lugar de keys reales
        current_config = {
            'BINANCE_API_KEY': 'TU_API_KEY_AQUI',
            'BINANCE_SECRET_KEY': 'TU_SECRET_KEY_AQUI',
            'DISCORD_WEBHOOK_URL': 'TU_WEBHOOK_URL_AQUI'
        }

    # Hacer respaldo del .env actual
    if os.path.exists('.env'):
        backup_name = f'.env.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.rename('.env', backup_name)
        print(f"✅ Respaldo creado: {backup_name}")

    # Configuración para trading real
    trading_real_config = f"""# 🚀 Professional Trading Bot - TRADING REAL ACTIVADO
# =====================================================
# ⚠️ CONFIGURACIÓN PARA TRADING REAL - DINERO REAL
# ⚠️ Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# 🔑 BINANCE API CREDENTIALS
BINANCE_API_KEY={current_config.get('BINANCE_API_KEY', 'TU_API_KEY')}
BINANCE_SECRET_KEY={current_config.get('BINANCE_SECRET_KEY', 'TU_SECRET_KEY')}

# 🌐 BINANCE ENVIRONMENT - PRODUCCIÓN
BINANCE_BASE_URL=https://api.binance.com
ENVIRONMENT=production

# 🔔 DISCORD NOTIFICATIONS
DISCORD_WEBHOOK_URL={current_config.get('DISCORD_WEBHOOK_URL', '')}

# 📊 TRADING CONFIGURATION - TRADING REAL
MAX_POSITION_SIZE_PERCENT=40
MAX_DAILY_LOSS_PERCENT=5
STOP_LOSS_PERCENT=3
TAKE_PROFIT_PERCENT=6
MAX_SIMULTANEOUS_POSITIONS=2
MIN_TRADE_VALUE_USDT=11

# 🎯 TRADING MODE - ¡TRADING REAL ACTIVADO!
TRADE_MODE=real        # 🔥 TRADING REAL - Ejecuta órdenes reales
DRY_RUN=false         # 🔥 NO SIMULACIÓN - Dinero real

# 📈 TECHNICAL ANALYSIS
DEFAULT_TIMEFRAME=1m
ANALYSIS_LOOKBACK=200

# 🔧 SYSTEM CONFIGURATION
CHECK_INTERVAL=60
TCN_REPORT_INTERVAL=300
LOG_LEVEL=INFO

# 🗄️ DATABASE
DATABASE_URL=sqlite:///trading_bot.db

# 🛡️ RISK MANAGEMENT
MIN_CONFIDENCE_THRESHOLD=0.70
MIN_TCN_CONFIDENCE=0.70
TRAILING_STOP_ACTIVATION_PERCENT=1.0
TRAILING_STOP_STEP_PERCENT=0.5

# 🔔 NOTIFICATION FILTERS
MIN_NOTIFICATION_TRADE_VALUE=12.0
MIN_NOTIFICATION_PNL_PERCENT=2.0
MAX_NOTIFICATIONS_PER_HOUR=8
MAX_NOTIFICATIONS_PER_DAY=40
SUPPRESS_SIMILAR_NOTIFICATIONS_MINUTES=10
ONLY_PROFITABLE_TRADES=false

# 🧪 DESARROLLO Y DEBUG
DEBUG=false
SAVE_TCN_PREDICTIONS=true
VERBOSE_API_LOGGING=false

# ⚠️ ADVERTENCIAS DE SEGURIDAD
# ============================
# - Este archivo ejecuta TRADES REALES con DINERO REAL
# - Monitorea constantemente el bot
# - Ten un plan de salida
# - Usa stop loss apropiados
"""

    # Escribir nueva configuración
    with open('.env', 'w') as f:
        f.write(trading_real_config)

    print("✅ Archivo .env creado para TRADING REAL")
    print()
    print("🔥 CAMBIOS REALIZADOS:")
    print("   TRADE_MODE=real       (antes: dry_run)")
    print("   DRY_RUN=false        (antes: true)")
    print("   ENVIRONMENT=production")
    print()
    print("⚠️ IMPORTANTE:")
    print("   - Ahora el bot ejecutará ÓRDENES REALES")
    print("   - Usa DINERO REAL de tu cuenta Binance")
    print("   - Monitorea constantemente")
    print("   - Verifica tu balance actual en Binance")

    return True

def verificar_trading_real():
    """✅ Verificar que el trading real esté activado"""

    try:
        from dotenv import load_dotenv
        load_dotenv()

        trade_mode = os.getenv('TRADE_MODE')
        dry_run = os.getenv('DRY_RUN', 'true').lower()

        print("\n🔍 VERIFICACIÓN POST-CONFIGURACIÓN:")
        print(f"   TRADE_MODE: {trade_mode}")
        print(f"   DRY_RUN: {dry_run}")

        if trade_mode == 'real' and dry_run == 'false':
            print("✅ TRADING REAL ACTIVADO CORRECTAMENTE")
            return True
        else:
            print("❌ Configuración incorrecta")
            return False

    except Exception as e:
        print(f"❌ Error verificando: {e}")
        return False

def main():
    """🚀 Función principal"""

    print("⚠️ ADVERTENCIA CRÍTICA ⚠️")
    print("Este script activará el TRADING REAL con DINERO REAL")
    print("Las señales BUY se ejecutarán como órdenes reales en Binance")
    print()

    respuesta = input("¿Estás seguro de activar TRADING REAL? (si/no): ").lower()

    if respuesta in ['si', 'sí', 's', 'yes', 'y']:
        if crear_env_trading_real():
            verificar_trading_real()
            print("\n🎯 PRÓXIMOS PASOS:")
            print("1. Ejecutar: python run_trading_manager.py")
            print("2. Monitorear las notificaciones Discord")
            print("3. Verificar que se ejecuten órdenes reales")
    else:
        print("❌ Activación cancelada. El sistema permanece en modo simulación.")

if __name__ == "__main__":
    main()
