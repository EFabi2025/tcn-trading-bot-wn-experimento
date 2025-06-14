#!/usr/bin/env python3
"""
Script para agregar configuraciones faltantes al archivo .env
"""
import os

def update_env_file():
    # Leer el archivo .env actual
    env_content = []
    
    # Verificar si el archivo existe
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            env_content = f.readlines()
    
    # Configuraciones adicionales que necesitamos agregar
    additional_configs = [
        "\n# =============================================================================",
        "# 🌐 BINANCE BASE URL",
        "# =============================================================================",
        "BINANCE_BASE_URL=https://api.binance.com",
        "",
        "# =============================================================================", 
        "# 🛡️ CONFIGURACIÓN DE TRADING Y RIESGO",
        "# =============================================================================",
        "MAX_POSITION_SIZE_PERCENT=15.0",
        "MAX_SIMULTANEOUS_POSITIONS=2", 
        "MAX_DAILY_LOSS_PERCENT=10.0",
        "STOP_LOSS_PERCENT=3.0",
        "TAKE_PROFIT_PERCENT=6.0",
        "TRAILING_STOP_ENABLED=true",
        "TRAILING_STOP_PERCENT=2.0",
        "TRADE_MODE=live",
        "MIN_TRADE_VALUE_USDT=11.0",
        "CHECK_INTERVAL=60",
        "",
        "# =============================================================================",
        "# 🤖 TCN MODEL CONFIGURATION",
        "# =============================================================================",
        "TCN_MODEL_PATH=models/",
        "TCN_LOOKBACK_WINDOW=50",
        "TCN_FEATURE_COUNT=21",
        "TCN_CONFIDENCE_THRESHOLD=0.7",
        "ENABLED_PAIRS=BTCUSDT,ETHUSDT,BNBUSDT",
        "PRIMARY_QUOTE_ASSET=USDT",
        "",
        "# =============================================================================",
        "# 📊 NOTIFICATION FILTERS",
        "# =============================================================================",
        "MIN_NOTIFICATION_TRADE_VALUE=12.0",
        "MIN_NOTIFICATION_PNL_PERCENT=2.0",
        "MAX_NOTIFICATIONS_PER_HOUR=8",
        "MAX_NOTIFICATIONS_PER_DAY=40",
        "SUPPRESS_SIMILAR_NOTIFICATIONS_MINUTES=10",
        "ONLY_PROFITABLE_TRADES=false",
        "",
        "# =============================================================================",
        "# 🗄️ DATABASE & LOGGING",
        "# =============================================================================",
        "DATABASE_URL=sqlite:///trading_bot.db",
        "DEBUG=false",
        "",
        "# =============================================================================",
        "# ⚠️ EMERGENCY CONTROLS",
        "# =============================================================================",
        "EMERGENCY_STOP_LOSS_PERCENT=15.0",
        "MAX_TRADES_PER_HOUR=10",
        ""
    ]
    
    # Verificar qué configuraciones ya existen
    existing_keys = set()
    for line in env_content:
        if '=' in line and not line.strip().startswith('#'):
            key = line.split('=')[0].strip()
            existing_keys.add(key)
    
    print("🔧 AGREGANDO CONFIGURACIONES FALTANTES:")
    print("=" * 50)
    
    # Agregar solo las configuraciones que no existen
    new_configs = []
    for config in additional_configs:
        if '=' in config and not config.strip().startswith('#'):
            key = config.split('=')[0].strip()
            if key not in existing_keys:
                new_configs.append(config + '\n')
                print(f"✅ Agregando: {key}")
        else:
            # Agregar comentarios y líneas vacías
            new_configs.append(config + '\n')
    
    if new_configs:
        # Crear una copia del archivo original
        if os.path.exists('.env'):
            with open('.env.backup', 'w', encoding='utf-8') as f:
                f.writelines(env_content)
            print(f"📋 Copia de seguridad creada: .env.backup")
        
        # Escribir el archivo actualizado
        with open('.env', 'a', encoding='utf-8') as f:
            f.writelines(new_configs)
        
        print("=" * 50)
        print("🎯 CONFIGURACIONES AGREGADAS EXITOSAMENTE")
    else:
        print("✅ Todas las configuraciones ya están presentes")

if __name__ == "__main__":
    update_env_file() 