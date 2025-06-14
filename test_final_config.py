#!/usr/bin/env python3
"""
🧪 TEST FINAL DE CONFIGURACIÓN COMPLETA
========================================
Prueba todas las configuraciones necesarias para el trading bot
"""

import os
import sys
import asyncio
import aiohttp
from dotenv import load_dotenv

async def test_binance_connection():
    """Prueba la conexión con Binance usando las credenciales del .env"""
    print("🔧 Probando conexión con Binance...")
    
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    base_url = os.getenv('BINANCE_BASE_URL')
    
    if not api_key or not base_url:
        print("❌ Credenciales de Binance no encontradas")
        return False
    
    try:
        headers = {'X-MBX-APIKEY': api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/v3/exchangeInfo", headers=headers) as response:
                if response.status == 200:
                    print("✅ Conexión con Binance exitosa")
                    return True
                else:
                    print(f"❌ Error de conexión Binance: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error conectando con Binance: {e}")
        return False

async def test_discord_webhook():
    """Prueba el webhook de Discord"""
    print("🔧 Probando Discord webhook...")
    
    discord_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not discord_url:
        print("❌ Discord webhook no encontrado")
        return False
    
    # Limpiar comillas si las hay
    discord_url = discord_url.strip('"')
    
    try:
        test_message = {
            "content": "🧪 **TEST BOT CONFIGURATION**\n✅ Bot configurado correctamente\n🎯 Sistema listo para trading"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(discord_url, json=test_message) as response:
                if response.status in [200, 204]:
                    print("✅ Discord webhook funcionando")
                    return True
                else:
                    print(f"❌ Error Discord webhook: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error enviando a Discord: {e}")
        return False

def test_environment_variables():
    """Verifica todas las variables de entorno necesarias"""
    print("🔧 Verificando variables de entorno...")
    
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_SECRET_KEY', 
        'DISCORD_WEBHOOK_URL',
        'BINANCE_BASE_URL',
        'ENVIRONMENT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Variables faltantes: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ Todas las variables de entorno presentes")
        return True

def test_model_files():
    """Verifica que los archivos de modelos existen"""
    print("🔧 Verificando archivos de modelos TCN...")
    
    models_dir = "models"
    required_models = [
        "tcn_final_btcusdt.h5",
        "tcn_final_ethusdt.h5", 
        "tcn_final_bnbusdt.h5",
        "feature_scalers_fixed.pkl"
    ]
    
    missing_models = []
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if not os.path.exists(model_path):
            missing_models.append(model)
    
    if missing_models:
        print(f"❌ Modelos faltantes: {', '.join(missing_models)}")
        return False
    else:
        print("✅ Todos los modelos TCN encontrados")
        return True

async def run_complete_test():
    """Ejecuta todas las pruebas"""
    print("🚀 INICIANDO PRUEBA COMPLETA DE CONFIGURACIÓN")
    print("=" * 60)
    
    load_dotenv()
    
    # Verificar variables de entorno
    env_ok = test_environment_variables()
    print()
    
    # Verificar modelos
    models_ok = test_model_files()
    print()
    
    # Verificar conexión Binance
    binance_ok = await test_binance_connection()
    print()
    
    # Verificar Discord webhook
    discord_ok = await test_discord_webhook()
    print()
    
    print("=" * 60)
    print("📊 RESUMEN DE PRUEBAS:")
    print(f"   {'✅' if env_ok else '❌'} Variables de entorno")
    print(f"   {'✅' if models_ok else '❌'} Archivos de modelos TCN")
    print(f"   {'✅' if binance_ok else '❌'} Conexión Binance API")
    print(f"   {'✅' if discord_ok else '❌'} Discord webhook")
    
    all_ok = env_ok and models_ok and binance_ok and discord_ok
    
    print("=" * 60)
    if all_ok:
        print("🎉 ¡CONFIGURACIÓN COMPLETAMENTE EXITOSA!")
        print("🚀 El bot está listo para trading en vivo")
        print("⚠️ RECORDATORIO: Estás en modo PRODUCCIÓN con dinero real")
    else:
        print("❌ CONFIGURACIÓN INCOMPLETA")
        print("🔧 Revisa los errores arriba antes de continuar")
    
    return all_ok

if __name__ == "__main__":
    asyncio.run(run_complete_test()) 