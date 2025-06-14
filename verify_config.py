#!/usr/bin/env python3
"""
Script para verificar la configuración del archivo .env
"""
import os
from dotenv import load_dotenv

def verify_config():
    # Cargar el archivo .env
    load_dotenv()
    
    print("🔧 VERIFICANDO CONFIGURACIÓN ACTUAL:")
    print("=" * 50)
    
    # Verificar credenciales de Binance
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if api_key:
        print(f"✅ BINANCE_API_KEY: {api_key[:20]}...{api_key[-10:]}")
    else:
        print("❌ BINANCE_API_KEY: NO ENCONTRADA")
    
    if secret_key:
        print(f"✅ BINANCE_SECRET_KEY: {secret_key[:20]}...{secret_key[-10:]}")
    else:
        print("❌ BINANCE_SECRET_KEY: NO ENCONTRADA")
    
    # Verificar Discord
    discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if discord_webhook:
        # Limpiar comillas si las hay
        discord_webhook = discord_webhook.strip('"')
        print(f"✅ DISCORD_WEBHOOK_URL: {discord_webhook[:50]}...{discord_webhook[-20:]}")
    else:
        print("❌ DISCORD_WEBHOOK_URL: NO ENCONTRADA")
    
    # Verificar configuración de ambiente
    environment = os.getenv('ENVIRONMENT', 'NO CONFIGURADO')
    testnet = os.getenv('BINANCE_TESTNET', 'NO CONFIGURADO')
    base_url = os.getenv('BINANCE_BASE_URL', 'NO CONFIGURADO')
    
    print(f"✅ ENVIRONMENT: {environment}")
    print(f"✅ BINANCE_TESTNET: {testnet}")
    print(f"✅ BINANCE_BASE_URL: {base_url}")
    
    print("=" * 50)
    
    # Determinar si está configurado para testnet o producción
    if testnet.lower() == 'false' and environment.lower() == 'production':
        print("🔥 CONFIGURACIÓN: PRODUCCIÓN (TRADING REAL)")
        print("⚠️ CUIDADO: Las operaciones serán con dinero real")
    else:
        print("🔧 CONFIGURACIÓN: TESTNET (SEGURO PARA PRUEBAS)")
    
    print()
    
    # Verificar que todas las credenciales estén presentes
    if api_key and secret_key and discord_webhook:
        print("🎯 CONFIGURACIÓN COMPLETA - LISTA PARA USAR")
        return True
    else:
        print("❌ CONFIGURACIÓN INCOMPLETA")
        return False

if __name__ == "__main__":
    verify_config() 