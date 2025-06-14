#!/usr/bin/env python3
"""Check Binance credentials format"""

import os
from dotenv import load_dotenv

load_dotenv()

def check_credentials():
    print("🔍 VERIFICANDO CREDENCIALES DE BINANCE")
    print("=" * 50)
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    print(f"API Key presente: {'✅' if api_key else '❌'}")
    if api_key:
        print(f"API Key length: {len(api_key)}")
        print(f"First 20 chars: {api_key[:20]}")
        print(f"Last 10 chars: ...{api_key[-10:]}")
        
        # Verificar si contiene caracteres no válidos
        if '\n' in api_key or '\r' in api_key:
            print("❌ PROBLEMA: API Key contiene saltos de línea")
        if len(api_key) != 64:
            print(f"⚠️ WARNING: API Key debería tener 64 caracteres, tiene {len(api_key)}")
    
    print()
    print(f"Secret Key presente: {'✅' if secret_key else '❌'}")
    if secret_key:
        print(f"Secret Key length: {len(secret_key)}")
        print(f"First 20 chars: {secret_key[:20]}")
        print(f"Last 10 chars: ...{secret_key[-10:]}")
        
        # Verificar si contiene caracteres no válidos
        if '\n' in secret_key or '\r' in secret_key:
            print("❌ PROBLEMA: Secret Key contiene saltos de línea")
        if len(secret_key) != 64:
            print(f"⚠️ WARNING: Secret Key debería tener 64 caracteres, tiene {len(secret_key)}")
    
    print()
    print("🔗 URLs configuradas:")
    print(f"BINANCE_BASE_URL: {os.getenv('BINANCE_BASE_URL')}")
    print(f"ENVIRONMENT: {os.getenv('ENVIRONMENT')}")
    
    return api_key and secret_key

if __name__ == "__main__":
    if check_credentials():
        print("\n✅ Credenciales básicas presentes")
    else:
        print("\n❌ Faltan credenciales") 