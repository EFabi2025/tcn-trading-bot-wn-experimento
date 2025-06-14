#!/usr/bin/env python3
"""
🔑 ACTUALIZAR CREDENCIALES DE BINANCE
Script interactivo para actualizar las credenciales de forma segura
"""

import os
import re

def validate_binance_key(key, key_type):
    """Validar formato de clave de Binance"""
    if not key:
        return False, f"{key_type} está vacía"
    
    # Remover espacios y saltos de línea
    key = key.strip()
    
    # Verificar longitud (normalmente 64 caracteres)
    if len(key) != 64:
        return False, f"{key_type} debe tener 64 caracteres (tiene {len(key)})"
    
    # Verificar que solo contenga caracteres alfanuméricos
    if not re.match(r'^[A-Za-z0-9]+$', key):
        return False, f"{key_type} contiene caracteres no válidos"
    
    return True, "Válida"

def update_env_file(api_key, secret_key):
    """Actualizar archivo .env con nuevas credenciales"""
    try:
        # Leer archivo actual
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Actualizar o agregar credenciales
        api_updated = False
        secret_updated = False
        
        for i, line in enumerate(lines):
            if line.startswith('BINANCE_API_KEY='):
                lines[i] = f'BINANCE_API_KEY={api_key}\n'
                api_updated = True
            elif line.startswith('BINANCE_SECRET_KEY='):
                lines[i] = f'BINANCE_SECRET_KEY={secret_key}\n'
                secret_updated = True
        
        # Si no existían, agregarlas
        if not api_updated:
            lines.append(f'BINANCE_API_KEY={api_key}\n')
        if not secret_updated:
            lines.append(f'BINANCE_SECRET_KEY={secret_key}\n')
        
        # Escribir archivo actualizado
        with open('.env', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
    
    except Exception as e:
        print(f"❌ Error actualizando .env: {e}")
        return False

def main():
    print("🔑 ACTUALIZAR CREDENCIALES DE BINANCE")
    print("=" * 50)
    print("Por favor, ingresa tus credenciales de Binance:")
    print("(Las credenciales deben tener exactamente 64 caracteres)")
    print()
    
    # Solicitar API Key
    while True:
        api_key = input("🔑 BINANCE_API_KEY: ").strip()
        valid, message = validate_binance_key(api_key, "API Key")
        
        if valid:
            print("✅ API Key válida")
            break
        else:
            print(f"❌ {message}")
            print("   Intenta nuevamente...")
            print()
    
    # Solicitar Secret Key
    while True:
        secret_key = input("🔐 BINANCE_SECRET_KEY: ").strip()
        valid, message = validate_binance_key(secret_key, "Secret Key")
        
        if valid:
            print("✅ Secret Key válida")
            break
        else:
            print(f"❌ {message}")
            print("   Intenta nuevamente...")
            print()
    
    print()
    print("📝 Actualizando archivo .env...")
    
    if update_env_file(api_key, secret_key):
        print("✅ Credenciales actualizadas correctamente!")
        print()
        print("🧪 Ahora puedes probar la conexión ejecutando:")
        print("   python quick_binance_fix.py")
    else:
        print("❌ Error actualizando credenciales")
    
    print()
    print("⚠️ RECORDATORIO DE SEGURIDAD:")
    print("  - Estas credenciales dan acceso a tu cuenta de Binance")
    print("  - Nunca las compartas con nadie")
    print("  - Asegúrate de tener permisos limitados en Binance")

if __name__ == "__main__":
    main() 