#!/usr/bin/env python3
"""
🔧 FIX BINANCE TIMESTAMP ERROR
Soluciona el error: "Timestamp for this request is outside of the recvWindow"
"""

import os
import time
import hmac
import hashlib
import asyncio
import aiohttp
import platform
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class BinanceTimestampFixer:
    """🔧 Solucionador de problemas de timestamp de Binance"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY') 
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.time_offset = 0
        
    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma para API de Binance"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def sync_time_with_binance(self):
        """🕐 Sincronizar tiempo local con servidor de Binance"""
        try:
            print("🕐 Sincronizando tiempo con servidor de Binance...")
            
            # Obtener tiempo del servidor de Binance
            async with aiohttp.ClientSession() as session:
                start_time = int(time.time() * 1000)
                
                async with session.get(f"{self.base_url}/api/v3/time") as response:
                    if response.status == 200:
                        data = await response.json()
                        server_time = data['serverTime']
                        
                        end_time = int(time.time() * 1000)
                        local_time = (start_time + end_time) // 2
                        
                        # Calcular offset
                        self.time_offset = server_time - local_time
                        
                        print(f"✅ Sincronización completada:")
                        print(f"   🖥️  Tiempo local: {datetime.fromtimestamp(local_time/1000)}")
                        print(f"   🌐 Tiempo servidor: {datetime.fromtimestamp(server_time/1000)}")
                        print(f"   ⚡ Offset: {self.time_offset}ms")
                        
                        return True
                    else:
                        print(f"❌ Error obteniendo tiempo del servidor: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Error en sincronización: {e}")
            return False
    
    def get_synchronized_timestamp(self):
        """⏰ Obtener timestamp sincronizado"""
        return int(time.time() * 1000) + self.time_offset
    
    async def test_binance_connection(self):
        """🧪 Probar conexión con Binance usando parámetros optimizados"""
        print("\n🧪 Probando conexión con Binance...")
        
        try:
            # Usar timestamp sincronizado
            timestamp = self.get_synchronized_timestamp()
            
            # Parámetros optimizados con recvWindow extendido
            params = f"timestamp={timestamp}&recvWindow=60000"  # 60 segundos
            signature = self._generate_signature(params)
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/account"
            full_params = f"{params}&signature={signature}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}?{full_params}", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Buscar balance USDT
                        usdt_balance = 0.0
                        for balance in data.get('balances', []):
                            if balance['asset'] == 'USDT':
                                usdt_balance = float(balance['free'])
                                break
                        
                        print("✅ CONEXIÓN EXITOSA!")
                        print(f"   💰 Balance USDT: ${usdt_balance:.2f}")
                        print(f"   🔑 API Key: ...{self.api_key[-10:]}")
                        print(f"   ⏰ Timestamp usado: {timestamp}")
                        print(f"   🌐 Servidor: {self.base_url}")
                        
                        return True
                    
                    elif response.status == 400:
                        error_data = await response.json()
                        error_code = error_data.get('code', 'Unknown')
                        error_msg = error_data.get('msg', 'Unknown error')
                        
                        print(f"❌ ERROR BINANCE:")
                        print(f"   📟 Código: {error_code}")
                        print(f"   💬 Mensaje: {error_msg}")
                        print(f"   ⏰ Timestamp: {timestamp}")
                        print(f"   🔄 Offset: {self.time_offset}ms")
                        
                        # Sugerencias específicas por código de error
                        if error_code == -1021:
                            print("\n🔧 SOLUCIONES SUGERIDAS:")
                            print("   1. Sincronizar reloj del sistema")
                            print("   2. Aumentar recvWindow")
                            print("   3. Verificar latencia de red")
                            print("   4. Actualizar offset de tiempo")
                            
                        return False
                    
                    else:
                        print(f"❌ Error HTTP: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"❌ Error en prueba: {e}")
            return False
    
    def sync_system_clock(self):
        """🕐 Sincronizar reloj del sistema (Windows)"""
        try:
            if platform.system() == "Windows":
                print("🕐 Sincronizando reloj del sistema Windows...")
                
                # Ejecutar w32tm para sincronizar
                result = subprocess.run(
                    ["w32tm", "/resync"],
                    capture_output=True,
                    text=True,
                    shell=True
                )
                
                if result.returncode == 0:
                    print("✅ Reloj del sistema sincronizado")
                    return True
                else:
                    print(f"⚠️ Warning: {result.stderr}")
                    return False
            else:
                print("ℹ️ Sincronización automática solo disponible en Windows")
                return True
                
        except Exception as e:
            print(f"⚠️ Error sincronizando reloj: {e}")
            return False
    
    async def update_env_with_recv_window(self):
        """📝 Actualizar .env con recvWindow optimizado"""
        try:
            print("📝 Actualizando configuración .env...")
            
            # Leer archivo .env actual
            with open('.env', 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Buscar si ya existe RECV_WINDOW
            recv_window_exists = False
            for i, line in enumerate(lines):
                if line.startswith('RECV_WINDOW='):
                    lines[i] = 'RECV_WINDOW=60000\n'
                    recv_window_exists = True
                    break
            
            # Si no existe, agregarlo
            if not recv_window_exists:
                lines.append('\n# Binance API Configuration\n')
                lines.append('RECV_WINDOW=60000\n')
                lines.append('TIMESTAMP_OFFSET=0\n')
            
            # Escribir archivo actualizado
            with open('.env', 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print("✅ Configuración .env actualizada")
            print("   🔧 RECV_WINDOW=60000 (60 segundos)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error actualizando .env: {e}")
            return False

async def main():
    """🚀 Función principal para solucionar problemas de timestamp"""
    print("🔧 BINANCE TIMESTAMP FIXER")
    print("=" * 50)
    
    fixer = BinanceTimestampFixer()
    
    # Verificar que tenemos credenciales
    if not fixer.api_key or not fixer.secret_key:
        print("❌ ERROR: No se encontraron credenciales de Binance")
        print("   Verifica que BINANCE_API_KEY y BINANCE_SECRET_KEY estén en .env")
        return
    
    # Paso 1: Sincronizar reloj del sistema
    print("📝 Paso 1: Sincronizando reloj del sistema...")
    fixer.sync_system_clock()
    
    # Paso 2: Sincronizar con servidor Binance
    print("\n📝 Paso 2: Sincronizando con servidor Binance...")
    if await fixer.sync_time_with_binance():
        print("✅ Sincronización exitosa")
    else:
        print("⚠️ No se pudo sincronizar, usando offset 0")
    
    # Paso 3: Actualizar configuración .env
    print("\n📝 Paso 3: Actualizando configuración...")
    await fixer.update_env_with_recv_window()
    
    # Paso 4: Probar conexión
    print("\n📝 Paso 4: Probando conexión...")
    success = await fixer.test_binance_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ¡PROBLEMA SOLUCIONADO!")
        print("✅ Tu bot ahora puede conectarse a Binance correctamente")
    else:
        print("❌ Problema persistente")
        print("💡 Soluciones adicionales:")
        print("   1. Verifica tus credenciales API")
        print("   2. Revisa restricciones de IP en Binance")
        print("   3. Contacta soporte si persiste")

if __name__ == "__main__":
    asyncio.run(main()) 