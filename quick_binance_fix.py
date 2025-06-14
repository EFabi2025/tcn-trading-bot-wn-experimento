#!/usr/bin/env python3
"""Quick Binance timestamp fix"""

import os
import time
import hmac
import hashlib
import asyncio
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class BinanceTimestampFixer:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY') 
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.time_offset = 0
        
    def _generate_signature(self, params):
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def sync_time_with_binance(self):
        try:
            print('🕐 Sincronizando tiempo con servidor de Binance...')
            async with aiohttp.ClientSession() as session:
                start_time = int(time.time() * 1000)
                async with session.get(f'{self.base_url}/api/v3/time') as response:
                    if response.status == 200:
                        data = await response.json()
                        server_time = data['serverTime']
                        end_time = int(time.time() * 1000)
                        local_time = (start_time + end_time) // 2
                        self.time_offset = server_time - local_time
                        print(f'✅ Offset calculado: {self.time_offset}ms')
                        return True
            return False
        except Exception as e:
            print(f'❌ Error: {e}')
            return False
    
    def get_synchronized_timestamp(self):
        return int(time.time() * 1000) + self.time_offset
    
    async def test_connection(self):
        try:
            timestamp = self.get_synchronized_timestamp() 
            params = f'timestamp={timestamp}&recvWindow=60000'
            signature = self._generate_signature(params)
            headers = {'X-MBX-APIKEY': self.api_key}
            url = f'{self.base_url}/api/v3/account?{params}&signature={signature}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        usdt_balance = 0.0
                        for balance in data.get('balances', []):
                            if balance['asset'] == 'USDT':
                                usdt_balance = float(balance['free'])
                                break
                        print(f'✅ CONEXIÓN EXITOSA! Balance: ${usdt_balance:.2f}')
                        return True
                    else:
                        error_data = await response.json()
                        print(f'❌ Error {response.status}: {error_data}')
                        return False
        except Exception as e:
            print(f'❌ Error: {e}')
            return False

async def main():
    print('🔧 BINANCE TIMESTAMP FIXER')
    print('=' * 40)
    
    fixer = BinanceTimestampFixer()
    
    if not fixer.api_key:
        print('❌ No se encontraron credenciales')
        return
    
    print('Paso 1: Sincronizando con Binance...')
    await fixer.sync_time_with_binance()
    
    print('Paso 2: Probando conexión...')
    success = await fixer.test_connection()
    
    if success:
        print('🎉 ¡PROBLEMA SOLUCIONADO!')
    else:
        print('❌ Problema persistente')

if __name__ == "__main__":
    asyncio.run(main()) 