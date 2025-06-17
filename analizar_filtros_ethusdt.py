#!/usr/bin/env python3
"""
🔍 ANÁLISIS DE FILTROS ETHUSDT
Script para analizar los filtros específicos de ETHUSDT en Binance
"""

import asyncio
import aiohttp
import math
import os
from dotenv import load_dotenv

load_dotenv()

async def analyze_ethusdt_filters():
    """🔍 Analizar filtros específicos de ETHUSDT"""
    
    base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
    symbol = 'ETHUSDT'
    
    print(f"🔍 ANALIZANDO FILTROS PARA {symbol}")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Obtener información del exchange
            url = f"{base_url}/api/v3/exchangeInfo"
            params = {'symbol': symbol}
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    print(f"❌ Error: {response.status}")
                    return
                
                data = await response.json()
                symbol_info = data['symbols'][0]
                
                print(f"📊 Información de {symbol}:")
                print(f"   Status: {symbol_info['status']}")
                print(f"   Base Asset: {symbol_info['baseAsset']}")
                print(f"   Quote Asset: {symbol_info['quoteAsset']}")
                print()
                
                print("🔧 FILTROS ACTIVOS:")
                print("-" * 30)
                
                for filter_info in symbol_info['filters']:
                    filter_type = filter_info['filterType']
                    
                    if filter_type == 'LOT_SIZE':
                        min_qty = float(filter_info['minQty'])
                        max_qty = float(filter_info['maxQty'])
                        step_size = float(filter_info['stepSize'])
                        
                        print(f"📏 LOT_SIZE:")
                        print(f"   Min Quantity: {min_qty:.8f} ETH")
                        print(f"   Max Quantity: {max_qty:.8f} ETH")
                        print(f"   Step Size: {step_size:.8f} ETH")
                        print()
                        
                        # Probar ajuste con cantidad original
                        original_qty = 0.018618
                        print(f"🧪 PRUEBA DE AJUSTE:")
                        print(f"   Cantidad original: {original_qty:.8f} ETH")
                        
                        # Método actual (problemático)
                        adjusted_ceil = math.ceil(original_qty / step_size) * step_size
                        if adjusted_ceil < min_qty:
                            adjusted_ceil = min_qty
                            
                        print(f"   Con CEIL: {adjusted_ceil:.8f} ETH")
                        
                        # Método correcto (redondear al múltiplo más cercano)
                        adjusted_round = round(original_qty / step_size) * step_size
                        if adjusted_round < min_qty:
                            adjusted_round = min_qty
                            
                        print(f"   Con ROUND: {adjusted_round:.8f} ETH")
                        print()
                        
                    elif filter_type == 'MIN_NOTIONAL':
                        min_notional = float(filter_info['minNotional'])
                        print(f"💵 MIN_NOTIONAL:")
                        print(f"   Valor mínimo: ${min_notional:.2f} USDT")
                        print()
                        
                        # Verificar si nuestras cantidades cumplen
                        current_price = 2533.80  # Precio aproximado actual
                        
                        value_ceil = adjusted_ceil * current_price
                        value_round = adjusted_round * current_price
                        
                        print(f"🔍 VERIFICACIÓN NOTIONAL:")
                        print(f"   Precio actual: ${current_price:.2f}")
                        print(f"   Valor con CEIL: ${value_ceil:.2f} {'✅' if value_ceil >= min_notional else '❌'}")
                        print(f"   Valor con ROUND: ${value_round:.2f} {'✅' if value_round >= min_notional else '❌'}")
                        print()
                        
                    elif filter_type == 'MARKET_LOT_SIZE':
                        min_qty = float(filter_info['minQty'])
                        max_qty = float(filter_info['maxQty'])
                        step_size = float(filter_info['stepSize'])
                        
                        print(f"🏪 MARKET_LOT_SIZE:")
                        print(f"   Min Quantity: {min_qty:.8f} ETH")
                        print(f"   Max Quantity: {max_qty:.8f} ETH")
                        print(f"   Step Size: {step_size:.8f} ETH")
                        print()
                        
                    elif filter_type == 'PERCENT_PRICE':
                        print(f"📈 PERCENT_PRICE:")
                        print(f"   Multiplier Up: {filter_info['multiplierUp']}")
                        print(f"   Multiplier Down: {filter_info['multiplierDown']}")
                        print()
                        
                    else:
                        print(f"🔧 {filter_type}: {filter_info}")
                        print()
                
    except Exception as e:
        print(f"❌ Error analizando filtros: {e}")

def test_quantity_adjustment():
    """🧪 Probar diferentes métodos de ajuste de cantidad"""
    
    print("\n" + "="*50)
    print("🧪 PRUEBA DE MÉTODOS DE AJUSTE")
    print("="*50)
    
    # Parámetros de ejemplo para ETH
    original_qty = 0.018618
    step_size = 0.00001  # Ejemplo típico para ETH
    min_qty = 0.0001     # Ejemplo típico para ETH
    price = 2533.80
    min_notional = 5.0   # Ejemplo típico
    
    print(f"📊 Parámetros de prueba:")
    print(f"   Cantidad original: {original_qty:.8f} ETH")
    print(f"   Step size: {step_size:.8f} ETH")
    print(f"   Cantidad mínima: {min_qty:.8f} ETH")
    print(f"   Precio: ${price:.2f}")
    print(f"   Notional mínimo: ${min_notional:.2f}")
    print()
    
    methods = {
        'FLOOR': lambda qty, step: math.floor(qty / step) * step,
        'CEIL': lambda qty, step: math.ceil(qty / step) * step,
        'ROUND': lambda qty, step: round(qty / step) * step
    }
    
    for method_name, method_func in methods.items():
        adjusted = method_func(original_qty, step_size)
        
        if adjusted < min_qty:
            adjusted = min_qty
            note = "(ajustado al mínimo)"
        else:
            note = ""
            
        value = adjusted * price
        notional_ok = value >= min_notional
        
        print(f"🔧 {method_name}:")
        print(f"   Cantidad: {adjusted:.8f} ETH {note}")
        print(f"   Valor: ${value:.2f} {'✅' if notional_ok else '❌'}")
        print()

async def main():
    """🎯 Función principal"""
    await analyze_ethusdt_filters()
    test_quantity_adjustment()

if __name__ == "__main__":
    asyncio.run(main()) 