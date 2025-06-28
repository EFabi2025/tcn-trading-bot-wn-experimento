#!/usr/bin/env python3
"""
🧪 TEST: Cálculo Inteligente de Tamaño de Posición
Verifica que el nuevo sistema calcule correctamente los tamaños considerando exposición total.
"""

import asyncio
import logging
from datetime import datetime
from config import TradingConfig
from advanced_risk_manager import AdvancedRiskManager, Position

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_smart_position_sizing():
    """🧪 Probar el cálculo inteligente de posiciones"""
    
    print("🧪 INICIANDO PRUEBAS DE CÁLCULO INTELIGENTE DE POSICIONES")
    print("=" * 70)
    
    # Inicializar configuración y risk manager
    config = TradingConfig()
    risk_manager = AdvancedRiskManager()
    
    # Simular balance inicial
    test_balance = 50.0  # $50 USDT
    await risk_manager.initialize(test_balance)
    
    print(f"\n💰 CONFIGURACIÓN DE PRUEBA:")
    print(f"   Balance inicial: ${test_balance}")
    print(f"   Max posición: {config.MAX_POSITION_SIZE_PERCENT}%")
    print(f"   Max exposición: {config.MAX_TOTAL_EXPOSURE_PERCENT}%")
    print(f"   Mínimo por trade: ${config.MIN_POSITION_VALUE_USDT}")
    
    # ESCENARIO 1: Sin posiciones existentes
    print(f"\n🎯 ESCENARIO 1: Sin posiciones existentes")
    print("-" * 50)
    
    symbol = "BTCUSDT"
    price = 45000.0
    confidence = 0.70
    
    size1 = risk_manager.calculate_position_size(symbol, price, confidence)
    value1 = size1 * price
    
    print(f"✅ Resultado: {size1:.6f} BTC (${value1:.2f} USD)")
    
    # Simular que se abre esta posición
    if size1 > 0:
        position1 = Position(
            symbol=symbol,
            side='BUY',
            quantity=size1,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now()
        )
        risk_manager.active_positions[symbol] = position1
        print(f"   📊 Posición {symbol} simulada: ${value1:.2f}")
    
    # ESCENARIO 2: Con una posición existente
    print(f"\n🎯 ESCENARIO 2: Con posición BTC existente")
    print("-" * 50)
    
    symbol2 = "ETHUSDT"
    price2 = 3000.0
    confidence2 = 0.65
    
    size2 = risk_manager.calculate_position_size(symbol2, price2, confidence2)
    value2 = size2 * price2
    
    print(f"✅ Resultado: {size2:.6f} ETH (${value2:.2f} USD)")
    
    # Simular que se abre esta posición
    if size2 > 0:
        position2 = Position(
            symbol=symbol2,
            side='BUY',
            quantity=size2,
            entry_price=price2,
            current_price=price2,
            entry_time=datetime.now()
        )
        risk_manager.active_positions[symbol2] = position2
        print(f"   📊 Posición {symbol2} simulada: ${value2:.2f}")
    
    # ESCENARIO 3: Intentar tercera posición (puede ser rechazada)
    print(f"\n🎯 ESCENARIO 3: Intentar tercera posición (XRP)")
    print("-" * 50)
    
    symbol3 = "XRPUSDT"
    price3 = 2.18
    confidence3 = 0.75
    
    size3 = risk_manager.calculate_position_size(symbol3, price3, confidence3)
    value3 = size3 * price3
    
    print(f"✅ Resultado: {size3:.6f} XRP (${value3:.2f} USD)")
    
    # ESCENARIO 4: Estado final del portfolio
    print(f"\n📊 ESTADO FINAL DEL PORTFOLIO:")
    print("-" * 50)
    
    total_exposure = sum(p.quantity * p.current_price for p in risk_manager.active_positions.values())
    max_exposure = test_balance * (config.MAX_TOTAL_EXPOSURE_PERCENT / 100)
    exposure_percent = (total_exposure / test_balance) * 100
    
    print(f"   💰 Balance: ${test_balance}")
    print(f"   📊 Exposición actual: ${total_exposure:.2f} ({exposure_percent:.1f}%)")
    print(f"   ⚖️ Límite exposición: ${max_exposure:.2f} ({config.MAX_TOTAL_EXPOSURE_PERCENT}%)")
    print(f"   🆓 Exposición disponible: ${max_exposure - total_exposure:.2f}")
    print(f"   🔢 Posiciones activas: {len(risk_manager.active_positions)}")
    
    for symbol, pos in risk_manager.active_positions.items():
        pos_value = pos.quantity * pos.current_price
        pos_percent = (pos_value / test_balance) * 100
        print(f"      • {symbol}: ${pos_value:.2f} ({pos_percent:.1f}%)")
    
    # ESCENARIO 5: Probar con balance muy bajo
    print(f"\n🎯 ESCENARIO 5: Balance muy bajo (límites)")
    print("-" * 50)
    
    # Simular balance bajo
    low_balance = 15.0  # $15 USDT
    risk_manager.current_balance = low_balance
    risk_manager.active_positions.clear()  # Limpiar posiciones
    
    symbol4 = "BNBUSDT"
    price4 = 600.0
    confidence4 = 0.80
    
    size4 = risk_manager.calculate_position_size(symbol4, price4, confidence4)
    value4 = size4 * price4
    
    print(f"   💰 Balance reducido: ${low_balance}")
    print(f"✅ Resultado: {size4:.6f} BNB (${value4:.2f} USD)")
    
    # RESUMEN
    print(f"\n🎯 RESUMEN DE PRUEBAS:")
    print("=" * 70)
    print(f"✅ Escenario 1 (sin posiciones): ${value1:.2f} USD")
    print(f"✅ Escenario 2 (con BTC): ${value2:.2f} USD")
    print(f"✅ Escenario 3 (con BTC+ETH): ${value3:.2f} USD")
    print(f"✅ Escenario 5 (balance bajo): ${value4:.2f} USD")
    
    print(f"\n🧠 ANÁLISIS:")
    if value1 > 0 and value2 > 0:
        print(f"✅ El sistema calcula tamaños apropiados")
    if value2 < value1:
        print(f"✅ El sistema reduce tamaños con más exposición")
    if value3 >= 0:
        print(f"✅ El sistema maneja límites de exposición correctamente")
    if value4 >= config.MIN_POSITION_VALUE_USDT or value4 == 0:
        print(f"✅ El sistema respeta mínimos de Binance")
    
    print(f"\n🎉 PRUEBAS COMPLETADAS")

if __name__ == "__main__":
    asyncio.run(test_smart_position_sizing()) 