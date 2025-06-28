#!/usr/bin/env python3
"""
🧪 TEST: Escenario Real - Problema de Exposición Solucionado
Simula exactamente el escenario reportado por el usuario para verificar la solución.
"""

import asyncio
import logging
from datetime import datetime
from config import TradingConfig
from advanced_risk_manager import AdvancedRiskManager, Position

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_real_scenario():
    """🧪 Simular el escenario real reportado"""
    
    print("🧪 SIMULANDO ESCENARIO REAL - PROBLEMA DE EXPOSICIÓN")
    print("=" * 70)
    
    # Inicializar configuración y risk manager
    config = TradingConfig()
    risk_manager = AdvancedRiskManager()
    
    # CONFIGURACIÓN REAL DEL ESCENARIO
    # Balance estimado basado en los cálculos del log real
    # Exposición actual: $13.97, Límite: $27.77 → Balance ≈ $37
    real_balance = 37.0  # $37 USDT (estimado)
    await risk_manager.initialize(real_balance)
    
    print(f"\n💰 CONFIGURACIÓN REAL:")
    print(f"   Balance: ${real_balance}")
    print(f"   Max posición: {config.MAX_POSITION_SIZE_PERCENT}%")
    print(f"   Max exposición: {config.MAX_TOTAL_EXPOSURE_PERCENT}%")
    print(f"   Límite exposición: ${real_balance * config.MAX_TOTAL_EXPOSURE_PERCENT/100:.2f}")
    
    # SIMULAR POSICIÓN EXISTENTE (BTC)
    print(f"\n📊 SIMULANDO POSICIÓN EXISTENTE:")
    btc_position = Position(
        symbol="BTCUSDT",
        side='BUY',
        quantity=0.00013,  # Aproximado para $13.97
        entry_price=107305.69,
        current_price=107305.69,
        entry_time=datetime.now()
    )
    risk_manager.active_positions["BTCUSDT"] = btc_position
    
    btc_value = btc_position.quantity * btc_position.current_price
    print(f"   BTC existente: {btc_position.quantity:.6f} BTC @ ${btc_position.current_price:.2f}")
    print(f"   Valor: ${btc_value:.2f}")
    
    # ESCENARIO 1: MÉTODO ANTERIOR (simular el problema)
    print(f"\n❌ MÉTODO ANTERIOR (PROBLEMÁTICO):")
    print("-" * 50)
    
    # Simular cálculo anterior (sin considerar exposición)
    old_position_percent = config.MAX_POSITION_SIZE_PERCENT  # 22%
    old_position_value = real_balance * (old_position_percent / 100)
    old_total_exposure = btc_value + old_position_value
    old_limit = real_balance * (config.MAX_TOTAL_EXPOSURE_PERCENT / 100)
    
    print(f"   🎯 Tamaño calculado (método anterior): ${old_position_value:.2f}")
    print(f"   📊 Exposición total resultante: ${old_total_exposure:.2f}")
    print(f"   ⚖️ Límite de exposición: ${old_limit:.2f}")
    print(f"   ❌ Resultado: {'RECHAZADO' if old_total_exposure > old_limit else 'APROBADO'}")
    
    if old_total_exposure > old_limit:
        print(f"   💥 PROBLEMA: Exposición excede límite por ${old_total_exposure - old_limit:.2f}")
    
    # ESCENARIO 2: NUEVO MÉTODO INTELIGENTE
    print(f"\n✅ NUEVO MÉTODO INTELIGENTE:")
    print("-" * 50)
    
    # Probar señal BUY para XRPUSDT (como en el escenario real)
    symbol = "XRPUSDT"
    price = 2.18
    confidence = 0.70
    
    print(f"   🚀 Probando señal BUY: {symbol}")
    print(f"   💱 Precio: ${price}")
    print(f"   🎯 Confianza: {confidence:.1%}")
    
    # Verificar límites básicos
    can_trade, reason = await risk_manager.check_risk_limits_before_trade(symbol, 'BUY', confidence)
    print(f"   🛡️ Límites básicos: {'✅ APROBADO' if can_trade else '❌ RECHAZADO'}")
    if not can_trade:
        print(f"      Razón: {reason}")
    
    # Calcular tamaño inteligente
    if can_trade:
        smart_size = risk_manager.calculate_position_size(symbol, price, confidence)
        smart_value = smart_size * price
        smart_total_exposure = btc_value + smart_value
        
        print(f"   💡 Tamaño inteligente calculado: {smart_size:.6f} XRP")
        print(f"   💵 Valor: ${smart_value:.2f}")
        print(f"   📊 Exposición total resultante: ${smart_total_exposure:.2f}")
        print(f"   ✅ Resultado: {'APROBADO' if smart_value > 0 else 'RECHAZADO'}")
        
        # Verificar que cumple con el mínimo
        if smart_value >= config.MIN_POSITION_VALUE_USDT:
            print(f"   ✅ Cumple mínimo de Binance (${config.MIN_POSITION_VALUE_USDT})")
        else:
            print(f"   ⚠️ No cumple mínimo de Binance (${config.MIN_POSITION_VALUE_USDT})")
    
    # ESCENARIO 3: MÚLTIPLES POSICIONES
    print(f"\n🎯 ESCENARIO: MÚLTIPLES POSICIONES ADICIONALES")
    print("-" * 50)
    
    # Simular que se abrió XRP y probar otra posición
    if can_trade and smart_size > 0:
        xrp_position = Position(
            symbol="XRPUSDT",
            side='BUY',
            quantity=smart_size,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now()
        )
        risk_manager.active_positions["XRPUSDT"] = xrp_position
        print(f"   📊 XRP agregado: {smart_size:.6f} XRP (${smart_value:.2f})")
    
    # Probar tercera posición
    symbol2 = "ETHUSDT"
    price2 = 3000.0
    confidence2 = 0.65
    
    print(f"   🚀 Probando tercera posición: {symbol2}")
    
    can_trade2, reason2 = await risk_manager.check_risk_limits_before_trade(symbol2, 'BUY', confidence2)
    if can_trade2:
        size2 = risk_manager.calculate_position_size(symbol2, price2, confidence2)
        value2 = size2 * price2
        print(f"   💡 Tercera posición: {size2:.6f} ETH (${value2:.2f})")
    else:
        print(f"   ❌ Tercera posición rechazada: {reason2}")
    
    # RESUMEN FINAL
    print(f"\n📊 ESTADO FINAL DEL PORTFOLIO:")
    print("=" * 70)
    
    total_exposure = sum(p.quantity * p.current_price for p in risk_manager.active_positions.values())
    max_exposure = real_balance * (config.MAX_TOTAL_EXPOSURE_PERCENT / 100)
    exposure_percent = (total_exposure / real_balance) * 100
    available_exposure = max_exposure - total_exposure
    
    print(f"💰 Balance: ${real_balance}")
    print(f"📊 Exposición total: ${total_exposure:.2f} ({exposure_percent:.1f}%)")
    print(f"⚖️ Límite exposición: ${max_exposure:.2f} ({config.MAX_TOTAL_EXPOSURE_PERCENT}%)")
    print(f"🆓 Exposición disponible: ${available_exposure:.2f}")
    print(f"🔢 Posiciones activas: {len(risk_manager.active_positions)}")
    
    for symbol, pos in risk_manager.active_positions.items():
        pos_value = pos.quantity * pos.current_price
        pos_percent = (pos_value / real_balance) * 100
        print(f"   • {symbol}: ${pos_value:.2f} ({pos_percent:.1f}%)")
    
    # ANÁLISIS
    print(f"\n🧠 ANÁLISIS DE LA SOLUCIÓN:")
    print("=" * 70)
    
    if old_total_exposure > old_limit and total_exposure <= max_exposure:
        print("✅ PROBLEMA RESUELTO: El método anterior rechazaba, el nuevo permite trades apropiados")
    
    if available_exposure > 0:
        print(f"✅ MARGEN DISPONIBLE: Aún hay ${available_exposure:.2f} para futuras posiciones")
    
    if len(risk_manager.active_positions) > 1:
        print("✅ DIVERSIFICACIÓN: Sistema permite múltiples posiciones balanceadas")
    
    print(f"\n🎉 PRUEBA COMPLETADA - SISTEMA OPTIMIZADO")

if __name__ == "__main__":
    asyncio.run(test_real_scenario()) 