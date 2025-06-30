#!/usr/bin/env python3
"""
🧪 TEST: Lógica Inteligente de Mínimo de Posición
Verificar que el sistema ajusta automáticamente el mínimo según el balance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_risk_manager import AdvancedRiskManager
from config import trading_config
import asyncio

async def test_smart_min_position_logic():
    """🧪 Probar la lógica inteligente de ajuste de mínimo de posición"""
    
    print("🧪 TEST: LÓGICA INTELIGENTE DE MÍNIMO DE POSICIÓN")
    print("=" * 60)
    
    # Crear risk manager
    risk_manager = AdvancedRiskManager()
    
    # Casos de prueba con diferentes balances
    test_cases = [
        {"balance": 50.0, "description": "Balance muy pequeño"},
        {"balance": 80.81, "description": "Balance actual del usuario"},
        {"balance": 150.0, "description": "Balance pequeño"},
        {"balance": 300.0, "description": "Balance medio"},
        {"balance": 600.0, "description": "Balance grande"},
        {"balance": 1500.0, "description": "Balance muy grande"},
    ]
    
    print(f"📋 Configuración base:")
    print(f"   MIN_POSITION_VALUE_USDT: ${trading_config.MIN_POSITION_VALUE_USDT}")
    print(f"   MAX_POSITION_SIZE_PERCENT: {trading_config.MAX_POSITION_SIZE_PERCENT}%")
    print()
    
    for case in test_cases:
        balance = case["balance"]
        description = case["description"]
        
        print(f"🔍 CASO: {description} (${balance})")
        
        # Simular balance
        await risk_manager.initialize(balance)
        
        # Obtener mínimo efectivo
        effective_min = risk_manager._get_effective_min_position_value()
        config_min = trading_config.MIN_POSITION_VALUE_USDT
        
        # Calcular máximo por posición
        max_per_position = balance * (trading_config.MAX_POSITION_SIZE_PERCENT / 100)
        
        # Verificar si puede hacer trades
        can_trade = max_per_position >= effective_min
        
        print(f"   💰 Balance: ${balance}")
        print(f"   📊 Max por posición: ${max_per_position:.2f}")
        print(f"   🎯 Mínimo config: ${config_min:.2f}")
        print(f"   🧠 Mínimo efectivo: ${effective_min:.2f}")
        
        if effective_min < config_min:
            reduction_percent = ((config_min - effective_min) / config_min) * 100
            print(f"   ✅ Ajuste aplicado: -{reduction_percent:.1f}%")
        else:
            print(f"   ⚪ Sin ajuste necesario")
            
        if can_trade:
            print(f"   🟢 RESULTADO: Puede ejecutar trades")
        else:
            print(f"   🔴 RESULTADO: No puede ejecutar trades")
            
        print()
    
    # Test específico para el caso del usuario
    print("🎯 CASO ESPECÍFICO DEL USUARIO:")
    print("=" * 40)
    
    user_balance = 80.81
    await risk_manager.initialize(user_balance)
    
    # Simular cálculo de posición para BTCUSDT
    btc_price = 107236.35
    confidence = 0.743
    
    print(f"Balance: ${user_balance}")
    print(f"Precio BTC: ${btc_price:,.2f}")
    print(f"Confianza: {confidence:.1%}")
    print()
    
    # Calcular cantidad usando el método inteligente
    calculated_amount = risk_manager.calculate_position_size("BTCUSDT", btc_price, confidence)
    
    if calculated_amount > 0:
        position_value = calculated_amount * btc_price
        effective_min = risk_manager._get_effective_min_position_value()
        
        print(f"✅ ÉXITO:")
        print(f"   Cantidad calculada: {calculated_amount:.8f} BTC")
        print(f"   Valor de posición: ${position_value:.2f}")
        print(f"   Mínimo efectivo: ${effective_min:.2f}")
        print(f"   ✅ Cumple requisitos: ${position_value:.2f} >= ${effective_min:.2f}")
    else:
        print(f"❌ FALLO: No se pudo calcular cantidad válida")

if __name__ == "__main__":
    asyncio.run(test_smart_min_position_logic()) 