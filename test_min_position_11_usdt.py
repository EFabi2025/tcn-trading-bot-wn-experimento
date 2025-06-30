#!/usr/bin/env python3
"""
🧪 TEST: Verificación del Mínimo de $11 USDT
Prueba exhaustiva para asegurar que el sistema respeta el mínimo de Binance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
from config import trading_config
from advanced_risk_manager import AdvancedRiskManager

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def test_min_position_11_usdt():
    """🧪 Probar que el sistema respeta el mínimo de $11 USDT en todos los escenarios"""
    
    print("🧪 TEST: VERIFICACIÓN DEL MÍNIMO DE $11 USDT")
    print("=" * 60)
    
    # Verificar configuración
    print(f"📋 CONFIGURACIÓN ACTUAL:")
    print(f"   MIN_POSITION_VALUE_USDT: ${trading_config.MIN_POSITION_VALUE_USDT}")
    print(f"   MAX_POSITION_SIZE_PERCENT: {trading_config.MAX_POSITION_SIZE_PERCENT}%")
    print(f"   MAX_TOTAL_EXPOSURE_PERCENT: {trading_config.MAX_TOTAL_EXPOSURE_PERCENT}%")
    print()
    
    # Casos de prueba con diferentes balances
    test_cases = [
        {
            "balance": 30.0,
            "description": "Balance muy pequeño (< $50)",
            "expected_behavior": "Permitir trading con advertencias"
        },
        {
            "balance": 50.0,
            "description": "Balance mínimo viable",
            "expected_behavior": "Trading normal con mínimo $11"
        },
        {
            "balance": 80.81,
            "description": "Balance actual del usuario",
            "expected_behavior": "Trading normal con mínimo $11"
        },
        {
            "balance": 150.0,
            "description": "Balance pequeño",
            "expected_behavior": "Trading normal con mínimo $11"
        },
        {
            "balance": 300.0,
            "description": "Balance medio",
            "expected_behavior": "Trading normal con mínimo $11"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        balance = case["balance"]
        description = case["description"]
        expected = case["expected_behavior"]
        
        print(f"🎯 CASO {i}: {description}")
        print(f"   Balance: ${balance}")
        print(f"   Comportamiento esperado: {expected}")
        print("-" * 50)
        
        # Crear risk manager para este caso
        risk_manager = AdvancedRiskManager()
        await risk_manager.initialize(balance)
        
        # Obtener mínimo efectivo
        effective_min = risk_manager._get_effective_min_position_value()
        config_min = trading_config.MIN_POSITION_VALUE_USDT
        
        print(f"   💰 Mínimo configurado: ${config_min}")
        print(f"   🧠 Mínimo efectivo: ${effective_min}")
        
        # Verificar si cumple el estándar de $11
        if effective_min >= config_min:
            print(f"   ✅ CUMPLE: Mínimo efectivo >= ${config_min}")
        else:
            print(f"   ⚠️ AJUSTE: Mínimo reducido para balance pequeño")
        
        # Probar cálculo de posición con diferentes símbolos y confianzas
        test_symbols = [
            {"symbol": "BTCUSDT", "price": 107000.0, "confidence": 0.75},
            {"symbol": "ETHUSDT", "price": 4000.0, "confidence": 0.70},
            {"symbol": "XRPUSDT", "price": 2.18, "confidence": 0.80}
        ]
        
        print(f"   📊 Pruebas de cálculo de posición:")
        
        for test in test_symbols:
            symbol = test["symbol"]
            price = test["price"]
            confidence = test["confidence"]
            
            # Calcular cantidad
            quantity = risk_manager.calculate_position_size(symbol, price, confidence)
            position_value = quantity * price
            
            # Verificar resultado
            if quantity > 0:
                status = "✅ APROBADO"
                meets_min = "✅" if position_value >= effective_min else "⚠️"
            else:
                status = "❌ RECHAZADO"
                meets_min = "❌"
            
            print(f"      • {symbol}: {quantity:.6f} (${position_value:.2f}) {meets_min} - {status}")
        
        print()
    
    # Caso especial: Múltiples posiciones
    print("🎯 CASO ESPECIAL: MÚLTIPLES POSICIONES")
    print("=" * 50)
    
    # Usar balance del usuario
    user_balance = 80.81
    risk_manager = AdvancedRiskManager()
    await risk_manager.initialize(user_balance)
    
    print(f"Balance inicial: ${user_balance}")
    
    # Simular apertura de múltiples posiciones
    positions_to_test = [
        {"symbol": "BTCUSDT", "price": 107000.0, "confidence": 0.75},
        {"symbol": "ETHUSDT", "price": 4000.0, "confidence": 0.70},
        {"symbol": "XRPUSDT", "price": 2.18, "confidence": 0.80}
    ]
    
    total_exposure = 0.0
    successful_positions = 0
    
    for i, pos in enumerate(positions_to_test, 1):
        symbol = pos["symbol"]
        price = pos["price"]
        confidence = pos["confidence"]
        
        print(f"\n📊 Posición {i}: {symbol}")
        
        # Verificar límites de riesgo
        can_trade, reason = await risk_manager.check_risk_limits_before_trade(symbol, 'BUY', confidence)
        
        if can_trade:
            # Calcular tamaño
            quantity = risk_manager.calculate_position_size(symbol, price, confidence)
            position_value = quantity * price
            
            if quantity > 0 and position_value >= risk_manager._get_effective_min_position_value():
                # Simular posición exitosa
                from advanced_risk_manager import Position
                from datetime import datetime
                
                position = Position(
                    symbol=symbol,
                    side='BUY',
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    entry_time=datetime.now()
                )
                
                risk_manager.active_positions[symbol] = position
                total_exposure += position_value
                successful_positions += 1
                
                print(f"   ✅ ÉXITO: {quantity:.6f} unidades (${position_value:.2f})")
                print(f"   📊 Exposición acumulada: ${total_exposure:.2f}")
            else:
                print(f"   ❌ FALLO: Cantidad insuficiente o no cumple mínimo")
                print(f"   💡 Cantidad calculada: {quantity:.6f} (${position_value:.2f})")
        else:
            print(f"   ❌ RECHAZADO: {reason}")
    
    # Resumen final
    max_exposure = user_balance * (trading_config.MAX_TOTAL_EXPOSURE_PERCENT / 100)
    exposure_percent = (total_exposure / user_balance) * 100
    
    print(f"\n📊 RESUMEN FINAL:")
    print(f"   💰 Balance inicial: ${user_balance}")
    print(f"   📊 Exposición total: ${total_exposure:.2f} ({exposure_percent:.1f}%)")
    print(f"   ⚖️ Límite exposición: ${max_exposure:.2f} ({trading_config.MAX_TOTAL_EXPOSURE_PERCENT}%)")
    print(f"   🔢 Posiciones exitosas: {successful_positions}/{len(positions_to_test)}")
    print(f"   💵 Mínimo por posición: ${trading_config.MIN_POSITION_VALUE_USDT}")
    
    # Verificaciones finales
    print(f"\n🔍 VERIFICACIONES FINALES:")
    all_positions_valid = True
    
    for symbol, pos in risk_manager.active_positions.items():
        pos_value = pos.quantity * pos.current_price
        meets_min = pos_value >= trading_config.MIN_POSITION_VALUE_USDT
        
        status = "✅" if meets_min else "❌"
        print(f"   {status} {symbol}: ${pos_value:.2f} ({'CUMPLE' if meets_min else 'NO CUMPLE'} mínimo)")
        
        if not meets_min:
            all_positions_valid = False
    
    print(f"\n🎯 RESULTADO GLOBAL:")
    if all_positions_valid and successful_positions > 0:
        print(f"   ✅ ÉXITO: Todas las posiciones cumplen el mínimo de ${trading_config.MIN_POSITION_VALUE_USDT}")
        print(f"   ✅ Sistema configurado correctamente para Binance")
    elif successful_positions > 0:
        print(f"   ⚠️ ADVERTENCIA: Algunas posiciones no cumplen el mínimo estándar")
        print(f"   💡 Considera aumentar el balance para trading óptimo")
    else:
        print(f"   ❌ PROBLEMA: No se pudo crear ninguna posición válida")
        print(f"   💡 Verifica configuración y balance disponible")
    
    print(f"\n🎉 PRUEBAS COMPLETADAS")

if __name__ == "__main__":
    asyncio.run(test_min_position_11_usdt()) 