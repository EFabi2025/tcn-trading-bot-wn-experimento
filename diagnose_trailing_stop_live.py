#!/usr/bin/env python3
"""
🔬 DIAGNÓSTICO ESPECÍFICO DEL TRAILING STOP EN VIVO
Simula tu posición real de BTC y verifica por qué no se activa el trailing stop.
"""

import asyncio
import logging
from datetime import datetime
from config import TradingConfig
from professional_portfolio_manager import ProfessionalPortfolioManager, Position
from real_market_data_provider import RealMarketDataProvider

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

async def diagnose_trailing_stop_live():
    """🔬 Diagnóstico específico del problema de trailing stop"""
    print("="*80)
    print("🔬 DIAGNÓSTICO DEL TRAILING STOP - POSICIÓN REAL DE BTC 🔬")
    print("="*80)
    
    try:
        # 1. Configuración
        print("\n[Paso 1/6] Cargando configuración...")
        config = TradingConfig()
        print(f"   ✅ Configuración cargada")
        
        # 2. Crear managers
        print("\n[Paso 2/6] Inicializando managers...")
        
        # Crear logger
        logger = logging.getLogger(__name__)
        
        # Símbolos de trading
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        
        portfolio_manager = ProfessionalPortfolioManager(config, symbols, logger)
        await portfolio_manager.initialize()
        print(f"   ✅ PortfolioManager inicializado")
        
        data_provider = RealMarketDataProvider(config)
        print(f"   ✅ DataProvider inicializado")
        
        # 3. Simular tu posición REAL de BTC basada en los logs
        print("\n[Paso 3/6] Simulando posición real de BTC...")
        btc_position = Position(
            symbol="BTCUSDT",
            side="BUY",
            size=0.00055877,  # Del log anterior
            entry_price=108736.73,  # Del log anterior
            current_price=108736.73,  # Inicializamos con precio de entrada
            market_value=60.80,  # Del log anterior
            unrealized_pnl_usd=0.0,
            unrealized_pnl_percent=0.0,
            entry_time=datetime.now(),
            duration_minutes=0,
            order_id="21ord_45544532173"
        )
        
        # Configurar trailing stop con valores actuales
        btc_position.trailing_stop_active = False
        btc_position.trailing_stop_price = None
        btc_position.trailing_stop_percent = 0.6  # Tu configuración actual
        btc_position.highest_price_since_entry = btc_position.entry_price
        btc_position.trailing_activation_threshold = 1.0  # 1% para activar
        btc_position.trailing_movements = 0
        
        print(f"   📍 Posición BTC simulada:")
        print(f"      💰 Entrada: ${btc_position.entry_price:.2f}")
        print(f"      📊 Tamaño: {btc_position.size:.8f} BTC")
        print(f"      💵 Valor: ${btc_position.market_value:.2f}")
        print(f"      📈 Trailing distancia: {btc_position.trailing_stop_percent}%")
        print(f"      🎯 Umbral activación: {btc_position.trailing_activation_threshold}%")
        
        # 4. Obtener precio actual de BTC
        print("\n[Paso 4/6] Obteniendo precio actual de BTC...")
        try:
            ticker = await data_provider.get_ticker_price("BTCUSDT")
            current_price = float(ticker['price'])
            print(f"   ✅ Precio actual de BTC: ${current_price:,.2f}")
        except Exception as e:
            print(f"   ⚠️ Error obteniendo precio: {e}. Usando precio de ejemplo.")
            current_price = 110000.00  # Precio ejemplo que generaría ganancia
        
        # 5. Simular varios escenarios de precio
        print("\n[Paso 5/6] Simulando escenarios de trailing stop...")
        
        # Escenario 1: Precio actual (0% ganancia)
        print(f"\n   🔍 ESCENARIO 1: Precio actual (${current_price:,.2f})")
        test_position = Position(**btc_position.__dict__)
        test_position.current_price = current_price
        pnl_percent = ((current_price - test_position.entry_price) / test_position.entry_price) * 100
        print(f"      PnL: {pnl_percent:+.2f}%")
        
        updated_pos, stop_triggered, reason = portfolio_manager.update_trailing_stop_professional(
            test_position, current_price
        )
        
        print(f"      Trailing activo: {'✅ SÍ' if updated_pos.trailing_stop_active else '❌ NO'}")
        if updated_pos.trailing_stop_price:
            print(f"      Trailing precio: ${updated_pos.trailing_stop_price:.2f}")
        print(f"      Stop activado: {'🚨 SÍ' if stop_triggered else '✅ NO'}")
        
        # Escenario 2: +1.1% ganancia (debe activar trailing)
        print(f"\n   🔍 ESCENARIO 2: +1.1% ganancia (debe activar trailing)")
        test_price = test_position.entry_price * 1.011  # +1.1%
        test_position2 = Position(**btc_position.__dict__)
        test_position2.current_price = test_price
        test_position2.highest_price_since_entry = test_price
        pnl_percent = ((test_price - test_position2.entry_price) / test_position2.entry_price) * 100
        print(f"      Precio simulado: ${test_price:,.2f}")
        print(f"      PnL: {pnl_percent:+.2f}%")
        
        updated_pos2, stop_triggered2, reason2 = portfolio_manager.update_trailing_stop_professional(
            test_position2, test_price
        )
        
        print(f"      Trailing activo: {'✅ SÍ' if updated_pos2.trailing_stop_active else '❌ NO'}")
        if updated_pos2.trailing_stop_price:
            print(f"      Trailing precio: ${updated_pos2.trailing_stop_price:.2f}")
            protected_profit = ((updated_pos2.trailing_stop_price - updated_pos2.entry_price) / updated_pos2.entry_price) * 100
            print(f"      Ganancia protegida: {protected_profit:+.2f}%")
        print(f"      Stop activado: {'🚨 SÍ' if stop_triggered2 else '✅ NO'}")
        
        # Escenario 3: +2% ganancia, luego baja a +0.5% (debe ejecutar trailing)
        print(f"\n   🔍 ESCENARIO 3: Pico +2%, luego baja a +0.5% (debe ejecutar trailing)")
        
        # Primero el pico
        peak_price = test_position.entry_price * 1.02  # +2%
        test_position3 = Position(**btc_position.__dict__)
        test_position3.current_price = peak_price
        test_position3.highest_price_since_entry = peak_price
        
        # Activar trailing con el pico
        updated_pos3, _, _ = portfolio_manager.update_trailing_stop_professional(
            test_position3, peak_price
        )
        
        print(f"      Pico alcanzado: ${peak_price:,.2f} (+2.0%)")
        print(f"      Trailing activado: {'✅ SÍ' if updated_pos3.trailing_stop_active else '❌ NO'}")
        if updated_pos3.trailing_stop_price:
            print(f"      Trailing inicial: ${updated_pos3.trailing_stop_price:.2f}")
        
        # Ahora simular la bajada
        current_low_price = test_position.entry_price * 1.005  # +0.5%
        updated_pos3.current_price = current_low_price
        
        final_pos, stop_triggered3, reason3 = portfolio_manager.update_trailing_stop_professional(
            updated_pos3, current_low_price
        )
        
        print(f"      Precio actual: ${current_low_price:,.2f} (+0.5%)")
        print(f"      Stop ejecutado: {'🚨 SÍ' if stop_triggered3 else '❌ NO'}")
        if stop_triggered3:
            print(f"      Razón: {reason3}")
        
        # 6. Diagnóstico final
        print("\n[Paso 6/6] Diagnóstico final...")
        
        if not updated_pos2.trailing_stop_active:
            print("❌ PROBLEMA ENCONTRADO: El trailing stop no se activa en +1.1% de ganancia")
            print("   🔧 Posibles causas:")
            print("      - El umbral de activación no está configurado correctamente")
            print("      - Hay un bug en el método update_trailing_stop_professional")
            print("      - El cálculo del PnL es incorrecto")
        else:
            print("✅ TRAILING STOP FUNCIONA: Se activa correctamente en +1.1%")
            
            if not stop_triggered3:
                print("❌ PROBLEMA ENCONTRADO: El trailing stop no se ejecuta cuando debería")
                print("   🔧 Posibles causas:")
                print("      - El precio de trailing no se calcula correctamente")
                print("      - La condición de ejecución tiene un bug")
            else:
                print("✅ EJECUCIÓN FUNCIONA: El trailing stop se ejecuta correctamente")
        
        print("\n📊 ANÁLISIS COMPLETO:")
        print(f"   1. Activación en +1%: {'✅' if updated_pos2.trailing_stop_active else '❌'}")
        print(f"   2. Protección de ganancias: {'✅' if updated_pos2.trailing_stop_price else '❌'}")
        print(f"   3. Ejecución automática: {'✅' if stop_triggered3 else '❌'}")
        
        # Cerrar conexiones
        await data_provider.client.close()
        
    except Exception as e:
        print(f"💥 Error durante el diagnóstico: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Función principal"""
    print("🚀 Iniciando diagnóstico del trailing stop...")
    try:
        asyncio.run(diagnose_trailing_stop_live())
    except KeyboardInterrupt:
        print("\n⏹️ Diagnóstico interrumpido por el usuario.")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main() 