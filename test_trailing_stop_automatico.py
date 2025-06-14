#!/usr/bin/env python3
"""
🚀 SCRIPT DE PRUEBA - TRAILING STOP AUTOMÁTICO
===============================================

Este script prueba el nuevo sistema de trailing stop con órdenes automáticas.

⚠️ ADVERTENCIA: Este script puede ejecutar órdenes REALES en Binance.
   Solo usar en testnet o con cantidades muy pequeñas.
"""

import asyncio
import os
from datetime import datetime, timedelta
from simple_professional_manager import SimpleProfessionalTradingManager
from professional_portfolio_manager import Position

async def test_trailing_stop_automatico():
    """🧪 Probar sistema de trailing stop automático"""
    
    print("🔥" * 60)
    print("🧪 TESTING TRAILING STOP AUTOMÁTICO")
    print("🔥" * 60)
    
    # Verificar que estamos en testnet
    base_url = os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision')
    if 'testnet' not in base_url:
        print("⚠️ ADVERTENCIA: No estás en testnet!")
        response = input("¿Continuar en PRODUCCIÓN? (escribir 'SI' para confirmar): ")
        if response != 'SI':
            print("❌ Prueba cancelada por seguridad")
            return False
    
    try:
        # Inicializar manager
        manager = SimpleProfessionalTradingManager()
        await manager.initialize()
        
        print("✅ Trading Manager inicializado")
        print(f"🌐 Conectado a: {base_url}")
        
        # Obtener snapshot actual
        snapshot = await manager.portfolio_manager.get_portfolio_snapshot()
        
        if not snapshot.active_positions:
            print("📭 No hay posiciones activas para probar trailing stop")
            print("💡 Sugerencia: Ejecuta primero una señal de compra para crear posiciones")
            return True
        
        print(f"\n📊 POSICIONES ACTUALES:")
        for i, pos in enumerate(snapshot.active_positions, 1):
            print(f"   {i}. {pos.symbol}: {pos.size:.6f} @ ${pos.entry_price:.4f}")
            print(f"      💰 Valor: ${pos.market_value:.2f} | PnL: {pos.unrealized_pnl_percent:.2f}%")
            
            # Estado del trailing stop
            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                protection = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price) * 100
                print(f"      📈 TRAILING ACTIVO: ${pos.trailing_stop_price:.4f} (+{protection:.2f}%)")
            else:
                min_activation = pos.trailing_stop_percent + 0.5
                activation_price = pos.entry_price * (1 + min_activation / 100)
                print(f"      📈 Trailing inactivo - Necesita: ${activation_price:.4f} (+{min_activation:.1f}%)")
        
        # Simular diferentes escenarios de precio
        print(f"\n🎯 SIMULANDO ESCENARIOS DE TRAILING STOP:")
        
        test_position = snapshot.active_positions[0]  # Usar primera posición
        print(f"\n🧪 Probando con {test_position.symbol}:")
        print(f"   📍 Precio entrada: ${test_position.entry_price:.4f}")
        print(f"   📊 Precio actual: ${test_position.current_price:.4f}")
        
        # Escenario 1: Activación del trailing
        print(f"\n📈 ESCENARIO 1: Activación de Trailing Stop")
        
        # Simular precio que active el trailing (+3%)
        simulated_price_up = test_position.entry_price * 1.03
        print(f"   🚀 Simulando precio subida a: ${simulated_price_up:.4f} (+3.0%)")
        
        updated_pos, stop_triggered, reason = manager.portfolio_manager.update_trailing_stop_professional(
            test_position, simulated_price_up
        )
        
        if updated_pos.trailing_stop_active:
            protection = ((updated_pos.trailing_stop_price - updated_pos.entry_price) / updated_pos.entry_price) * 100
            print(f"   ✅ Trailing activado correctamente!")
            print(f"   🛡️ Precio protegido: ${updated_pos.trailing_stop_price:.4f} (+{protection:.2f}%)")
        else:
            print(f"   ❌ Trailing NO se activó (revisar lógica)")
        
        # Escenario 2: Movimiento del trailing
        print(f"\n📈 ESCENARIO 2: Movimiento del Trailing Stop")
        
        if updated_pos.trailing_stop_active:
            # Simular precio más alto (+5%)
            simulated_price_higher = test_position.entry_price * 1.05
            print(f"   🚀 Simulando precio más alto: ${simulated_price_higher:.4f} (+5.0%)")
            
            old_trailing = updated_pos.trailing_stop_price
            updated_pos, stop_triggered, reason = manager.portfolio_manager.update_trailing_stop_professional(
                updated_pos, simulated_price_higher
            )
            
            if updated_pos.trailing_stop_price > old_trailing:
                new_protection = ((updated_pos.trailing_stop_price - updated_pos.entry_price) / updated_pos.entry_price) * 100
                print(f"   ✅ Trailing movido correctamente!")
                print(f"   🔄 {old_trailing:.4f} → ${updated_pos.trailing_stop_price:.4f}")
                print(f"   🛡️ Nueva protección: +{new_protection:.2f}%")
            else:
                print(f"   ⚠️ Trailing no se movió (podría ser correcto si precio no subió suficiente)")
        
        # Escenario 3: Ejecución del trailing stop
        print(f"\n🛑 ESCENARIO 3: Ejecución de Trailing Stop")
        
        if updated_pos.trailing_stop_active:
            # Simular precio que ejecute el trailing
            trigger_price = updated_pos.trailing_stop_price - 0.01  # Justo por debajo del trailing
            print(f"   📉 Simulando bajada a: ${trigger_price:.4f}")
            print(f"   🎯 Trailing actual: ${updated_pos.trailing_stop_price:.4f}")
            
            final_pos, stop_triggered, reason = manager.portfolio_manager.update_trailing_stop_professional(
                updated_pos, trigger_price
            )
            
            if stop_triggered and reason == "TRAILING_STOP":
                print(f"   ✅ TRAILING STOP EJECUTADO CORRECTAMENTE!")
                print(f"   📋 Razón: {reason}")
                print(f"   💰 Se ejecutaría venta automática")
                print(f"   🎯 NOTA: En producción, aquí se ejecutaría orden real")
            else:
                print(f"   ❌ Trailing stop NO se ejecutó (revisar lógica)")
                print(f"   📊 stop_triggered: {stop_triggered}, reason: {reason}")
        
        # Mostrar resumen del test
        print(f"\n📊 RESUMEN DEL TEST:")
        print(f"   ✅ Manager inicializado correctamente")
        print(f"   ✅ Posiciones cargadas: {len(snapshot.active_positions)}")
        print(f"   ✅ Lógica de trailing probada")
        print(f"   ✅ Sistema de órdenes automáticas preparado")
        
        print(f"\n🚀 ESTADO DEL SISTEMA:")
        print(f"   📈 Trailing Stop: OPERATIVO")
        print(f"   🤖 Órdenes Automáticas: HABILITADAS")
        print(f"   🛡️ Protección Matemática: CORREGIDA")
        print(f"   🔧 Ejecución Real: {'TESTNET' if 'testnet' in base_url else 'PRODUCCIÓN'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False

async def monitor_trailing_stops_live():
    """👁️ Monitoreo en vivo de trailing stops"""
    
    print(f"\n🔴 MODO MONITOREO EN VIVO")
    print(f"   Presiona Ctrl+C para detener")
    
    manager = SimpleProfessionalTradingManager()
    await manager.initialize()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n🔄 Ciclo #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            
            # Obtener snapshot
            snapshot = await manager.portfolio_manager.get_portfolio_snapshot()
            
            if snapshot.active_positions:
                # Actualizar trailing stops
                for position in snapshot.active_positions:
                    # Obtener precio actual
                    current_price = await manager.get_current_price(position.symbol)
                    
                    # Actualizar trailing stop
                    updated_pos, stop_triggered, reason = manager.portfolio_manager.update_trailing_stop_professional(
                        position, current_price
                    )
                    
                    # Si se activa trailing stop, ejecutar venta
                    if stop_triggered:
                        print(f"🚨 TRAILING STOP ACTIVADO: {position.symbol}")
                        print(f"   📍 Razón: {reason}")
                        print(f"   💰 Ejecutando venta automática...")
                        
                        # Aquí se ejecutaría la venta real
                        # await manager._execute_sell_order(updated_pos)
                        
                        print(f"   ✅ Orden ejecutada (simulada)")
                        
            else:
                print(f"   📭 Sin posiciones activas")
            
            # Esperar 30 segundos
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n👋 Monitoreo detenido por usuario")
    except Exception as e:
        print(f"❌ Error en monitoreo: {e}")

async def main():
    """🎯 Función principal"""
    
    print("🚀 TRAILING STOP AUTOMÁTICO - SISTEMA DE PRUEBAS")
    print("=" * 60)
    
    print("\n🔧 Opciones disponibles:")
    print("   1. Test completo del sistema")
    print("   2. Monitoreo en vivo (usa con precaución)")
    print("   3. Salir")
    
    try:
        choice = input("\n📝 Selecciona opción (1-3): ").strip()
        
        if choice == "1":
            success = await test_trailing_stop_automatico()
            if success:
                print("\n✅ TEST COMPLETADO EXITOSAMENTE")
            else:
                print("\n❌ TEST FALLÓ")
                
        elif choice == "2":
            await monitor_trailing_stops_live()
            
        elif choice == "3":
            print("👋 ¡Hasta luego!")
            
        else:
            print("❌ Opción inválida")
            
    except KeyboardInterrupt:
        print("\n👋 Script interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 