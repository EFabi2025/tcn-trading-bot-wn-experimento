#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE STOP LOSS Y LIQUIDACIÓN AUTOMÁTICA
¿Por qué las posiciones en pérdida no se liquidan automáticamente?
"""

import asyncio
from simple_professional_manager import TradingManager

async def diagnose_stop_loss_system():
    print("🔍 DIAGNÓSTICO DE STOP LOSS Y LIQUIDACIÓN AUTOMÁTICA")
    print("=" * 60)
    
    manager = TradingManager()
    await manager.initialize()
    
    # Obtener posiciones actuales
    snapshot = await manager.portfolio_manager.get_portfolio_snapshot()
    print(f"📊 Posiciones activas: {len(snapshot.active_positions)}")
    
    if not snapshot.active_positions:
        print("📭 No hay posiciones para analizar")
        return
    
    print("\n🎯 ANÁLISIS DETALLADO DE STOP LOSS:")
    print("-" * 50)
    
    for i, pos in enumerate(snapshot.active_positions, 1):
        print(f"\n📍 POSICIÓN #{i}: {pos.symbol}")
        print(f"   💰 Precio entrada: ${pos.entry_price:.4f}")
        print(f"   📊 Precio actual: ${pos.current_price:.4f}")
        
        # Calcular PnL actual
        pnl_percent = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
        pnl_usd = (pos.current_price - pos.entry_price) * pos.size
        
        print(f"   📈 PnL: {pnl_percent:+.2f}% (${pnl_usd:+.2f})")
        
        # Verificar configuración de stop loss
        stop_loss_price = getattr(pos, 'stop_loss_price', None)
        stop_loss_percent = getattr(pos, 'stop_loss_percent', 3.0)
        
        if stop_loss_price:
            print(f"   🛑 Stop Loss configurado: ${stop_loss_price:.4f}")
            
            # ¿DEBERÍA HABERSE EJECUTADO?
            if pos.current_price <= stop_loss_price:
                print(f"   🚨 ¡CRÍTICO! Stop Loss DEBERÍA HABERSE EJECUTADO")
                print(f"   📉 Precio actual <= Stop Loss")
                
                # Calcular pérdida excesiva
                actual_loss = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                stop_loss_target = -stop_loss_percent
                excess_loss = actual_loss - stop_loss_target
                
                print(f"   💸 Pérdida objetivo: {stop_loss_target:.1f}%")
                print(f"   💸 Pérdida actual: {actual_loss:.2f}%")
                print(f"   💸 Pérdida EXCESIVA: {excess_loss:.2f}%")
                
            else:
                print(f"   ✅ Stop Loss aún no alcanzado")
                remaining_drop = ((pos.current_price - stop_loss_price) / pos.current_price) * 100
                print(f"   📊 Puede bajar {remaining_drop:.2f}% más antes del stop")
        else:
            print(f"   ❌ Stop Loss NO configurado - PROBLEMA CRÍTICO")
    
    # DIAGNÓSTICO GENERAL DEL SISTEMA
    print(f"\n🔧 DIAGNÓSTICO GENERAL DEL SISTEMA:")
    print("-" * 50)
    
    # Verificar balance para ejecutar órdenes
    balance = snapshot.free_usdt
    print(f"   💰 Balance para órdenes: ${balance:.2f}")
    
    # BUSCAR LA CAUSA RAÍZ
    print(f"\n🎯 POSIBLES CAUSAS DEL PROBLEMA:")
    print("-" * 50)
    
    losing_positions = [pos for pos in snapshot.active_positions 
                       if ((pos.current_price - pos.entry_price) / pos.entry_price) * 100 < -2.0]
    
    if losing_positions:
        print(f"   🚨 ENCONTRADAS {len(losing_positions)} posiciones con pérdidas > 2%")
        
        for pos in losing_positions:
            pnl = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
            stop_price = getattr(pos, 'stop_loss_price', 0)
            
            if pos.current_price <= stop_price:
                print(f"   ❌ {pos.symbol}: Pérdida {pnl:.2f}% - DEBERÍA ESTAR LIQUIDADA")
            else:
                print(f"   ⚠️ {pos.symbol}: Pérdida {pnl:.2f}% - Stop aún no alcanzado")
    
    print(f"\n💡 CAUSAS PROBABLES:")
    print(f"   1. 🤖 Sistema de monitoreo no ejecutándose continuamente")
    print(f"   2. 🔧 Función de liquidación automática deshabilitada")
    print(f"   3. 📊 Stop loss configurado pero no monitoreado")
    print(f"   4. ⏰ Falta de loop de verificación cada X segundos")
    print(f"   5. 🛑 Ejecución manual requerida vs automática")

if __name__ == "__main__":
    asyncio.run(diagnose_stop_loss_system()) 