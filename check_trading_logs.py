#!/usr/bin/env python3
"""
📊 CONSULTAR LOGS DE TRADING
Script para revisar la actividad actual del bot de trading
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

def check_trading_logs():
    """📊 Revisar logs recientes del trading"""
    print("📊 CONSULTANDO LOGS DE TRADING")
    print("=" * 50)
    
    # Verificar si existe la base de datos
    db_path = "trading_bot.db"
    if not os.path.exists(db_path):
        print("❌ Base de datos no encontrada")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # 1. Ver eventos recientes (últimas 2 horas)
        print("🕐 EVENTOS RECIENTES (últimas 2 horas):")
        recent_events = pd.read_sql_query("""
            SELECT timestamp, level, category, message 
            FROM system_events 
            WHERE timestamp > datetime('now', '-2 hours')
            ORDER BY timestamp DESC 
            LIMIT 20
        """, conn)
        
        if not recent_events.empty:
            for _, event in recent_events.iterrows():
                emoji = "🔴" if event['level'] == 'ERROR' else "🟡" if event['level'] == 'WARNING' else "🟢"
                print(f"{emoji} {event['timestamp']} [{event['category']}] {event['message']}")
        else:
            print("   No hay eventos recientes")
        
        print("\n" + "-" * 50)
        
        # 2. Ver trades ejecutados hoy
        print("💰 TRADES DE HOY:")
        trades_today = pd.read_sql_query("""
            SELECT * FROM trades 
            WHERE DATE(execution_time) = DATE('now')
            ORDER BY execution_time DESC
        """, conn)
        
        if not trades_today.empty:
            total_pnl = 0
            for _, trade in trades_today.iterrows():
                status_emoji = "✅" if trade['status'] == 'FILLED' else "⏳"
                pnl = trade.get('realized_pnl', 0) or 0
                total_pnl += pnl
                print(f"{status_emoji} {trade['execution_time']} | {trade['symbol']} {trade['side']} | ${trade['entry_price']:.4f} | PnL: ${pnl:+.2f}")
            
            print(f"\n💎 PnL Total Hoy: ${total_pnl:+.2f}")
        else:
            print("   No hay trades hoy")
        
        print("\n" + "-" * 50)
        
        # 3. Ver posiciones activas
        print("📈 POSICIONES ACTIVAS:")
        active_positions = pd.read_sql_query("""
            SELECT * FROM positions 
            WHERE status = 'OPEN'
            ORDER BY entry_time DESC
        """, conn)
        
        if not active_positions.empty:
            for _, pos in active_positions.iterrows():
                duration = datetime.now() - pd.to_datetime(pos['entry_time'])
                print(f"🎯 {pos['symbol']} {pos['side']} | Entry: ${pos['entry_price']:.4f} | Size: {pos['quantity']:.6f} | Duración: {duration}")
        else:
            print("   No hay posiciones activas")
        
        print("\n" + "-" * 50)
        
        # 4. Estadísticas del día
        print("📊 ESTADÍSTICAS DEL DÍA:")
        
        # Contar trades
        trade_count = len(trades_today)
        wins = len(trades_today[trades_today['realized_pnl'] > 0]) if not trades_today.empty else 0
        losses = len(trades_today[trades_today['realized_pnl'] < 0]) if not trades_today.empty else 0
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        
        print(f"   🔢 Total trades: {trade_count}")
        print(f"   🟢 Ganadores: {wins}")
        print(f"   🔴 Perdedores: {losses}")
        print(f"   📊 Win rate: {win_rate:.1f}%")
        
        # Predicciones TCN recientes
        print("\n🤖 PREDICCIONES TCN RECIENTES:")
        try:
            predictions = pd.read_sql_query("""
                SELECT timestamp, symbol, predicted_action, confidence 
                FROM tcn_predictions 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC 
                LIMIT 10
            """, conn)
            
            if not predictions.empty:
                for _, pred in predictions.iterrows():
                    conf_emoji = "🔥" if pred['confidence'] > 0.8 else "⚡" if pred['confidence'] > 0.7 else "📊"
                    print(f"{conf_emoji} {pred['timestamp']} | {pred['symbol']} {pred['predicted_action']} ({pred['confidence']:.1%})")
            else:
                print("   No hay predicciones recientes")
        except:
            print("   Tabla de predicciones no disponible")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error consultando base de datos: {e}")

if __name__ == "__main__":
    check_trading_logs() 