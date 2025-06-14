#!/usr/bin/env python3
"""
🔍 EXPLORAR ESTRUCTURA DE BASE DE DATOS
"""

import sqlite3
import pandas as pd

def explore_database():
    print("🔍 EXPLORANDO BASE DE DATOS")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect("trading_bot.db")
        
        # Ver todas las tablas
        tables = pd.read_sql_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table'
            ORDER BY name
        """, conn)
        
        print("📋 TABLAS DISPONIBLES:")
        for table in tables['name']:
            print(f"   📊 {table}")
            
            # Ver estructura de cada tabla
            try:
                columns = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                print(f"      Columnas: {', '.join(columns['name'].tolist())}")
                
                # Ver cantidad de registros
                count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
                print(f"      Registros: {count['count'].iloc[0]}")
                
                # Ver algunos datos recientes si hay
                if count['count'].iloc[0] > 0:
                    sample = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 3", conn)
                    print(f"      Últimos registros:")
                    for _, row in sample.iterrows():
                        print(f"        {dict(row)}")
                
                print()
            except Exception as e:
                print(f"      Error: {e}")
                print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    explore_database() 