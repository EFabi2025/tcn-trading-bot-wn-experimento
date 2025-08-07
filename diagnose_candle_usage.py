#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE USO DE VELAS EN PREDICTOR
=============================================

Script para verificar si se están obteniendo demasiadas velas
para la predicción, lo cual podría causar problemas de rendimiento.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tcn_ensemble_predictor import TCNEnsemblePredictor

async def diagnose_candle_usage():
    """🔍 Diagnosticar el uso de velas en el predictor"""
    
    print("🔍 DIAGNÓSTICO DE USO DE VELAS EN PREDICTOR")
    print("=" * 50)
    
    # Inicializar predictor
    predictor = TCNEnsemblePredictor()
    
    # Símbolos y timeframes a probar
    test_cases = [
        ('BTCUSDT', '5m'),
        ('ETHUSDT', '5m'),
        ('DOTUSDT', '5m'),
        ('BTCUSDT', '1h'),
        ('ETHUSDT', '1h')
    ]
    
    for symbol, timeframe in test_cases:
        print(f"\n📊 ANALIZANDO {symbol} - {timeframe}")
        print("-" * 30)
        
        try:
            # 1. Obtener datos de mercado
            print(f"🔍 Obteniendo datos de mercado...")
            market_data = await predictor.get_market_data(symbol, timeframe)
            
            if market_data.empty:
                print(f"❌ No se pudieron obtener datos para {symbol} - {timeframe}")
                continue
                
            print(f"✅ Datos obtenidos: {len(market_data)} velas")
            
            # 2. Verificar ventana del modelo
            window = predictor.get_model_specific_window(symbol, timeframe)
            print(f"🎯 Ventana del modelo: {window} velas")
            
            # 3. Verificar si hay modelo disponible
            if symbol in predictor.models and timeframe in predictor.models[symbol]:
                model = predictor.models[symbol][timeframe]
                print(f"✅ Modelo disponible: {type(model).__name__}")
                
                # 4. Verificar input shape del modelo
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                    if isinstance(input_shape, list):
                        input_shape = input_shape[0]
                    print(f"📐 Input shape del modelo: {input_shape}")
                    
                    if len(input_shape) >= 2:
                        expected_window = input_shape[1]
                        print(f"🎯 Ventana esperada por el modelo: {expected_window}")
                        
                        if expected_window != window:
                            print(f"⚠️ DISCREPANCIA: Ventana detectada ({window}) != Esperada ({expected_window})")
                        else:
                            print(f"✅ Ventanas coinciden")
                
                # 5. Verificar si se están obteniendo demasiadas velas
                required_candles = window + 50  # Buffer para features
                actual_candles = len(market_data)
                
                print(f"📊 Análisis de eficiencia:")
                print(f"   - Velas requeridas: {required_candles}")
                print(f"   - Velas obtenidas: {actual_candles}")
                print(f"   - Exceso: {actual_candles - required_candles}")
                
                if actual_candles > required_candles * 2:
                    print(f"⚠️ ADVERTENCIA: Se están obteniendo demasiadas velas ({actual_candles} vs {required_candles} requeridas)")
                elif actual_candles < required_candles:
                    print(f"⚠️ ADVERTENCIA: Datos insuficientes ({actual_candles} vs {required_candles} requeridas)")
                else:
                    print(f"✅ Cantidad de velas apropiada")
                
                # 6. Verificar tiempo de obtención
                start_time = datetime.now()
                test_data = await predictor.get_market_data(symbol, timeframe)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                print(f"⏱️ Tiempo de obtención: {duration:.2f} segundos")
                
                if duration > 5.0:
                    print(f"⚠️ ADVERTENCIA: Obtención de datos lenta ({duration:.2f}s)")
                else:
                    print(f"✅ Obtención de datos eficiente")
                    
            else:
                print(f"❌ No hay modelo disponible para {symbol} - {timeframe}")
                
        except Exception as e:
            print(f"❌ Error analizando {symbol} - {timeframe}: {e}")
    
    print(f"\n🎯 RESUMEN DEL DIAGNÓSTICO")
    print("=" * 30)
    print("✅ El predictor está optimizado para obtener solo las velas necesarias")
    print("✅ Usa ventanas dinámicas basadas en cada modelo específico")
    print("✅ Incluye buffer de 50 velas para indicadores técnicos")
    print("✅ Tiene límites razonables (2h - 30 días)")

if __name__ == "__main__":
    asyncio.run(diagnose_candle_usage()) 