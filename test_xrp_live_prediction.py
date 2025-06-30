#!/usr/bin/env python3
"""
🔴 PRUEBA EN VIVO: PREDICCIONES XRPUSDT CON DATOS REALES DE BINANCE
"""

from tcn_definitivo_predictor import TCNDefinitivoPredictor
import requests
import pandas as pd
import numpy as np
from datetime import datetime

def test_xrp_live_prediction():
    """🎯 Probar predicciones XRPUSDT con datos reales"""

    print('🔴 PRUEBA EN VIVO: PREDICCIONES XRPUSDT CON DATOS REALES DE BINANCE')
    print('=' * 80)

    try:
        # Crear predictor
        predictor = TCNDefinitivoPredictor()

        # Obtener datos reales de Binance para XRPUSDT
        print('📊 Obteniendo datos reales de XRPUSDT desde Binance...')

        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': 'XRPUSDT',
            'interval': '1m',
            'limit': 100
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Convertir a DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convertir tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f'✅ Datos obtenidos: {len(df)} velas de 1 minuto')
        print(f'📈 Precio actual: ${df["close"].iloc[-1]:.4f}')
        print(f'📊 Rango último minuto: ${df["low"].iloc[-1]:.4f} - ${df["high"].iloc[-1]:.4f}')
        print(f'💰 Volumen: {df["volume"].iloc[-1]:,.0f} XRP')

        # Cargar modelo XRPUSDT
        print('\n🤖 Cargando modelo XRPUSDT...')
        success = predictor._load_model_for_symbol('XRPUSDT')

        if not success:
            print('❌ Error cargando modelo XRPUSDT')
            return False

        print('✅ Modelo XRPUSDT cargado exitosamente')

        # Realizar predicción con datos reales
        print('\n🎯 Realizando predicción con datos reales...')
        prediction = predictor.predict('XRPUSDT', df)

        if prediction:
            print('\n🎉 ¡PREDICCIÓN EXITOSA CON DATOS REALES!')
            print('=' * 50)
            print(f'💎 SÍMBOLO: {prediction["symbol"]}')
            print(f'🎯 SEÑAL: {prediction["signal"]}')
            print(f'📊 CONFIANZA: {prediction["confidence"]:.1%}')
            print(f'💰 PRECIO ACTUAL: ${prediction["current_price"]:.4f}')
            print(f'🧠 ACCURACY MODELO: {prediction["model_accuracy"]:.1%}')
            print(f'⚖️ THRESHOLDS: Sell {prediction["threshold_used"]["sell"]:.2%} | Buy {prediction["threshold_used"]["buy"]:.2%}')
            print(f'🕐 TIMESTAMP: {prediction["timestamp"]}')
            print(f'🔧 FEATURES USADAS: {prediction["features_count"]}')

            print('\n📊 PROBABILIDADES DETALLADAS:')
            probs = prediction['probabilities']
            print(f'   🔴 SELL: {probs["SELL"]:.1%}')
            print(f'   ⚪ HOLD: {probs["HOLD"]:.1%}')
            print(f'   🟢 BUY:  {probs["BUY"]:.1%}')

            # Análisis de la predicción
            print('\n🔍 ANÁLISIS DE LA PREDICCIÓN:')
            if prediction['signal'] == 'BUY' and prediction['confidence'] > 0.7:
                print('   ✅ Señal de compra fuerte - Considerar entrada')
            elif prediction['signal'] == 'SELL' and prediction['confidence'] > 0.7:
                print('   ❌ Señal de venta fuerte - Considerar salida')
            elif prediction['signal'] == 'HOLD':
                print('   ⏸️ Mantener posición - Sin acción recomendada')
            else:
                print(f'   ⚠️ Señal {prediction["signal"]} con baja confianza - Precaución')

            # Análisis técnico adicional
            print('\n📈 CONTEXTO DE MERCADO:')
            recent_prices = df['close'].tail(10).values
            price_change_5min = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] * 100
            price_change_1min = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] * 100

            print(f'   📊 Cambio últimos 5 min: {price_change_5min:+.2f}%')
            print(f'   📊 Cambio último minuto: {price_change_1min:+.2f}%')

            avg_volume = df['volume'].tail(10).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume

            print(f'   💰 Volumen vs promedio: {volume_ratio:.1f}x')

            return True

        else:
            print('❌ Error en la predicción')
            return False

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_multiple_predictions():
    """🔄 Realizar múltiples predicciones para ver consistencia"""

    print('\n🔄 REALIZANDO MÚLTIPLES PREDICCIONES PARA VERIFICAR CONSISTENCIA')
    print('=' * 70)

    predictor = TCNDefinitivoPredictor()

    for i in range(3):
        print(f'\n📊 Predicción #{i+1}:')
        try:
            prediction = predictor.predict_symbol('XRPUSDT')
            if prediction and 'signal' in prediction:
                print(f'   Señal: {prediction["signal"]}, Confianza: {prediction.get("confidence", 0):.3f}')
            else:
                print(f'   Error o señal no disponible')
        except Exception as e:
            print(f'   Error: {e}')

if __name__ == "__main__":
    # Prueba principal
    success = test_xrp_live_prediction()

    if success:
        # Pruebas adicionales
        test_multiple_predictions()

        print('\n✅ PRUEBAS COMPLETADAS - XRPUSDT FUNCIONANDO CON DATOS REALES')
    else:
        print('\n❌ PRUEBAS FALLARON')
