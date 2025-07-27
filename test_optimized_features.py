#!/usr/bin/env python3
"""
🧪 TEST MOTOR DE FEATURES OPTIMIZADO
====================================

Script para comparar el motor original vs optimizado
y verificar las mejoras en calidad de features.
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta

# Importar ambos motores
from centralized_features_engine2 import CentralizedFeaturesEngine
from centralized_features_engine_optimized import OptimizedFeaturesEngine

async def get_test_data(symbol="BTCUSDT", days=7):
    """Obtener datos reales para prueba"""

    print(f"📊 Obteniendo {days} días de datos para {symbol}...")

    base_url = "https://api.binance.com"
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    async with aiohttp.ClientSession() as session:
        url = f"{base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        all_data = []
        current_start = start_time

        while current_start < end_time:
            params['startTime'] = current_start

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data:
                        break
                    all_data.extend(data)
                    current_start = data[-1][6] + 1
                else:
                    print(f"❌ Error API: {response.status}")
                    break

            await asyncio.sleep(0.1)

    # Convertir a DataFrame
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]
    df = pd.DataFrame(all_data, columns=columns)

    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp').sort_index()

    print(f"✅ Obtenidos {len(df)} registros")
    return df

def analyze_features_quality(features_df, name):
    """Analizar calidad de las features"""

    print(f"\n📊 ANÁLISIS DE FEATURES: {name}")
    print("=" * 50)

    # Estadísticas básicas
    print(f"   🔢 Total features: {len(features_df.columns)}")
    print(f"   📈 Filas de datos: {len(features_df)}")

    # NaN analysis
    nan_count = features_df.isnull().sum().sum()
    nan_pct = nan_count / (len(features_df) * len(features_df.columns)) * 100
    print(f"   🔍 NaN total: {nan_count} ({nan_pct:.2f}%)")

    # Valores infinitos
    inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"   ♾️ Infinitos: {inf_count}")

    # Rango de valores
    numeric_df = features_df.select_dtypes(include=[np.number])
    print(f"   📊 Rango valores: [{numeric_df.min().min():.4f}, {numeric_df.max().max():.4f}]")

    # Análisis de correlaciones
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().abs()

        # Correlaciones altas (>0.8)
        high_corr_count = 0
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr_count += 1
                    if len(high_corr_pairs) < 5:  # Solo mostrar primeros 5
                        high_corr_pairs.append({
                            'pair': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                            'corr': corr_matrix.iloc[i, j]
                        })

        print(f"   🔗 Correlaciones altas (>0.8): {high_corr_count}")
        if high_corr_pairs:
            print("   📋 Ejemplos de correlaciones altas:")
            for pair in high_corr_pairs:
                print(f"      - {pair['pair']}: {pair['corr']:.3f}")

        avg_corr = corr_matrix.mean().mean()
        print(f"   📈 Correlación promedio: {avg_corr:.3f}")

        # Features con mayor varianza (más informativas)
        feature_variance = numeric_df.var().sort_values(ascending=False)
        print(f"   🎯 Top 5 features por varianza:")
        for i, (feature, variance) in enumerate(feature_variance.head().items()):
            print(f"      {i+1}. {feature}: {variance:.6f}")

async def compare_engines():
    """Comparar motor original vs optimizado"""

    print("🚀 COMPARACIÓN DE MOTORES DE FEATURES")
    print("=" * 80)

    # Obtener datos de prueba
    df = await get_test_data("BTCUSDT", days=5)

    # Crear motores
    original_engine = CentralizedFeaturesEngine()
    optimized_engine = OptimizedFeaturesEngine()

    print(f"\n🔄 Calculando features con ambos motores...")

    # Motor original
    try:
        original_features = original_engine.calculate_features(df, 'tcn_definitivo')
        analyze_features_quality(original_features, "MOTOR ORIGINAL")
    except Exception as e:
        print(f"❌ Error con motor original: {e}")
        original_features = None

    # Motor optimizado
    try:
        optimized_features = optimized_engine.calculate_features(df, 'optimized_tcn')
        analyze_features_quality(optimized_features, "MOTOR OPTIMIZADO")
    except Exception as e:
        print(f"❌ Error con motor optimizado: {e}")
        optimized_features = None

    # Comparación directa
    if original_features is not None and optimized_features is not None:
        print(f"\n🎯 COMPARACIÓN DIRECTA")
        print("=" * 50)
        print(f"   Original: {len(original_features.columns)} features")
        print(f"   Optimizado: {len(optimized_features.columns)} features")
        print(f"   Reducción: {len(original_features.columns) - len(optimized_features.columns)} features")

        # Análisis de correlaciones
        if hasattr(optimized_engine, 'analyze_feature_correlations'):
            corr_analysis = optimized_engine.analyze_feature_correlations(optimized_features)
            print(f"   Correlaciones altas optimizado: {len(corr_analysis['high_correlations'])}")

    # Probar conjuntos mínimos
    print(f"\n🎯 PROBANDO CONJUNTOS MÍNIMOS")
    print("=" * 50)

    for feature_set in ['minimal_power', 'directional_only']:
        try:
            minimal_features = optimized_engine.calculate_features(df, feature_set)
            analyze_features_quality(minimal_features, f"CONJUNTO {feature_set.upper()}")
        except Exception as e:
            print(f"❌ Error con {feature_set}: {e}")

def test_directional_features():
    """Test específico de features direccionales"""

    print(f"\n🎯 TEST DE FEATURES DIRECCIONALES")
    print("=" * 50)

    # Crear datos simulados con tendencia clara
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')

    # Simular tendencia alcista
    base_price = 50000
    trend = np.linspace(0, 0.05, 200)  # 5% de subida
    noise = np.random.normal(0, 0.005, 200)  # Ruido
    returns = trend + noise
    prices = base_price * np.exp(np.cumsum(returns))

    # Crear OHLCV realista
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, 200)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, 200))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, 200))),
        'close': prices,
        'volume': np.random.lognormal(8, 0.3, 200)
    }, index=dates)

    # Calcular features direccionales
    engine = OptimizedFeaturesEngine()
    directional_features = engine.calculate_features(test_data, 'directional_only')

    print(f"✅ Features direccionales calculadas: {len(directional_features.columns)}")

    # Verificar que detecta la tendencia
    if 'trend_strength' in directional_features.columns:
        final_trend = directional_features['trend_strength'].iloc[-10:].mean()
        print(f"🎯 Trend strength detectado: {final_trend:.3f} (debería ser > 0)")

    if 'ema_crossover' in directional_features.columns:
        final_crossover = directional_features['ema_crossover'].iloc[-10:].mean()
        print(f"🎯 EMA crossover detectado: {final_crossover:.6f} (debería ser > 0)")

    if 'momentum_acceleration' in directional_features.columns:
        avg_momentum = directional_features['momentum_acceleration'].iloc[-20:].mean()
        print(f"🎯 Momentum acceleration: {avg_momentum:.6f}")

async def main():
    """Función principal de testing"""

    print("🧪 INICIANDO TESTS DE MOTOR OPTIMIZADO")
    print("=" * 80)

    # Test 1: Comparación de motores
    await compare_engines()

    # Test 2: Features direccionales
    test_directional_features()

    print(f"\n✅ TESTS COMPLETADOS")
    print("=" * 80)
    print("🎯 RECOMENDACIONES:")
    print("1. Si las correlaciones del motor optimizado son menores, ¡migra!")
    print("2. Prueba el conjunto 'minimal_power' para modelos más rápidos")
    print("3. Usa 'directional_only' para señales más claras")
    print("4. Las features direccionales deberían mostrar la tendencia correcta")

if __name__ == "__main__":
    asyncio.run(main())
