#!/usr/bin/env python3
"""
🧪 SCRIPT DE PRUEBA PARA SISTEMA ARMONIZADO
Verificar que la integración del motor de features funciona correctamente
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from tcn_hybrid_trainer import TradingConfig, ProfessionalCryptoTrader

def test_configuration_validation():
    """Probar validación de configuraciones"""
    
    print("🧪 Probando validación de configuraciones...")
    
    # Configuración válida
    try:
        config_valid = TradingConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            model_type="tcn_basic",
            feature_set="tcn_final"
        )
        print("✅ Configuración válida creada correctamente")
    except Exception as e:
        print(f"❌ Error en configuración válida: {e}")
    
    # Configuración inválida - timeframe
    try:
        config_invalid = TradingConfig(
            timeframe="invalid"
        )
        print("❌ Se esperaba error con timeframe inválido")
    except ValueError as e:
        print(f"✅ Error esperado capturado: {e}")
    
    # Configuración con fechas
    try:
        config_dates = TradingConfig(
            start_date="2024-01-01",
            end_date="2024-06-01"
        )
        period_info = config_dates.get_training_period_info()
        print(f"✅ Configuración con fechas: {period_info['total_days']} días")
    except Exception as e:
        print(f"❌ Error en configuración con fechas: {e}")

def test_features_engine_integration():
    """Probar integración con motor de features"""
    
    print("\n🧪 Probando integración con motor de features...")
    
    # Crear datos de prueba
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    # Simular datos OHLCV realistas
    base_price = 50000
    returns = np.random.normal(0, 0.01, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, 100)
    }, index=dates)
    
    # Probar diferentes feature sets
    feature_sets = ['tcn_final', 'tcn_definitivo', 'full_set']
    
    for feature_set in feature_sets:
        try:
            config = TradingConfig(
                symbol="BTCUSDT",
                timeframe="5m",
                feature_set=feature_set,
                training_days=1  # Datos mínimos para prueba
            )
            
            trader = ProfessionalCryptoTrader(config)
            features = trader.calculate_technical_features(test_data)
            
            print(f"✅ Feature set '{feature_set}': {len(features.columns)} features calculadas")
            
        except Exception as e:
            print(f"❌ Error con feature set '{feature_set}': {e}")

def test_model_types():
    """Probar diferentes tipos de modelos"""
    
    print("\n🧪 Probando diferentes tipos de modelos...")
    
    model_types = ['tcn_basic', 'tcn_advanced', 'lstm_tcn', 'transformer_tcn']
    
    for model_type in model_types:
        try:
            config = TradingConfig(
                model_type=model_type,
                epochs=1,  # Mínimo para prueba
                batch_size=16
            )
            
            trader = ProfessionalCryptoTrader(config)
            
            # Crear modelo de prueba
            input_shape = (24, 10)  # lookback_periods, num_features
            model = trader.create_model(input_shape)
            
            print(f"✅ Modelo '{model_type}': {model.count_params():,} parámetros")
            
        except Exception as e:
            print(f"❌ Error con modelo '{model_type}': {e}")

async def test_data_fetching():
    """Probar descarga de datos"""
    
    print("\n🧪 Probando descarga de datos...")
    
    try:
        config = TradingConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            training_days=1  # Solo 1 día para prueba rápida
        )
        
        trader = ProfessionalCryptoTrader(config)
        df = await trader.fetch_market_data()
        
        print(f"✅ Datos descargados: {len(df)} registros")
        print(f"   📅 Desde: {df.index.min()}")
        print(f"   📅 Hasta: {df.index.max()}")
        print(f"   📊 Columnas: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error descargando datos: {e}")
        return None

async def run_all_tests():
    """Ejecutar todas las pruebas"""
    
    print("🎯 INICIANDO PRUEBAS DEL SISTEMA ARMONIZADO")
    print("=" * 60)
    
    # Test 1: Validación de configuraciones
    test_configuration_validation()
    
    # Test 2: Integración con motor de features
    test_features_engine_integration()
    
    # Test 3: Tipos de modelos
    test_model_types()
    
    # Test 4: Descarga de datos
    df = await test_data_fetching()
    
    # Test 5: Pipeline completo rápido (si hay datos)
    if df is not None:
        print("\n🧪 Probando pipeline completo...")
        try:
            config = TradingConfig(
                symbol="BTCUSDT",
                timeframe="5m",
                model_type="tcn_basic",
                feature_set="tcn_final",
                epochs=1,
                batch_size=8,
                training_days=1,
                lookback_periods=12
            )
            
            trader = ProfessionalCryptoTrader(config)
            
            # Solo probar cálculo de features y preparación de secuencias
            features = trader.calculate_technical_features(df)
            print(f"✅ Features calculadas: {len(features.columns)}")
            
            # Crear labels de prueba
            df_labeled = df.copy()
            df_labeled['label'] = np.random.choice([0, 1, 2], size=len(df))
            
            if len(df) > config.lookback_periods:
                X, y = trader.prepare_sequences(features, df_labeled['label'])
                print(f"✅ Secuencias preparadas: X={X.shape}, y={y.shape}")
            
        except Exception as e:
            print(f"❌ Error en pipeline completo: {e}")
    
    print("\n🎯 PRUEBAS COMPLETADAS")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_all_tests())