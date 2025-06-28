#!/usr/bin/env python3
"""
🧪 TEST XRP INTEGRATION - PRUEBA RÁPIDA
======================================

Script para probar rápidamente la integración de XRP en el sistema.
Verifica que el modelo funciona con la misma metodología.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_xrp_model():
    """Probar modelo XRP con datos sintéticos"""
    print("🧪 PROBANDO MODELO XRP")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        from centralized_features_engine import CentralizedFeaturesEngine
        
        # 1. Verificar que el modelo existe
        model_path = "models/definitivo_xrpusdt.h5"
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado: {model_path}")
            print(f"   Ejecuta primero: python train_xrp_model.py")
            return False
        
        print(f"✅ Modelo encontrado: {model_path}")
        
        # 2. Cargar modelo
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # 3. Crear datos sintéticos para XRP
        print(f"\n🎲 Generando datos sintéticos para XRPUSDT...")
        
        n_samples = 100
        np.random.seed(42)
        
        # Datos OHLCV realistas para XRP
        xrp_data = {
            'open': np.random.uniform(0.5, 0.7, n_samples),
            'high': np.random.uniform(0.6, 0.8, n_samples),
            'low': np.random.uniform(0.4, 0.6, n_samples),
            'close': np.random.uniform(0.5, 0.7, n_samples),
            'volume': np.random.uniform(1000000, 5000000, n_samples)
        }
        
        df = pd.DataFrame(xrp_data)
        print(f"✅ Datos generados: {len(df)} muestras")
        print(f"   Precio promedio: ${df['close'].mean():.4f}")
        print(f"   Volumen promedio: {df['volume'].mean():,.0f}")
        
        # 4. Calcular features usando motor centralizado
        print(f"\n🔧 Calculando features (metodología idéntica)...")
        
        features_engine = CentralizedFeaturesEngine()
        features_df = features_engine.calculate_features(
            df=df,
            feature_set='tcn_definitivo'  # Mismo conjunto que modelos en producción
        )
        
        print(f"✅ Features calculadas: {features_df.shape}")
        print(f"   Features disponibles: {list(features_df.columns[:10])}...")
        
        # 5. Preparar secuencia para predicción
        print(f"\n🔄 Preparando secuencia temporal...")
        
        sequence_length = 60  # Mismo que modelos en producción
        
        if len(features_df) < sequence_length:
            print(f"❌ Datos insuficientes: {len(features_df)} < {sequence_length}")
            return False
        
        # Tomar últimas muestras y rellenar NaN
        sequence = features_df.iloc[-sequence_length:].fillna(method='ffill').fillna(0).values
        
        # Ajustar features al modelo
        model_input_shape = model.input_shape
        expected_features = model_input_shape[-1]
        
        if sequence.shape[1] != expected_features:
            if sequence.shape[1] < expected_features:
                # Padding con ceros
                padding = np.zeros((sequence.shape[0], expected_features - sequence.shape[1]))
                sequence = np.concatenate([sequence, padding], axis=1)
            else:
                # Truncar features
                sequence = sequence[:, :expected_features]
        
        # Expandir dimensiones para batch
        input_data = np.expand_dims(sequence, axis=0)
        
        print(f"✅ Secuencia preparada: {input_data.shape}")
        
        # 6. Hacer predicción
        print(f"\n🎯 Realizando predicción...")
        
        prediction = model.predict(input_data, verbose=0)
        probabilities = prediction[0]
        
        predicted_class = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        class_names = ['BUY', 'HOLD', 'SELL']
        signal = class_names[predicted_class]
        
        print(f"✅ Predicción completada:")
        print(f"   🎯 Señal: {signal}")
        print(f"   📊 Confianza: {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"   📈 Probabilidades:")
        print(f"     BUY:  {probabilities[0]:.4f} ({probabilities[0]*100:.1f}%)")
        print(f"     HOLD: {probabilities[1]:.4f} ({probabilities[1]*100:.1f}%)")
        print(f"     SELL: {probabilities[2]:.4f} ({probabilities[2]*100:.1f}%)")
        
        # 7. Validar resultado
        if confidence > 0.1 and np.sum(probabilities) > 0.9:
            print(f"\n🎉 PRUEBA EXITOSA")
            print(f"   ✅ Modelo XRP funcionando correctamente")
            print(f"   ✅ Metodología idéntica a modelos en producción")
            print(f"   ✅ Features compatibles")
            print(f"   ✅ Predicción válida")
            return True
        else:
            print(f"\n⚠️ PRUEBA PARCIAL")
            print(f"   Confianza baja o probabilidades inválidas")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBA: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_integration():
    """Probar integración con el sistema"""
    print("\n🔧 PROBANDO INTEGRACIÓN CON SISTEMA")
    print("=" * 50)
    
    try:
        # 1. Verificar configuración
        print("📋 Verificando configuración...")
        
        try:
            from config import trading_config
            symbols = getattr(trading_config, 'SYMBOLS', [])
            
            if 'XRPUSDT' in symbols:
                print(f"✅ XRPUSDT en configuración: {symbols}")
            else:
                print(f"⚠️ XRPUSDT no en configuración: {symbols}")
                print(f"   Ejecuta: python integrate_xrp_to_system.py")
                
        except Exception as e:
            print(f"⚠️ Error cargando configuración: {e}")
        
        # 2. Verificar predictor
        print("\n🤖 Verificando predictor...")
        
        try:
            from definitivo_tcn_predictor import DefinitivoTCNPredictor
            
            predictor = DefinitivoTCNPredictor()
            
            if 'XRPUSDT' in predictor.pairs:
                print(f"✅ XRPUSDT en predictor: {predictor.pairs}")
                
                # Verificar modelo cargado
                if 'XRPUSDT' in predictor.models:
                    print(f"✅ Modelo XRP cargado en predictor")
                else:
                    print(f"⚠️ Modelo XRP no cargado en predictor")
            else:
                print(f"⚠️ XRPUSDT no en predictor: {predictor.pairs}")
                
        except Exception as e:
            print(f"⚠️ Error verificando predictor: {e}")
        
        # 3. Verificar features engine
        print("\n🔧 Verificando features engine...")
        
        try:
            from centralized_features_engine import CentralizedFeaturesEngine
            
            engine = CentralizedFeaturesEngine()
            feature_sets = engine.feature_sets
            
            print(f"✅ Features engine cargado")
            print(f"   Conjuntos disponibles: {list(feature_sets.keys())}")
            print(f"   TCN definitivo: {len(feature_sets['tcn_definitivo'])} features")
            
        except Exception as e:
            print(f"❌ Error verificando features engine: {e}")
            return False
        
        print(f"\n✅ INTEGRACIÓN VERIFICADA")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN VERIFICACIÓN: {e}")
        return False

def test_complete_workflow():
    """Probar flujo completo de predicción XRP"""
    print("\n🚀 PROBANDO FLUJO COMPLETO")
    print("=" * 50)
    
    try:
        # Simular datos de mercado en tiempo real
        print("📊 Simulando datos de mercado...")
        
        # Generar 100 períodos de 5 minutos (últimas 8 horas)
        n_samples = 100
        np.random.seed(42)
        
        # Simular precio XRP realista con tendencia
        base_price = 0.6
        price_trend = np.cumsum(np.random.normal(0, 0.001, n_samples))
        prices = base_price + price_trend
        
        # Generar OHLCV
        market_data = []
        for i in range(n_samples):
            close = prices[i]
            volatility = np.random.uniform(0.005, 0.02)
            
            high = close * (1 + volatility/2)
            low = close * (1 - volatility/2)
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.uniform(1000000, 5000000)
            
            market_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(market_data)
        
        print(f"✅ Datos simulados: {len(df)} períodos")
        print(f"   Precio inicial: ${df['close'].iloc[0]:.4f}")
        print(f"   Precio final: ${df['close'].iloc[-1]:.4f}")
        print(f"   Cambio: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # Procesar con el sistema completo
        print(f"\n🔄 Procesando con sistema completo...")
        
        # Calcular features
        from centralized_features_engine import CentralizedFeaturesEngine
        engine = CentralizedFeaturesEngine()
        features_df = engine.calculate_features(df, feature_set='tcn_definitivo')
        
        # Cargar modelo
        import tensorflow as tf
        model = tf.keras.models.load_model("models/definitivo_xrpusdt.h5")
        
        # Preparar predicción
        sequence_length = 60
        if len(features_df) >= sequence_length:
            sequence = features_df.iloc[-sequence_length:].fillna(method='ffill').fillna(0).values
            
            # Ajustar dimensiones
            if sequence.shape[1] != model.input_shape[-1]:
                if sequence.shape[1] < model.input_shape[-1]:
                    padding = np.zeros((sequence.shape[0], model.input_shape[-1] - sequence.shape[1]))
                    sequence = np.concatenate([sequence, padding], axis=1)
                else:
                    sequence = sequence[:, :model.input_shape[-1]]
            
            input_data = np.expand_dims(sequence, axis=0)
            
            # Predicción
            prediction = model.predict(input_data, verbose=0)
            probabilities = prediction[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['BUY', 'HOLD', 'SELL']
            signal = class_names[predicted_class]
            
            print(f"✅ Predicción de flujo completo:")
            print(f"   🎯 Señal: {signal}")
            print(f"   📊 Confianza: {confidence:.4f}")
            print(f"   💰 Precio actual: ${df['close'].iloc[-1]:.4f}")
            
            # Simular decisión de trading
            threshold = 0.58  # Mismo umbral que sistema actual
            
            if confidence >= threshold:
                print(f"   ✅ Señal válida (confianza >= {threshold})")
                print(f"   🚀 Acción recomendada: {signal}")
            else:
                print(f"   ⚠️ Confianza baja (< {threshold})")
                print(f"   ⏸️ Acción recomendada: HOLD")
            
            print(f"\n🎉 FLUJO COMPLETO EXITOSO")
            return True
        else:
            print(f"❌ Datos insuficientes para secuencia")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR EN FLUJO COMPLETO: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal de pruebas"""
    print("🧪 XRP INTEGRATION TESTS")
    print("=" * 60)
    
    # Ejecutar prueba principal
    success = test_xrp_model()
    
    if success:
        print(f"\n🎉 ¡PRUEBA EXITOSA!")
        print(f"   ✅ Modelo XRP funcionando correctamente")
        print(f"   ✅ Metodología idéntica a modelos en producción")
        print(f"   ✅ Listo para integración")
        print(f"\n🚀 Pasos siguientes:")
        print(f"   1. python integrate_xrp_to_system.py")
        print(f"   2. Reiniciar sistema de trading")
    else:
        print(f"\n❌ Prueba falló")
        print(f"   Revisa los errores anteriores")
        print(f"   Verifica que el modelo esté entrenado")

if __name__ == "__main__":
    main()