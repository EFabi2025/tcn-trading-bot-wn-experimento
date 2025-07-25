#!/usr/bin/env python3
"""
🎯 TCN HYBRID TRAINER - LO MEJOR DE AMBOS MUNDOS
Combina: Etiquetado dinámico por volatilidad + Arquitectura TCN Simplificada.
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import talib
import warnings
import pickle
import os
from collections import Counter
warnings.filterwarnings('ignore')

from centralized_features_engine2 import CentralizedFeaturesEngine


class TCNHybridTrainer:
    """🎯 Entrenador híbrido: Etiquetado por volatilidad + Arquitectura Simplificada"""

    def __init__(self, atr_multiplier: float = 1.5, atr_period: int = 24, prediction_horizon: int = 10):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        self.lookback_window = 24
        self.prediction_horizon = prediction_horizon 
        self.features_engine = CentralizedFeaturesEngine()
        
        # 🎯 PARÁMETROS DINÁMICOS
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period

    async def get_real_market_data(self, symbol: str, days: int = 180) -> pd.DataFrame:
        """📊 Obtener datos reales de mercado (del definitivo_trainer.py)"""

        print(f"📊 Obteniendo {days} días de datos reales para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
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
        df = pd.DataFrame(all_data, columns=columns)  # type: ignore

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros de {symbol}")
        return df

    def create_volatility_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """🎯 Crear etiquetas de 3 clases (SELL/HOLD/BUY) basadas en volatilidad (ATR)"""
        print(f"🎯 Creando etiquetas de 3 clases con ATR (x{self.atr_multiplier}, {self.prediction_horizon} mins)...")

        df_copy = df.copy()
        
        # 1. Calcular ATR
        df_copy['atr'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=self.atr_period)

        # 2. Definir barreras dinámicas y resultado futuro
        df_copy['upper_barrier'] = df_copy['close'] + (df_copy['atr'] * self.atr_multiplier)
        df_copy['lower_barrier'] = df_copy['close'] - (df_copy['atr'] * self.atr_multiplier)
        
        # 3. Encontrar si alguna barrera es tocada en el futuro
        # Usamos rolling window sobre el futuro para encontrar el max/min en el horizonte
        df_copy['future_max_price'] = df_copy['high'].shift(-self.prediction_horizon).rolling(window=self.prediction_horizon).max()
        df_copy['future_min_price'] = df_copy['low'].shift(-self.prediction_horizon).rolling(window=self.prediction_horizon).min()

        # 4. Limpiar NaNs generados por ATR y rolling windows
        df_copy.dropna(inplace=True)

        # 5. Aplicar la lógica de etiquetado
        def get_label(row):
            touched_upper = row['future_max_price'] >= row['upper_barrier']
            touched_lower = row['future_min_price'] <= row['lower_barrier']
            
            if touched_upper and not touched_lower:
                return 2  # BUY
            elif touched_lower and not touched_upper:
                return 0  # SELL
            else:
                # Si ninguna es tocada, o ambas lo son (indecisión), es HOLD.
                return 1 # HOLD

        df_copy['label'] = df_copy.apply(get_label, axis=1)

        # Verificar distribución
        label_counts = df_copy['label'].value_counts().sort_index()
        total = len(df_copy)

        print("📊 Distribución de etiquetas (3 clases):")
        class_names = ['SELL (0)', 'HOLD (1)', 'BUY (2)']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_copy.drop(columns=['atr', 'upper_barrier', 'lower_barrier', 'future_max_price', 'future_min_price'])

    def prepare_training_data(self, df_labeled: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento (híbrido)"""

        print("🔧 Preparando datos para entrenamiento...")

        # Alinear labels y features usando el índice
        common_index = df_labeled.index.intersection(features.index)
        
        df_labeled_aligned = df_labeled.loc[common_index]
        features_aligned = features.loc[common_index]

        # Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        # Normalizar features
        scaler = RobustScaler()  # Más robusto a outliers
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # Crear secuencias temporales
        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            # Secuencia de features
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)

            # Label correspondiente
            y.append(df_labeled_aligned['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features utilizadas: {len(feature_columns)}")

        # 🎯 CALCULAR CLASS WEIGHTS PARA BALANCEAR (3 CLASES)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"🎯 Class weights calculados:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, weight in class_weight_dict.items():
            print(f"   - {class_names[i]}: {weight:.3f}")

        return X, y, scaler, feature_columns, class_weight_dict

    def create_simplified_tcn_model(self, input_shape: tuple):
        """🎯 Modelo TCN simplificado para mayor robustez"""
        
        print("🎯 Creando modelo TCN SIMPLIFICADO (estilo CNN)...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            tf.keras.layers.LayerNormalization(),
            
            # Bloques TCN más simples
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.1),
            
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.15),
            
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.2),
            
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.2),
            
            tf.keras.layers.GlobalAveragePooling1D(),
            
            # Capas densas más simples con regularización L2
            tf.keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            
            # 🎯 Capa de salida para 3 CLASES
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Optimizador y loss para 3 CLASES
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo TCN SIMPLIFICADO (3 Clases) creado: {model.count_params():,} parámetros")
        return model

    def analyze_training_stability(self, history):
        """📊 Analiza la estabilidad del entrenamiento (del V3)"""
        
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        
        # Calcular volatilidad de últimas 10 epochs
        if len(val_acc) >= 10:
            recent_acc_std = np.std(val_acc[-10:])
            recent_loss_std = np.std(val_loss[-10:])
            
            print(f"📊 Estabilidad últimas 10 epochs:")
            print(f"   Val_accuracy std: {recent_acc_std:.4f}")
            print(f"   Val_loss std: {recent_loss_std:.4f}")
            
            # Criterios de estabilidad
            stable_acc = recent_acc_std < 0.02  # 2% variación
            stable_loss = recent_loss_std < 0.1
            
            if stable_acc and stable_loss:
                print("✅ Entrenamiento estable")
            else:
                print("⚠️ Entrenamiento inestable - considerar ajustes")
                
            return stable_acc and stable_loss
        
        return True

    async def train_hybrid_model(self, symbol: str) -> bool:
        """🎯 Entrenamiento híbrido definitivo"""

        print(f"\n🎯 ENTRENANDO MODELO TCN HÍBRIDO PARA {symbol}")
        print("=" * 70)
        print("🔄 Etiquetado: Dinámico por Volatilidad (ATR) | Arquitectura: TCN v3 Simplificada")
        print("=" * 70)

        try:
            # 1. Obtener datos reales (180 DÍAS ES CRÍTICO PARA 3 CLASES)
            df = await self.get_real_market_data(symbol, days=180)

            # 2. Crear features usando el motor centralizado
            print(f"🔄 Calculando features con motor centralizado...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            print(f"✅ {len(features.columns)} features técnicos creados")

            # 3. Crear etiquetas de 3 clases basadas en volatilidad
            df_labeled = self.create_volatility_based_labels(df)

            # 4. Preparar datos de entrenamiento
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # 5. Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear modelo TCN simplificado
            model = self.create_simplified_tcn_model((X.shape[1], X.shape[2]))

            # 7. Directorio híbrido V3 (para diferenciarlo)
            model_dir = f'models/definitivo_v3_5m_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)

            # 8. Callbacks avanzados
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=15,
                    restore_best_weights=True,
                    monitor='val_accuracy',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=8,
                    factor=0.7,
                    monitor='val_loss',
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{model_dir}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    verbose=1
                ),
                tf.keras.callbacks.TerminateOnNaN()
            ]

            # 9. Entrenar con class weights
            print("🚀 Entrenando modelo TCN SIMPLIFICADO...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                shuffle=True
            )

            # 10. Análisis de estabilidad
            is_stable = self.analyze_training_stability(history)

            # 11. Evaluar modelo
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ RESULTADOS FINALES:")
            print(f"   - Loss: {test_loss:.3f}")
            print(f"   - Accuracy: {test_acc:.3f}")
            print(f"   - Estabilidad: {'SÍ' if is_stable else 'NO'}")

            # Criterio de aceptación
            if test_acc < 0.65:
                print("⚠️ Accuracy insuficiente, pero guardando modelo")

            # 12. Verificar distribución de predicciones
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)

            pred_counts = Counter(y_pred_classes)
            print(f"\n📊 Distribución de predicciones en test:")
            class_names = ['SELL', 'HOLD', 'BUY']
            for i, name in enumerate(class_names):
                count = pred_counts.get(i, 0)
                pct = count / len(y_pred_classes) * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")

            # 13. Guardar modelo y componentes (compatible con predictor)
            print("💾 Guardando modelo híbrido...")

            # Guardar modelo principal
            model.save(f'{model_dir}/model.h5')
            print("✅ Modelo principal guardado")

            # Guardar scaler
            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print("✅ Scaler guardado")

            # Guardar feature columns
            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)
            print("✅ Feature columns guardados")

            # Guardar class weights
            with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)
            print("✅ Class weights guardados")

            # Guardar métricas híbridas
            hybrid_metrics = {
                'final_accuracy': test_acc,
                'final_loss': test_loss,
                'is_stable': is_stable,
                'training_epochs': len(history.history['val_accuracy']),
                'best_epoch': np.argmax(history.history['val_accuracy']) + 1,
                'model_type': 'simplified_tcn_3_classes',
                'etiquetado': 'volatility_based_atr_3_classes',
                'arquitectura': 'tcn_v3_simplified'
            }
            
            with open(f'{model_dir}/hybrid_metrics.pkl', 'wb') as f:
                pickle.dump(hybrid_metrics, f)
            print("✅ Métricas de entrenamiento guardadas")

            print(f"\n🎯 MODELO TCN SIMPLIFICADO V3 COMPLETADO PARA {symbol}")
            print(f"📁 Guardado en: {model_dir}/")
            
            return True

        except Exception as e:
            print(f"❌ Error entrenando modelo híbrido para {symbol}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

async def main():
    """🎯 Entrenar modelos híbridos"""

    print("🎯 ENTRENADOR TCN HÍBRIDO (V3.2 - 3 CLASES, ROBUSTO)")
    print("=" * 80)
    print("🔄 Combinando: Etiquetado 3 Clases (ATR) + Horizonte 30min + 180 Días de Datos")
    print("🎯 Objetivo: Modelo compatible y robusto")
    print("🔧 Guardado como: definitivo_v3 (versión robusta 3 clases)")
    print("=" * 80)

    trainer = TCNHybridTrainer(prediction_horizon=30)

    # Entrenar solo un par para prueba
    symbol = "XRPUSDT"
    print(f"\n🚀 Entrenando {symbol} con modelo robusto...")
    
    success = await trainer.train_hybrid_model(symbol)
    
    if success:
        print(f"\n✅ {symbol}: ENTRENAMIENTO ROBUSTO V3 EXITOSO")
        print(f"🎯 Modelo guardado en models/definitivo_v3_{symbol.lower()}/")
    else:
        print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO ROBUSTO")

    # Opcionalmente entrenar todos los símbolos
    train_all = input("\n🤔 ¿Entrenar todos los símbolos? (y/n): ").lower().strip()
    
    if train_all == 'y':
        print("\n🚀 Entrenando todos los símbolos...")
        results = {}
        
        # Asumiendo que el primero ya se entrenó si no se saltó
        initial_symbol_trained = "XRPUSDT"
        results[initial_symbol_trained] = success

        for symbol in trainer.pairs:
            if symbol != initial_symbol_trained:
                print(f"\n🔄 Entrenando {symbol}...")
                success = await trainer.train_hybrid_model(symbol)
                results[symbol] = success

        print(f"\n🎯 RESUMEN FINAL:")
        print("=" * 50)
        for symbol, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"   {symbol}: {status}")

        successful = sum(results.values())
        print(f"\n🎯 Modelos robustos entrenados: {successful}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main()) 