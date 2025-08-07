#!/usr/bin/env python3
"""
🎯 TCN TRAINER V3 REFACTORIZADO - ETIQUETADO RESPONSIVO Y PROBABLE
Versión refactorizada con configuración centralizada, sin "números mágicos" y
preparada para pruebas unitarias.
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
warnings.filterwarnings('ignore')

from centralized_features_engine2 import CentralizedFeaturesEngine


class RefactoredTCNTrainer:
    """
    🎯 Entrenador TCN V3 REFACTORIZADO.

    Esta clase mejora la v2 al:
    1. Centralizar todos los parámetros de configuración, eliminando "números mágicos".
    2. Utilizar el optimizador Adam moderno de Keras.
    3. Estar diseñada para ser fácilmente probable con pruebas unitarias.
    4. Mejorar la claridad del código y la documentación.
    """

    def __init__(self, config=None):
        # ✅ CONFIGURACIÓN CENTRALIZADA
        self.config = config or {}

        # Configuración general
        self.pairs = self.config.get('pairs', ["BTCUSDT"])
        self.timeframe = self.config.get('timeframe', '5m')
        self.lookback_window = self.config.get('lookback_window', 24)
        self.prediction_horizon = self.config.get('prediction_horizon', 6)
        self.days = self.config.get('days', 30)
        self.features_engine = CentralizedFeaturesEngine()

        # Configuración de etiquetado responsivo
        self.aggressiveness = self.config.get('aggressiveness', 'balanced')
        self.use_adaptive_thresholds = self.config.get('use_adaptive_thresholds', True)
        self.force_signals = self.config.get('force_signals', True)

        # ✅ PARÁMETROS DE LÓGICA DE ETIQUETADO (SIN NÚMEROS MÁGICOS)
        self.labeling_params = self.config.get('labeling_params', {
            'weak_signal_rsi_buffer': 5,
            'weak_signal_fallback_threshold': 0.002,
            'neutral_momentum_window': 2,
            'neutral_momentum_threshold': 0.005,
            'forced_signal_trend_window': 5,
            'forced_signal_trend_threshold': 0.003
        })
        
        # Actualizar buffer de RSI basado en agresividad
        self.aggressiveness_factors = {
            'conservative': {'factor': 0.7, 'rsi_buffer': 10},
            'balanced': {'factor': 1.0, 'rsi_buffer': 5},
            'aggressive': {'factor': 1.3, 'rsi_buffer': 2}
        }
        self.labeling_params['weak_signal_rsi_buffer'] = self.aggressiveness_factors[self.aggressiveness]['rsi_buffer']

        # Thresholds fijos (si no se usan adaptativos)
        self.fixed_thresholds = self.config.get('fixed_thresholds', {
            'BTCUSDT': {'strong_sell': -0.006, 'weak_sell': -0.003, 'weak_buy': 0.003, 'strong_buy': 0.006},
            'ETHUSDT': {'strong_sell': -0.008, 'weak_sell': -0.004, 'weak_buy': 0.004, 'strong_buy': 0.008},
            'BNBUSDT': {'strong_sell': -0.005, 'weak_sell': -0.0025, 'weak_buy': 0.0025, 'strong_buy': 0.005},
            'XRPUSDT': {'strong_sell': -0.010, 'weak_sell': -0.005, 'weak_buy': 0.005, 'strong_buy': 0.010},
            'DOTUSDT': {'strong_sell': -0.012, 'weak_sell': -0.006, 'weak_buy': 0.006, 'strong_buy': 0.012}
        })

        print("🎯 CONFIGURACIÓN REFACTORIZADA V3:")
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"   - {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     - {sub_key}: {sub_value}")
            else:
                print(f"   - {key}: {value}")

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """Calcula thresholds adaptativos basados en la volatilidad (ATR)."""
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]

        try:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            atr_14 = talib.ATR(high, low, close, timeperiod=14)
            avg_atr = np.nanmean(atr_14[-50:])
            avg_price = np.mean(close[-50:])
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02

            factor = self.aggressiveness_factors[self.aggressiveness]['factor']
            base_threshold = atr_percent * factor

            thresholds = {
                'strong_sell': -base_threshold * 1.0,
                'weak_sell': -base_threshold * 0.5,
                'weak_buy': base_threshold * 0.5,
                'strong_buy': base_threshold * 1.0
            }
            print(f"🎯 {symbol}: Thresholds adaptativos (ATR {atr_percent:.4f}): Buy {thresholds['strong_buy']:.4f}, Sell {thresholds['strong_sell']:.4f}")
            return thresholds
        except Exception as e:
            print(f"⚠️ Error en thresholds adaptativos: {e}. Usando fijos.")
            return self.fixed_thresholds[symbol]

    def create_responsive_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Etiquetado responsivo que favorece señales activas sobre HOLD."""
        print(f"🎯 Creando etiquetas RESPONSIVAS para {symbol} ({self.timeframe})...")

        close = df['close'].values
        thresholds = self.calculate_adaptive_thresholds(df, symbol)
        
        p = self.labeling_params
        rsi_buffer = p['weak_signal_rsi_buffer']

        labels = []
        for i in range(len(close) - self.prediction_horizon):
            future_return = (close[i + self.prediction_horizon] - close[i]) / close[i]
            label = 1  # Default to HOLD

            # Lógica de BUY/SELL fuerte
            if future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return <= thresholds['strong_sell']:
                label = 0  # SELL

            # Lógica de BUY/SELL débil (zona gris)
            elif future_return >= thresholds['weak_buy']:
                try:
                    if features['rsi_14'].iloc[i] < (30 + rsi_buffer):
                        label = 2  # BUY
                except (IndexError, KeyError):
                    if future_return > p['weak_signal_fallback_threshold']:
                         label = 2 # BUY
            elif future_return <= thresholds['weak_sell']:
                try:
                    if features['rsi_14'].iloc[i] > (70 - rsi_buffer):
                        label = 0  # SELL
                except (IndexError, KeyError):
                    if future_return < -p['weak_signal_fallback_threshold']:
                        label = 0 # SELL

            # Lógica para zona neutral (si aún es HOLD)
            else:
                if i >= p['neutral_momentum_window']:
                    momentum = (close[i] - close[i - p['neutral_momentum_window']]) / close[i - p['neutral_momentum_window']]
                    if momentum > p['neutral_momentum_threshold']:
                        label = 2  # BUY por momentum
                    elif momentum < -p['neutral_momentum_threshold']:
                        label = 0  # SELL por momentum

            # Forzar señal si está activado y sigue siendo HOLD
            if self.force_signals and label == 1 and i >= p['forced_signal_trend_window']:
                trend = (close[i] - close[i - p['forced_signal_trend_window']]) / close[i - p['forced_signal_trend_window']]
                if trend > p['forced_signal_trend_threshold']:
                    label = 2  # BUY por tendencia
                elif trend < -p['forced_signal_trend_threshold']:
                    label = 0  # SELL por tendencia
            
            labels.append(label)

        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Reporte de distribución
        label_counts = pd.Series(labels).value_counts(normalize=True).sort_index()
        print("📊 Distribución de etiquetas:")
        print(f"   - SELL (0): {label_counts.get(0, 0):.2%}")
        print(f"   - HOLD (1): {label_counts.get(1, 0):.2%}")
        print(f"   - BUY (2):  {label_counts.get(2, 0):.2%}")
        
        return df_labeled

    async def get_real_market_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Obtiene datos de mercado de Binance de forma asíncrona."""
        print(f"📊 Obteniendo {days} días de datos {self.timeframe} para {symbol}...")
        base_url = "https://api.binance.com/api/v3/klines"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_data = []
        async with aiohttp.ClientSession() as session:
            current_start = start_time
            while current_start < end_time:
                params = {'symbol': symbol, 'interval': self.timeframe, 'startTime': current_start, 'limit': 1000}
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        print(f"❌ Error API: {response.status}"); break
                    data = await response.json()
                    if not data: break
                    all_data.extend(data)
                    current_start = data[-1][6] + 1
                await asyncio.sleep(0.1)

        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(all_data, columns=cols)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        print(f"✅ Obtenidos {len(df)} registros de {symbol}")
        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """Prepara los datos para el entrenamiento del modelo TCN."""
        print("🔧 Preparando datos de entrenamiento...")
        features_aligned = features.iloc[:-self.prediction_horizon]
        
        # ✅ FIX PARA WINDOWS: Incluir TODAS las features calculadas, no solo float64/int64
        print(f"🔧 Seleccionando features para entrenamiento...")
        print(f"📊 Total de features disponibles: {len(features_aligned.columns)}")
        
        # ✅ INCLUIR TODAS LAS FEATURES CALCULADAS
        feature_columns = []
        for col in features_aligned.columns:
            # Convertir a numérico si es posible
            try:
                features_aligned[col] = pd.to_numeric(features_aligned[col], errors='coerce')
                if not features_aligned[col].isna().all():  # Si no son todos NaN
                    feature_columns.append(col)
            except:
                # Si no se puede convertir, intentar con valores booleanos
                if features_aligned[col].dtype == 'bool':
                    features_aligned[col] = features_aligned[col].astype(int)
                    feature_columns.append(col)
        
        print(f"📊 Features seleccionadas para entrenamiento: {len(feature_columns)}")
        print(f"📊 Features excluidas: {len(features_aligned.columns) - len(feature_columns)}")
        
        if len(feature_columns) != 66:
            print(f"⚠️ ADVERTENCIA: Se seleccionaron {len(feature_columns)} features en lugar de 66")
            missing_features = set(features_aligned.columns) - set(feature_columns)
            if missing_features:
                print(f"📋 Features excluidas: {missing_features}")

        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        X, y = [], []
        for i in range(self.lookback_window, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_window:i])
            y.append(df['label'].iloc[i])

        X, y = np.array(X), np.array(y)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        print(f"✅ Datos preparados: X shape: {X.shape}, y shape: {y.shape}")
        print(f"✅ Features finales: {len(feature_columns)}")
        return X, y, scaler, feature_columns, class_weight_dict

    def create_optimized_tcn_model(self, input_shape: tuple):
        """Crea un modelo TCN ligero y optimizado."""
        print("🎯 Creando modelo TCN LIGERO (<350k params)...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # ✅ OPTIMIZADOR ADAM MODERNO
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✅ Modelo creado con {model.count_params():,} parámetros.")
        return model

    async def train_model(self, symbol: str) -> bool:
        """Orquesta el proceso completo de entrenamiento para un símbolo."""
        print(f"\n{'='*70}\n🎯 ENTRENANDO MODELO V3 REFACTORIZADO PARA {symbol}\n{'='*70}")
        try:
            df = await self.get_real_market_data(symbol, self.days)
            
            # ✅ FIX PARA WINDOWS: Asegurar que se calculen las 66 features correctas
            print(f"🔧 Calculando features con engine centralizado...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
            
            # ✅ VERIFICACIÓN CRÍTICA: Asegurar que tengamos las 66 features
            expected_features = self.features_engine.feature_sets['tcn_definitivo']
            actual_features = list(features.columns)
            
            print(f"📊 Features esperadas: {len(expected_features)}")
            print(f"📊 Features calculadas: {len(actual_features)}")
            
            if len(actual_features) != len(expected_features):
                missing_features = set(expected_features) - set(actual_features)
                print(f"❌ ERROR: Faltan {len(missing_features)} features: {missing_features}")
                print(f"🔧 Intentando recalcular features...")
                
                # ✅ REINTENTO CON VERIFICACIÓN EXPLÍCITA
                features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
                actual_features = list(features.columns)
                
                if len(actual_features) != len(expected_features):
                    raise ValueError(f"CRÍTICO: No se pueden calcular las 66 features. Solo se obtuvieron {len(actual_features)}")
            
            print(f"✅ Features calculadas correctamente: {len(actual_features)}")
            
            if features.empty: raise ValueError("Cálculo de features falló.")
            
            df_labeled = self.create_responsive_labels(df, features, symbol)
            X, y, scaler, f_cols, c_weights = self.prepare_training_data(df_labeled, features)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y)

            model = self.create_optimized_tcn_model((X.shape[1], X.shape[2]))

            # ✅ DIRECTORIO DE MODELO V3
            model_dir = f'models/refactored_v3_{self.timeframe}_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)

            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
                tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.7, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(f'{model_dir}/best_model.h5', save_best_only=True, monitor='val_accuracy')
            ]

            print("🚀 Iniciando entrenamiento...")
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150,
                      batch_size=64, callbacks=callbacks, class_weight=c_weights, verbose=1)

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Accuracy final: {test_acc:.3f}")

            # Guardar todos los artefactos
            model.save(f'{model_dir}/model.h5')
            with open(f'{model_dir}/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f: pickle.dump(f_cols, f)
            with open(f'{model_dir}/config.pkl', 'wb') as f: pickle.dump(self.config, f)
            
            print(f"✅ Modelo V3 guardado en {model_dir}/")
            return True
        except Exception as e:
            print(f"❌ ERROR DURANTE EL ENTRENAMIENTO: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

def get_user_configuration():
    """Obtiene la configuración de entrenamiento de forma interactiva."""
    print("\n🎯 CONFIGURACIÓN PERSONALIZADA DEL ENTRENADOR V3")
    config = {}
    
    available_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT']
    symbol = input(f"🎯 Símbolo ({', '.join(available_symbols)}): ").upper().strip() or 'BTCUSDT'
    config['pairs'] = [symbol if symbol in available_symbols else 'BTCUSDT']

    available_timeframes = ['1m', '3m', '5m', '15m', '1h', '4h']
    timeframe = input(f"⏰ Timeframe ({', '.join(available_timeframes)}): ").lower().strip() or '5m'
    config['timeframe'] = timeframe if timeframe in available_timeframes else '5m'

    aggressiveness = input("🎯 Agresividad (conservative/balanced/aggressive): ").lower().strip() or 'balanced'
    config['aggressiveness'] = aggressiveness if aggressiveness in ['conservative', 'balanced', 'aggressive'] else 'balanced'

    try: config['days'] = int(input("📅 Días de datos (ej: 30): ").strip() or 30)
    except ValueError: config['days'] = 30

    if input("🔧 ¿Configurar parámetros avanzados? (s/n): ").lower().strip() == 's':
        try: config['lookback_window'] = int(input("   - Lookback window (ej: 24): ").strip() or 24)
        except ValueError: config['lookback_window'] = 24
        try: config['prediction_horizon'] = int(input("   - Prediction horizon (ej: 6): ").strip() or 6)
        except ValueError: config['prediction_horizon'] = 6
        config['force_signals'] = input("   - ¿Forzar señales? (s/n): ").lower().strip() == 's'
        config['use_adaptive_thresholds'] = input("   - ¿Thresholds adaptativos? (s/n): ").lower().strip() == 's'
    
    return config

async def main():
    """Función principal para ejecutar el entrenador."""
    print("🎯 TCN TRAINER V3 REFACTORIZADO")
    while True:
        config = get_user_configuration()
        trainer = RefactoredTCNTrainer(config)
        symbol = config['pairs'][0]
        success = await trainer.train_model(symbol)

        if success:
            print(f"\n✅ {symbol}: ENTRENAMIENTO V3 COMPLETADO EXITOSAMENTE.")
        else:
            print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO V3.")

        if input("\n🤔 ¿Entrenar otro modelo? (s/n): ").lower().strip() != 's':
            break
    print("\n🎉 ¡Proceso de entrenamiento finalizado!")

if __name__ == "__main__":
    # Para evitar problemas con asyncio en ciertos entornos
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
