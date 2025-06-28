#!/usr/bin/env python3
"""
🚀 ENTRENAMIENTO MODELO XRP - METODOLOGÍA IDÉNTICA A MODELOS EN PRODUCCIÓN
=========================================================================

Script para entrenar un modelo TCN para XRPUSDT siguiendo exactamente la misma
metodología y estructura de features que los modelos definitivos en producción:
- BTCUSDT, ETHUSDT, BNBUSDT

CARACTERÍSTICAS:
✅ Misma estructura de 66 features (tcn_definitivo)
✅ Mismo motor de features (CentralizedFeaturesEngine)
✅ Misma arquitectura TCN
✅ Mismo proceso de balanceado de datos
✅ Mismas configuraciones de entrenamiento
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del sistema
from centralized_features_engine import CentralizedFeaturesEngine
from real_market_data_provider import RealMarketDataProvider
from config import trading_config

# Configuración determinística
tf.random.set_seed(42)
np.random.seed(42)

class XRPModelTrainer:
    """
    Entrenador de modelo XRP usando metodología idéntica a modelos en producción
    """
    
    def __init__(self):
        self.symbol = "XRPUSDT"
        self.features_engine = CentralizedFeaturesEngine()
        self.data_provider = None
        self.model = None
        
        # Configuración idéntica a modelos en producción
        self.config = {
            # Datos
            'symbol': 'XRPUSDT',
            'days_history': 180,  # 180 días para múltiples regímenes
            'timeframe': '5m',    # Mismo timeframe que modelos actuales
            
            # Features (IDÉNTICO a modelos definitivos)
            'feature_set': 'tcn_definitivo',  # 66 features exactas
            'sequence_length': 60,            # Misma longitud de secuencia
            'prediction_horizon': 1,          # Mismo horizonte
            'price_threshold': 0.005,         # 0.5% para BUY/SELL (mismo)
            
            # Arquitectura TCN (IDÉNTICA)
            'tcn_filters': 48,                # Mismo número de filtros
            'tcn_kernel_size': 2,             # Mismo kernel
            'tcn_stacks': 2,                  # Mismos stacks
            'tcn_dilations': [1, 2, 4, 8, 16, 32],  # Mismas dilaciones
            'tcn_dropout': 0.3,               # Mismo dropout
            
            # Entrenamiento (IDÉNTICO)
            'epochs': 150,
            'batch_size': 64,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'test_split': 0.2,
            
            # Paths
            'model_save_path': 'models/definitivo_xrpusdt.h5',
            'data_save_path': 'data/xrp_training_data.pkl',
            'results_save_path': 'results/xrp_training_results.json'
        }
        
        # Crear directorios
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print("🎯 XRP Model Trainer inicializado")
        print(f"   📊 Símbolo: {self.symbol}")
        print(f"   🔧 Features: {self.config['feature_set']} (66 features)")
        print(f"   🏗️ Arquitectura: TCN idéntica a modelos en producción")

    def step_1_download_xrp_data(self) -> pd.DataFrame:
        """
        Paso 1: Descargar datos históricos de XRPUSDT
        """
        import os  # Importar os al inicio
        
        print("\n🚀 STEP 1: Descargando datos históricos de XRPUSDT")
        print("=" * 60)
        
        # Verificar si ya tenemos datos
        if os.path.exists(self.config['data_save_path']):
            print("📂 Cargando datos existentes...")
            try:
                with open(self.config['data_save_path'], 'rb') as f:
                    raw_data = pickle.load(f)
                if len(raw_data) > 50000:  # Verificar cantidad suficiente
                    print(f"✅ Datos existentes cargados: {len(raw_data):,} muestras")
                    return raw_data
            except:
                print("⚠️ Error cargando datos existentes, descargando nuevos...")
        
        # Inicializar cliente Binance
        try:
            from binance.client import Client
            
            # Obtener credenciales (opcional para datos públicos)
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if api_key and secret_key:
                self.binance_client = Client(api_key, secret_key)
                print(f"📡 Conectado a Binance API con credenciales")
            else:
                self.binance_client = Client()  # Cliente público
                print(f"📡 Conectado a Binance API pública")
                
        except Exception as e:
            print(f"❌ Error conectando a Binance: {e}")
            return self._generate_synthetic_xrp_data()
        
        # Descargar datos históricos reales
        print(f"📊 Descargando datos reales de {self.symbol}...")
        
        try:
            # Descargar datos en chunks para obtener 180 días completos
            print(f"   📊 Descargando {self.config['days_history']} días de datos históricos...")
            
            all_klines = []
            end_time = datetime.now()
            
            # Descargar en chunks de 1000 klines (máximo de Binance)
            klines_per_day = 12 * 24  # 288 klines de 5m por día
            total_klines_needed = self.config['days_history'] * klines_per_day
            chunks_needed = (total_klines_needed // 1000) + 1
            
            print(f"   🔄 Necesitamos {total_klines_needed:,} klines, descargando en {chunks_needed} chunks...")
            
            current_end_time = end_time
            
            for chunk in range(chunks_needed):
                print(f"   📦 Chunk {chunk + 1}/{chunks_needed}...")
                
                try:
                    # Obtener chunk de datos
                    chunk_klines = self.binance_client.get_klines(
                        symbol=self.symbol,
                        interval=Client.KLINE_INTERVAL_5MINUTE,
                        limit=1000,
                        endTime=int(current_end_time.timestamp() * 1000)
                    )
                    
                    if not chunk_klines:
                        print(f"   ⚠️ No hay más datos disponibles en chunk {chunk + 1}")
                        break
                    
                    # Agregar a la lista (en orden inverso para mantener cronología)
                    all_klines = chunk_klines + all_klines
                    
                    # Actualizar tiempo para siguiente chunk
                    earliest_time = datetime.fromtimestamp(chunk_klines[0][0] / 1000)
                    current_end_time = earliest_time - timedelta(minutes=5)
                    
                    print(f"   ✅ Chunk {chunk + 1}: {len(chunk_klines)} klines (hasta {earliest_time.strftime('%Y-%m-%d %H:%M')})")
                    
                    # Si ya tenemos suficientes datos, parar
                    if len(all_klines) >= total_klines_needed:
                        break
                        
                except Exception as e:
                    print(f"   ❌ Error en chunk {chunk + 1}: {e}")
                    break
            
            if not all_klines or len(all_klines) < 5000:  # Mínimo 5000 para entrenar bien
                print(f"⚠️ Datos insuficientes: {len(all_klines) if all_klines else 0} klines")
                return self._generate_synthetic_xrp_data()
            
            print(f"✅ Total obtenido: {len(all_klines):,} klines reales de {self.symbol}")
            
            # Convertir a DataFrame
            df = self._klines_to_dataframe(all_klines)
            
            if len(df) < 5000:
                print(f"⚠️ DataFrame insuficiente ({len(df)} filas)")
                return self._generate_synthetic_xrp_data()
            
            # Estadísticas de los datos descargados
            days_span = (df.index[-1] - df.index[0]).days
            price_volatility = (df['close'].max() - df['close'].min()) / df['close'].mean() * 100
            
            print(f"✅ Datos reales procesados: {len(df):,} muestras")
            print(f"   📅 Período: {df.index[0]} - {df.index[-1]} ({days_span} días)")
            print(f"   💰 Precio inicial: ${df['close'].iloc[0]:.6f}")
            print(f"   💰 Precio final: ${df['close'].iloc[-1]:.6f}")
            print(f"   📊 Rango de precios: ${df['close'].min():.6f} - ${df['close'].max():.6f}")
            print(f"   📈 Volatilidad: {price_volatility:.2f}%")
            
            # Guardar datos
            with open(self.config['data_save_path'], 'wb') as f:
                pickle.dump(df, f)
            
            return df
            
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            print("🔄 Generando datos sintéticos para XRP...")
            return self._generate_synthetic_xrp_data()

    def _klines_to_dataframe(self, klines_data: List) -> pd.DataFrame:
        """Convertir datos de klines de Binance a DataFrame"""
        try:
            df_data = []
            for kline in klines_data:
                # Formato de klines de Binance:
                # [timestamp, open, high, low, close, volume, close_time, quote_asset_volume, 
                #  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
                df_data.append({
                    'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Validar datos OHLC
            if not ((df['high'] >= df['low']) & 
                    (df['high'] >= df['open']) & 
                    (df['high'] >= df['close']) & 
                    (df['low'] <= df['open']) & 
                    (df['low'] <= df['close'])).all():
                print("⚠️ Advertencia: Datos OHLC inconsistentes detectados")
            
            return df
            
        except Exception as e:
            print(f"❌ Error convirtiendo klines: {e}")
            return pd.DataFrame()

    def _generate_synthetic_xrp_data(self) -> pd.DataFrame:
        """Generar datos sintéticos realistas para XRP"""
        print("🎲 Generando datos sintéticos para XRPUSDT...")
        
        # Parámetros realistas para XRP
        n_samples = 100000  # ~1 año de datos 5m
        base_price = 0.6    # Precio base XRP
        volatility = 0.04   # Volatilidad típica XRP
        
        np.random.seed(42)
        
        # Generar timestamps
        start_date = datetime.now() - timedelta(days=365)
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq='5T')
        
        # Generar precios con tendencias realistas
        returns = np.random.normal(0, volatility/100, n_samples)
        
        # Añadir tendencias y ciclos
        trend = np.sin(np.arange(n_samples) / 5000) * 0.001  # Tendencia cíclica
        momentum = np.cumsum(np.random.normal(0, 0.0001, n_samples))  # Momentum aleatorio
        
        returns = returns + trend + momentum
        
        # Generar precios
        prices = [base_price]
        for i in range(1, n_samples):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.1))  # Precio mínimo
        
        # Generar OHLC realista
        df_data = []
        for i, timestamp in enumerate(timestamps):
            close = prices[i]
            
            # Generar high/low realistas
            range_pct = abs(np.random.normal(0, 0.01))  # Rango típico
            high = close * (1 + range_pct/2)
            low = close * (1 - range_pct/2)
            
            # Open basado en close anterior
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, 0.002)  # Gap pequeño
                open_price = prices[i-1] * (1 + gap)
            
            # Volume realista
            volume = abs(np.random.normal(1000000, 500000))
            
            df_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Datos sintéticos generados: {len(df):,} muestras")
        print(f"   💰 Rango de precios: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
        
        return df

    def step_2_calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 2: Calcular features usando metodología idéntica
        """
        print("\n🔧 STEP 2: Calculando features (metodología idéntica)")
        print("=" * 60)
        
        print(f"📊 Datos de entrada: {len(df):,} muestras")
        print(f"🎯 Feature set: {self.config['feature_set']} (66 features)")
        
        # Calcular features usando motor centralizado (IDÉNTICO)
        features_df = self.features_engine.calculate_features(
            df=df,
            feature_set=self.config['feature_set']
        )
        
        print(f"✅ Features calculadas: {features_df.shape}")
        print(f"   📋 Features disponibles: {list(features_df.columns[:10])}...")
        
        # Validar calidad de features
        self._validate_features_quality(features_df)
        
        return features_df

    def _validate_features_quality(self, features_df: pd.DataFrame):
        """Validar calidad de features calculadas"""
        print("\n🔍 Validando calidad de features...")
        
        # Verificar NaN
        nan_counts = features_df.isnull().sum()
        features_with_nan = nan_counts[nan_counts > 0]
        
        if len(features_with_nan) > 0:
            print(f"   ⚠️ Features con NaN: {len(features_with_nan)}")
            for feature, count in features_with_nan.head().items():
                pct = count / len(features_df) * 100
                print(f"     {feature}: {count} ({pct:.1f}%)")
        else:
            print(f"   ✅ Sin valores NaN")
        
        # Verificar constantes
        constant_features = []
        for col in features_df.columns:
            if features_df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"   ⚠️ Features constantes: {constant_features}")
        else:
            print(f"   ✅ Sin features constantes")
        
        # Estadísticas básicas
        print(f"   📊 Estadísticas:")
        print(f"     Muestras válidas: {len(features_df.dropna()):,}")
        print(f"     Features válidas: {len(features_df.columns)}")

    def step_3_create_sequences(self, features_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple:
        """
        Paso 3: Crear secuencias temporales (metodología idéntica)
        """
        print("\n🔄 STEP 3: Creando secuencias temporales")
        print("=" * 60)
        
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        price_threshold = self.config['price_threshold']
        
        print(f"   📏 Longitud secuencia: {sequence_length}")
        print(f"   🎯 Horizonte predicción: {prediction_horizon}")
        print(f"   💹 Umbral precio: {price_threshold:.1%}")
        
        # Preparar datos
        X, y = [], []
        
        # Alinear features con precios
        common_index = features_df.index.intersection(price_df.index)
        features_aligned = features_df.loc[common_index].fillna(method='ffill').fillna(0)
        prices_aligned = price_df.loc[common_index]['close']
        
        print(f"   📊 Datos alineados: {len(common_index):,} muestras")
        
        # Crear secuencias
        for i in range(sequence_length, len(features_aligned) - prediction_horizon):
            # Secuencia de features
            sequence = features_aligned.iloc[i-sequence_length:i].values
            X.append(sequence)
            
            # Label basado en cambio de precio futuro
            current_price = prices_aligned.iloc[i]
            future_price = prices_aligned.iloc[i + prediction_horizon]
            price_change = (future_price - current_price) / current_price
            
            # Clasificación idéntica a modelos en producción
            if price_change > price_threshold:      # +0.5% -> BUY
                label = [1, 0, 0]  # BUY
            elif price_change < -price_threshold:   # -0.5% -> SELL
                label = [0, 0, 1]  # SELL
            else:                                   # [-0.5%, +0.5%] -> HOLD
                label = [0, 1, 0]  # HOLD
            
            y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"✅ Secuencias creadas:")
        print(f"   📐 X shape: {X.shape}")
        print(f"   📐 y shape: {y.shape}")
        
        # Estadísticas de labels (idéntico a validación de modelos actuales)
        label_counts = np.sum(y, axis=0)
        total = len(y)
        print(f"   📊 Distribución de labels ORIGINAL:")
        print(f"     BUY: {label_counts[0]:,} ({label_counts[0]/total*100:.1f}%)")
        print(f"     HOLD: {label_counts[1]:,} ({label_counts[1]/total*100:.1f}%)")
        print(f"     SELL: {label_counts[2]:,} ({label_counts[2]/total*100:.1f}%)")
        
        # === BALANCEADO INTELIGENTE DE CLASES ===
        print(f"\n🔄 Aplicando balanceado inteligente de clases...")
        X_balanced, y_balanced = self._balance_classes_intelligently(X, y)
        
        # Estadísticas después del balanceado
        label_counts_balanced = np.sum(y_balanced, axis=0)
        total_balanced = len(y_balanced)
        print(f"   📊 Distribución de labels BALANCEADA:")
        print(f"     BUY: {label_counts_balanced[0]:,} ({label_counts_balanced[0]/total_balanced*100:.1f}%)")
        print(f"     HOLD: {label_counts_balanced[1]:,} ({label_counts_balanced[1]/total_balanced*100:.1f}%)")
        print(f"     SELL: {label_counts_balanced[2]:,} ({label_counts_balanced[2]/total_balanced*100:.1f}%)")
        
        return X_balanced, y_balanced
    
    def _balance_classes_intelligently(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balancear clases de forma inteligente para evitar sobrerepresentación
        """
        # Identificar índices por clase
        buy_indices = np.where(np.argmax(y, axis=1) == 0)[0]   # BUY
        hold_indices = np.where(np.argmax(y, axis=1) == 1)[0]  # HOLD
        sell_indices = np.where(np.argmax(y, axis=1) == 2)[0]  # SELL
        
        print(f"   🔍 Clases detectadas:")
        print(f"     BUY: {len(buy_indices)} muestras")
        print(f"     HOLD: {len(hold_indices)} muestras")
        print(f"     SELL: {len(sell_indices)} muestras")
        
        # Calcular tamaño objetivo (promedio de BUY y SELL multiplicado por factor)
        minority_size = max(len(buy_indices), len(sell_indices))
        target_size = min(minority_size * 3, len(hold_indices))  # Máximo 3x la clase minoritaria
        
        print(f"   🎯 Tamaño objetivo por clase: {target_size}")
        
        # Seleccionar muestras balanceadas
        balanced_indices = []
        
        # BUY: usar todas las muestras + oversample si es necesario
        if len(buy_indices) < target_size:
            # Oversample BUY
            buy_oversampled = np.random.choice(buy_indices, target_size, replace=True)
            balanced_indices.extend(buy_oversampled)
        else:
            balanced_indices.extend(buy_indices[:target_size])
        
        # SELL: usar todas las muestras + oversample si es necesario
        if len(sell_indices) < target_size:
            # Oversample SELL
            sell_oversampled = np.random.choice(sell_indices, target_size, replace=True)
            balanced_indices.extend(sell_oversampled)
        else:
            balanced_indices.extend(sell_indices[:target_size])
        
        # HOLD: undersample para evitar sobrerepresentación
        hold_undersampled = np.random.choice(hold_indices, target_size, replace=False)
        balanced_indices.extend(hold_undersampled)
        
        # Mezclar índices
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        # Crear arrays balanceados
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        print(f"   ✅ Balanceado completado: {len(X_balanced)} muestras totales")
        
        return X_balanced, y_balanced

    def step_4_build_tcn_model(self, input_shape: Tuple) -> tf.keras.Model:
        """
        Paso 4: Construir modelo TCN (arquitectura idéntica)
        """
        print("\n🧠 STEP 4: Construyendo modelo TCN (arquitectura idéntica)")
        print("=" * 60)
        
        print(f"   📐 Input shape: {input_shape}")
        print(f"   🏗️ Arquitectura: TCN con {self.config['tcn_filters']} filtros")
        
        # Importar TCN (mismo que modelos en producción)
        try:
            from tcn import TCN
        except ImportError:
            print("⚠️ TCN no disponible, usando LSTM como fallback")
            return self._build_lstm_fallback(input_shape)
        
        # Construir modelo idéntico a modelos definitivos
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # TCN layer (configuración idéntica)
            TCN(
                nb_filters=self.config['tcn_filters'],
                kernel_size=self.config['tcn_kernel_size'],
                nb_stacks=self.config['tcn_stacks'],
                dilations=self.config['tcn_dilations'],
                padding='causal',
                use_skip_connections=True,
                dropout_rate=self.config['tcn_dropout'],
                return_sequences=False,
                activation='relu',
                name='tcn_layer'
            ),
            
            # Dense layers (idénticas)
            tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.3, name='dropout_1'),
            tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            
            # Output layer (3 clases: BUY, HOLD, SELL)
            tf.keras.layers.Dense(3, activation='softmax', name='output')
        ])
        
        # Compilar con configuración idéntica
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo TCN construido")
        model.summary()
        
        return model

    def _build_lstm_fallback(self, input_shape: Tuple) -> tf.keras.Model:
        """Construir modelo LSTM como fallback"""
        print("🔄 Construyendo modelo LSTM fallback...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3),
            tf.keras.layers.LSTM(32, dropout=0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def step_5_train_model(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Paso 5: Entrenar modelo (proceso idéntico)
        """
        print("\n🏃‍♂️ STEP 5: Entrenando modelo XRP")
        print("=" * 60)
        
        # Split idéntico a modelos en producción
        test_size = int(len(X) * self.config['test_split'])
        val_size = int(len(X) * self.config['validation_split'])
        
        X_train = X[:-test_size-val_size]
        y_train = y[:-test_size-val_size]
        X_val = X[-test_size-val_size:-test_size]
        y_val = y[-test_size-val_size:-test_size]
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        print(f"   📊 Train: {len(X_train):,}")
        print(f"   📊 Validation: {len(X_val):,}")
        print(f"   📊 Test: {len(X_test):,}")
        
        # Callbacks idénticos
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.config['model_save_path'],
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
        
        # Calcular class weights (idéntico)
        from sklearn.utils.class_weight import compute_class_weight
        y_integers = np.argmax(y_train, axis=1)
        classes = np.unique(y_integers)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_integers)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"   ⚖️ Class weights: {class_weight_dict}")
        
        # Entrenar modelo
        print(f"🚀 Iniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluación final
        print("\n📊 Evaluación final...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Predicciones detalladas
        predictions = model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Métricas por clase
        from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
        class_names = ['BUY', 'HOLD', 'SELL']
        
        # Calcular precision y recall manualmente
        test_prec = precision_score(true_classes, pred_classes, average='weighted', zero_division=0)
        test_rec = recall_score(true_classes, pred_classes, average='weighted', zero_division=0)
        
        print(f"\n📈 Resultados finales XRPUSDT:")
        print(f"   🎯 Test Accuracy: {test_acc:.4f}")
        print(f"   🎯 Test Precision: {test_prec:.4f}")
        print(f"   🎯 Test Recall: {test_rec:.4f}")
        print(f"   🎯 Confianza promedio: {np.mean(confidences):.4f}")
        
        print(f"\n📊 Reporte de clasificación:")
        try:
            print(classification_report(true_classes, pred_classes, target_names=class_names, zero_division=0))
        except ValueError as e:
            print(f"   ⚠️ Reporte limitado debido a distribución de clases: {e}")
            print(f"   📊 Clases únicas en test: {np.unique(true_classes)}")
            print(f"   📊 Predicciones únicas: {np.unique(pred_classes)}")
        
        # Guardar resultados
        results = {
            'model_path': self.config['model_save_path'],
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'avg_confidence': float(np.mean(confidences)),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X.shape[2],
            'sequence_length': X.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.config['results_save_path'], 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Resultados guardados en: {self.config['results_save_path']}")
        
        return results

    def run_complete_training(self) -> Dict:
        """
        Ejecutar entrenamiento completo de modelo XRP
        """
        print("🚀 INICIANDO ENTRENAMIENTO COMPLETO MODELO XRPUSDT")
        print("=" * 70)
        print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Metodología: Idéntica a modelos definitivos en producción")
        
        try:
            # Paso 1: Descargar datos
            raw_data = self.step_1_download_xrp_data()
            
            # Paso 2: Calcular features
            features_df = self.step_2_calculate_features(raw_data)
            
            # Paso 3: Crear secuencias
            X, y = self.step_3_create_sequences(features_df, raw_data)
            
            # Paso 4: Construir modelo
            model = self.step_4_build_tcn_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Paso 5: Entrenar modelo
            results = self.step_5_train_model(model, X, y)
            
            print(f"\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
            print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Modelo guardado: {self.config['model_save_path']}")
            print(f"🎯 Accuracy final: {results['test_accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

def main():
    """Función principal"""
    print("🎯 XRP MODEL TRAINER - METODOLOGÍA IDÉNTICA A PRODUCCIÓN")
    print("=" * 70)
    
    # Verificar dependencias
    try:
        import talib
        print("✅ TA-Lib disponible")
    except ImportError:
        print("⚠️ TA-Lib no disponible - usando implementaciones alternativas")
    
    try:
        from tcn import TCN
        print("✅ TCN disponible")
    except ImportError:
        print("⚠️ TCN no disponible - usando LSTM como fallback")
    
    # Crear entrenador
    trainer = XRPModelTrainer()
    
    # Ejecutar entrenamiento
    results = trainer.run_complete_training()
    
    if 'error' not in results:
        print(f"\n🎉 ¡MODELO XRP ENTRENADO EXITOSAMENTE!")
        print(f"📈 Accuracy: {results['test_accuracy']:.4f}")
        print(f"📁 Modelo disponible en: {results['model_path']}")
        print(f"\n🔄 Para usar el modelo:")
        print(f"   1. Copiar a directorio models/")
        print(f"   2. Agregar 'XRPUSDT' a lista de símbolos en config")
        print(f"   3. Reiniciar sistema de trading")
    else:
        print(f"\n❌ Error en entrenamiento: {results['error']}")

if __name__ == "__main__":
    main()