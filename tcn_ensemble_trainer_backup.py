#!/usr/bin/env python3
"""
🎯 TCN ENSEMBLE TRAINER - SISTEMA DE ENSAMBLE AVANZADO CONFIGURABLE
Combina modelos de diferentes timeframes para generar predicciones más robustas y estables
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
import pickle
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Any
import talib
warnings.filterwarnings('ignore')

from centralized_features_engine2 import CentralizedFeaturesEngine


class TCNEnsembleTrainer:
    """🎯 Sistema de ensamble configurable que combina modelos de diferentes timeframes"""

    def __init__(self, config=None):
        # Configuración por de2fecto BALANCEADA + RENTABLE
        self.pairs = ["XRPUSDT", "ETHUSDT"]  # Agregar DOTUSDT
        self.features_engine = CentralizedFeaturesEngine()
        self.timeframe = "5m"
        self.days = 30  # ✅ RENTABLE: Más datos para mejor accuracy
        self.limit = 1000
        self.start_time = None
        self.end_time = None

        # ✅ CONFIGURACIÓN ENSEMBLE CONFIGURABLE (ARMONIZADA CON PREDICTOR)
        # Eliminadas claves duplicadas para consistencia
        self.timeframes = {
            '1m': {'lookback_window': 48, 'prediction_horizon': 12},  # Configuración estándar para 1m
            '5m': {'lookback_window': 48, 'prediction_horizon': 12},  # Configuración estándar para 5m
            '15m': {'lookback_window': 16, 'prediction_horizon': 8},
            '1h': {'lookback_window': 12, 'prediction_horizon': 6},
            '4h': {'lookback_window': 8, 'prediction_horizon': 4}     # 32 horas lookback, 16 horas horizon
        }

        # Aplicar configuración personalizada si se proporciona
        if config:
            self.pairs = [config.get('symbol', 'XRPUSDT')]
            self.timeframe = config.get('timeframe', '5m')
            self.days = config.get('days', 30)
            self.limit = config.get('limit', 1000)
            self.start_time = config.get('start_time')
            self.end_time = config.get('end_time')

            # Configurar timeframes personalizados
            if 'lookback_window' in config:
                self.timeframes[self.timeframe]['lookback_window'] = config['lookback_window']
            if 'prediction_horizon' in config:
                self.timeframes[self.timeframe]['prediction_horizon'] = config['prediction_horizon']

        # 🎯 THRESHOLDS RENTABLES - CONSIDERAN COSTOS DE TRADING
        # Costos totales: ~0.3% (comisiones 0.2% + spread 0.05% + slippage 0.05%)
        # Mínimo rentable: Costos + Margen = 0.3% + 0.5% = 0.8%
        self.thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.012,   # -1.2% para SELL fuerte (1 hora)
                'weak_sell': -0.006,     # -0.6% para SELL débil
                'weak_buy': 0.006,       # +0.6% para BUY débil
                'strong_buy': 0.012      # +1.2% para BUY fuerte (1 hora)
            },
            'ETHUSDT': {
                'strong_sell': -0.010,   # -1.5% (ETH más volátil)
                'weak_sell': -0.008,     # -0.8%
                'weak_buy': 0.008,       # +0.8%
                'strong_buy': 0.010      # +1.5%
            },
            'BNBUSDT': {
                'strong_sell': -0.012,   # -1.2%
                'weak_sell': -0.006,     # -0.6%
                'weak_buy': 0.006,       # +0.6%
                'strong_buy': 0.012      # +1.2%
            },
            'XRPUSDT': {
                'strong_sell': -0.018,   # -1.8% (XRP más volátil)
                'weak_sell': -0.009,     # -0.9%
                'weak_buy': 0.009,       # +0.9%
                'strong_buy': 0.018      # +1.8%
            },
            'DOTUSDT': {
                'strong_sell': -0.020,   # -2.0% (DOT más volátil, armonizado con predictor)
                'weak_sell': -0.010,     # -1.0%
                'weak_buy': 0.010,       # +1.0%
                'strong_buy': 0.020      # +2.0%
            }
        }

    async def get_market_data(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """📊 Obtener datos de mercado para el timeframe especificado"""

        print(f"📊 Obteniendo datos {timeframe} para {symbol}...")

        base_url = "https://api.binance.com"

        # Configurar fechas
        if self.end_time:
            end_time = int(self.end_time.timestamp() * 1000)
        else:
            end_time = int(datetime.now().timestamp() * 1000)

        if self.start_time:
            start_time = int(self.start_time.timestamp() * 1000)
        else:
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
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

        print(f"✅ Obtenidos {len(df)} registros {timeframe} de {symbol}")
        return df

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame,
                             symbol: str, timeframe: str) -> pd.DataFrame:
        """🎯 Crear etiquetas HÍBRIDAS usando ATR (volatilidad) - Mejor que percentiles"""

        print(f"🎯 Creando etiquetas HÍBRIDAS con ATR para {symbol} - {timeframe}...")

        df_copy = df.copy()

        # ✅ CORRECCIÓN: Obtener prediction_horizon con valores por defecto
        prediction_horizon = self.timeframes.get(timeframe, {}).get('prediction_horizon')
        if prediction_horizon is None:
            if timeframe == '1m':
                prediction_horizon = 6  # 30 minutos
            elif timeframe == '5m':
                prediction_horizon = 8  # 1 hora
            elif timeframe == '15m':
                prediction_horizon = 12   # 2 horas
            elif timeframe == '1h':
                prediction_horizon = 12  # 6 horas
            else:
                prediction_horizon = 12  # Default

        print(f"🔧 Usando prediction_horizon: {prediction_horizon} períodos")

        # ✅ ETIQUETADO HÍBRIDO INTELIGENTE - BASADO EN VOLATILIDAD (ATR)
        print(f"🔧 Método: ATR (volatilidad) + barreras dinámicas")

        # 1. Calcular ATR para volatilidad dinámica usando el motor centralizado
        atr_period = 24  # 2 horas en 5m
        atr_multiplier = 1.5  # Multiplicador conservador

        # ✅ CORRECCIÓN: Usar features del motor centralizado en lugar de cálculo manual
        if 'atr_14' in features.columns:
            df_copy['atr'] = features['atr_14']
        elif 'atr_20' in features.columns:
            df_copy['atr'] = features['atr_20']
        else:
            # Fallback: calcular ATR manualmente solo si no está disponible en features
            df_copy['atr'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=atr_period)

        # 2. Definir barreras dinámicas basadas en volatilidad
        df_copy['upper_barrier'] = df_copy['close'] + (df_copy['atr'] * atr_multiplier)
        df_copy['lower_barrier'] = df_copy['close'] - (df_copy['atr'] * atr_multiplier)

        # 3. Encontrar si alguna barrera es tocada en el futuro
        df_copy['future_max_price'] = df_copy['high'].shift(-prediction_horizon).rolling(window=prediction_horizon).max()
        df_copy['future_min_price'] = df_copy['low'].shift(-prediction_horizon).rolling(window=prediction_horizon).min()

        # 4. Limpiar NaNs generados por ATR y rolling windows
        df_copy.dropna(inplace=True)

        # 5. Aplicar lógica de etiquetado inteligente
        def get_label(row):
            touched_upper = row['future_max_price'] >= row['upper_barrier']
            touched_lower = row['future_min_price'] <= row['lower_barrier']

            if touched_upper and not touched_lower:
                return 2  # BUY - Solo barrera superior tocada
            elif touched_lower and not touched_upper:
                return 0  # SELL - Solo barrera inferior tocada
            else:
                # Si ninguna es tocada, o ambas lo son (indecisión), es HOLD
                return 1  # HOLD

        df_copy['label'] = df_copy.apply(get_label, axis=1)

        # 6. Filtros técnicos adicionales para mejorar calidad
        print(f"🔧 Aplicando filtros técnicos adicionales...")

        labels_filtered = []
        for i, row in df_copy.iterrows():
            candidate_label = row['label']

            # Obtener indicadores técnicos para confirmación
            try:
                idx_pos = df_copy.index.get_loc(i)
                if idx_pos < len(features):
                    current_rsi = features['rsi_14'].iloc[idx_pos] if 'rsi_14' in features.columns else 50
                    current_macd = features['macd_histogram'].iloc[idx_pos] if 'macd_histogram' in features.columns else 0
                else:
                    current_rsi = 50
                    current_macd = 0
            except:
                current_rsi = 50
                current_macd = 0

            # Filtros de confirmación técnica
            if candidate_label == 0:  # SELL candidato
                # Confirmar SELL con indicadores bajistas
                if current_rsi > 70 or (current_rsi > 60 and current_macd > 0):
                    label = 1  # HOLD si indicadores no confirman fuertemente
                else:
                    label = 0  # SELL confirmado
            elif candidate_label == 2:  # BUY candidato
                # Confirmar BUY con indicadores alcistas
                if current_rsi < 30 or (current_rsi < 40 and current_macd < 0):
                    label = 1  # HOLD si indicadores no confirman fuertemente
                else:
                    label = 2  # BUY confirmado
            else:
                label = 1  # HOLD mantenido

            labels_filtered.append(label)

        df_copy['label'] = labels_filtered

        # 7. Verificar distribución final
        label_counts = df_copy['label'].value_counts().sort_index()
        total = len(df_copy)

        print(f"💡 Parámetros ATR utilizados:")
        print(f"   - ATR período: {atr_period}")
        print(f"   - ATR multiplicador: {atr_multiplier}")
        print(f"   - Horizonte predicción: {prediction_horizon} períodos")

        print("📊 Distribución de etiquetas híbridas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")

        # 8. Validar balanceo
        max_class_pct = max([count/total for count in label_counts.values]) * 100
        min_class_pct = min([count/total for count in label_counts.values]) * 100
        balance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')

        if balance_ratio > 3.0:
            print(f"⚠️ ADVERTENCIA: Distribución desbalanceada (ratio: {balance_ratio:.1f})")
        else:
            print(f"✅ Distribución balanceada: ratio max/min = {balance_ratio:.1f}")

        # 9. Análisis de rentabilidad potencial
        self._analyze_atr_profitability(df_copy, symbol, atr_multiplier, prediction_horizon)

        # Limpiar columnas auxiliares
        return df_copy.drop(columns=['atr', 'upper_barrier', 'lower_barrier', 'future_max_price', 'future_min_price'])

    def _analyze_atr_profitability(self, df: pd.DataFrame, symbol: str, atr_multiplier: float, prediction_horizon: int):
        """💰 Análisis de rentabilidad potencial con método ATR"""
        try:
            print(f"\n💰 ANÁLISIS DE RENTABILIDAD ATR - {symbol}")
            print("=" * 60)

            close_prices = df['close'].values
            trading_costs = 0.003  # 0.3%

            profitable_buys = 0
            profitable_sells = 0
            total_buys = 0
            total_sells = 0
            total_profit_buys = 0.0
            total_profit_sells = 0.0

            for i, label in enumerate(df['label']):
                if i + prediction_horizon >= len(close_prices):
                    break

                current_price = close_prices[i]
                future_price = close_prices[i + prediction_horizon]
                gross_return = (future_price - current_price) / current_price

                if label == 2:  # BUY
                    total_buys += 1
                    net_profit = gross_return - trading_costs
                    total_profit_buys += net_profit
                    if net_profit > 0:
                        profitable_buys += 1

                elif label == 0:  # SELL
                    total_sells += 1
                    net_profit = -gross_return - trading_costs
                    total_profit_sells += net_profit
                    if net_profit > 0:
                        profitable_sells += 1

            # Calcular métricas
            buy_win_rate = (profitable_buys / total_buys * 100) if total_buys > 0 else 0
            sell_win_rate = (profitable_sells / total_sells * 100) if total_sells > 0 else 0
            avg_profit_buy = (total_profit_buys / total_buys * 100) if total_buys > 0 else 0
            avg_profit_sell = (total_profit_sells / total_sells * 100) if total_sells > 0 else 0

            print(f"📊 MÉTODO ATR:")
            print(f"   - ATR multiplicador: {atr_multiplier}")
            print(f"   - Horizonte: {prediction_horizon} períodos")

            print(f"📊 MÉTRICAS DE RENTABILIDAD:")
            print(f"   🟢 BUY Trades: {total_buys}")
            print(f"      Win Rate: {buy_win_rate:.1f}%")
            print(f"      Avg Profit: {avg_profit_buy:+.2f}%")
            print(f"   🔴 SELL Trades: {total_sells}")
            print(f"      Win Rate: {sell_win_rate:.1f}%")
            print(f"      Avg Profit: {avg_profit_sell:+.2f}%")

            # Evaluación general
            overall_win_rate = ((profitable_buys + profitable_sells) / (total_buys + total_sells) * 100) if (total_buys + total_sells) > 0 else 0
            total_profit = total_profit_buys + total_profit_sells

            print(f"   📈 RESUMEN GENERAL:")
            print(f"      Win Rate Total: {overall_win_rate:.1f}%")
            print(f"      Profit Total: {total_profit:+.4f}")

            # Validación mejorada
            if overall_win_rate >= 70 and total_profit > 0.02:
                print(f"   ✅ MODELO ATR ALTAMENTE RENTABLE")
            elif overall_win_rate >= 60 and total_profit > 0.01:
                print(f"   ✅ MODELO ATR RENTABLE")
            elif overall_win_rate >= 50:
                print(f"   ⚠️ MODELO ATR EN EL LÍMITE")
            else:
                print(f"   ❌ MODELO ATR NECESITA AJUSTES")

            print("=" * 60)

        except Exception as e:
            print(f"❌ Error en análisis ATR: {e}")

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame,
                            timeframe: str) -> tuple:
        """🔧 Preparar datos para entrenamiento por timeframe (OPTIMIZADO + CORREGIDO)"""

        print(f"🔧 Preparando datos para {timeframe}...")

        # ✅ CORRECCIÓN: Asegurar que lookback_window no sea None
        lookback_window = self.timeframes.get(timeframe, {}).get('lookback_window')
        prediction_horizon = self.timeframes.get(timeframe, {}).get('prediction_horizon')

        # Valores por defecto si no están definidos
        if lookback_window is None:
            if timeframe == '1m':
                lookback_window = 60  # 1 hora en 1m
            elif timeframe == '5m':
                lookback_window = 24  # 2 horas en 5m
            elif timeframe == '15m':
                lookback_window = 16  # 4 horas en 15m
            elif timeframe == '1h':
                lookback_window = 12  # 12 horas en 1h
            else:
                lookback_window = 24  # Default

        if prediction_horizon is None:
            if timeframe == '1m':
                prediction_horizon = 30  # 30 minutos
            elif timeframe == '5m':
                prediction_horizon = 12  # 1 hora
            elif timeframe == '15m':
                prediction_horizon = 8   # 2 horas
            elif timeframe == '1h':
                prediction_horizon = 6   # 6 horas
            else:
                prediction_horizon = 12  # Default

        print(f"📊 Configuración (con corrección):")
        print(f"   - Lookback window: {lookback_window}")
        print(f"   - Prediction horizon: {prediction_horizon}")

        # ✅ CORRECCIÓN: Alinear correctamente features con labels
        # El df ya viene filtrado del etiquetado ATR, así que necesitamos alinear features

        # 1. Obtener índices comunes entre df y features
        common_indices = df.index.intersection(features.index)

        # 2. Filtrar ambos DataFrames con índices comunes
        df_aligned = df.loc[common_indices].copy()
        features_aligned = features.loc[common_indices].copy()

        print(f"✅ Datos alineados:")
        print(f"   - DF original: {len(df)} registros")
        print(f"   - Features original: {len(features)} registros")
        print(f"   - Datos alineados: {len(df_aligned)} registros")

        # 3. Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns
                         if features_aligned[col].dtype in ['float64', 'int64']]
        print(f"✅ Features numéricas seleccionadas: {len(feature_columns)}")

        # 4. Normalizar features
        print(f"📊 Normalizando features...")
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])
        print(f"✅ Features normalizadas")

        # 5. Crear secuencias temporales (CORREGIDO)
        print(f"🔄 Creando secuencias temporales...")
        X = []
        y = []

        # ✅ CORRECCIÓN: Verificar que tenemos suficientes datos
        max_sequences = len(features_scaled) - lookback_window

        if max_sequences <= 0:
            raise ValueError(f"No hay suficientes datos. Necesita al menos {lookback_window + 1} registros, tiene {len(features_scaled)}")

        print(f"📊 Secuencias disponibles: {max_sequences}")

        # Procesar en chunks para mejor rendimiento
        chunk_size = 1000
        for i in range(0, max_sequences, chunk_size):
            end_idx = min(i + chunk_size, max_sequences)

            for j in range(i, end_idx):
                # Verificar que el índice está dentro del rango
                if j + lookback_window < len(df_aligned):
                    sequence = features_scaled[j:j+lookback_window]
                    X.append(sequence)
                    # Usar .iloc con índice verificado
                    y.append(df_aligned['label'].iloc[j+lookback_window])

            # Mostrar progreso
            if i % 5000 == 0:
                print(f"   📈 Secuencias creadas: {len(X)}/{max_sequences}")

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos {timeframe} preparados CORRECTAMENTE:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features: {len(feature_columns)}")
        print(f"   - Lookback window: {lookback_window}")
        print(f"   - Secuencias válidas: {len(X)}")

        # 6. Verificar que tenemos datos válidos
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No se generaron secuencias válidas. Verificar datos de entrada.")

        # 7. Calcular class weights
        print(f"⚖️ Calculando class weights...")
        unique_classes = np.unique(y)
        print(f"   - Clases encontradas: {unique_classes}")

        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"✅ Class weights calculados: {class_weight_dict}")

        return X, y, scaler, feature_columns, class_weight_dict

    def create_tcn_model(self, input_shape: tuple, timeframe: str):
        """🎯 Crear modelo TCN HÍBRIDO SIMPLIFICADO - Arquitectura probada y efectiva"""

        print(f"🎯 Creando modelo TCN HÍBRIDO SIMPLIFICADO para {timeframe}...")

        # ✅ ARQUITECTURA HÍBRIDA SIMPLIFICADA - PROBADA Y EFECTIVA
        # Usar la misma arquitectura para todos los timeframes (más consistente)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),

            # ✅ BLOQUES TCN SIMPLIFICADOS - PROBADOS
            # Progresión: 32 → 64 → 128 → 64
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

            # ✅ POOLING GLOBAL
            tf.keras.layers.GlobalAveragePooling1D(),

            # ✅ CAPAS DENSAS SIMPLIFICADAS CON REGULARIZACIÓN L2
            tf.keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            # ✅ CAPA DE SALIDA PARA 3 CLASES
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # ✅ CONFIGURACIÓN HÍBRIDA OPTIMIZADA
        # Learning rate conservador como en el híbrido original
        learning_rate = 0.0005  # Conservador y probado

        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo {timeframe} HÍBRIDO SIMPLIFICADO creado: {model.count_params():,} parámetros")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Regularización L2: 0.001")
        print(f"   - Dropout progresivo: 0.1-0.3")
        print(f"   - Arquitectura: HÍBRIDA PROBADA")
        return model

    async def train_ensemble_models(self, symbol: str) -> bool:
        """🎯 Entrenar modelos de ensamble para un símbolo (OPTIMIZADO)"""

        print(f"\n🎯 ENTRENANDO ENSEMBLE PARA {symbol}")
        print("=" * 70)

        results = {}

        # Solo entrenar el timeframe configurado
        timeframe = self.timeframe
        print(f"\n🔄 Entrenando modelo {timeframe}...")

        try:
            # 1. Obtener datos
            print(f"📊 PASO 1: Obteniendo datos de mercado...")
            df = await self.get_market_data(symbol, timeframe, days=self.days)

            # 2. Crear features
            print(f"🔧 PASO 2: Calculando features...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
            if features.empty:
                print(f"❌ Error calculando features {timeframe}")
                return False
            print(f"✅ Features calculadas exitosamente")

            # 3. Crear etiquetas
            print(f"🏷️ PASO 3: Creando etiquetas...")
            df_labeled = self.create_balanced_labels(df, features, symbol, timeframe)

            # 4. Preparar datos
            print(f"📊 PASO 4: Preparando datos de entrenamiento...")
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(
                df_labeled, features, timeframe
            )

            # 5. Split
            print(f"✂️ PASO 5: Dividiendo datos...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"✅ Datos divididos: Train={len(X_train)}, Test={len(X_test)}")

            # 6. Crear modelo
            print(f"🎯 PASO 6: Creando modelo TCN...")
            model = self.create_tcn_model((X.shape[1], X.shape[2]), timeframe)

            # 7. Directorio de modelos ARMONIZADO con predictor
            if timeframe == '1m':
                model_dir = f'models/definitivo_v3_{symbol.lower()}'
            else:  # 5m y otros timeframes
                model_dir = f'models/definitivo_v3_{timeframe}_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)
            print(f"📁 Directorio creado: {model_dir} (ARMONIZADO)")

            # 8. Callbacks HÍBRIDOS SIMPLIFICADOS
            print(f"⚙️ PASO 7: Configurando callbacks HÍBRIDOS...")

            # ✅ CALLBACKS SIMPLIFICADOS COMO EN EL HÍBRIDO ORIGINAL
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=15,  # Paciente pero no excesivo
                    restore_best_weights=True,
                    monitor='val_accuracy',
                    verbose=1
                ),

                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=8,   # Respuesta más rápida
                    factor=0.7,   # Reducción moderada
                    monitor='val_loss',
                    verbose=1
                ),

                tf.keras.callbacks.ModelCheckpoint(
                    f'{model_dir}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    verbose=1
                ),

                # Callback para detectar NaN
                tf.keras.callbacks.TerminateOnNaN()
            ]

            # 9. Entrenar con configuración HÍBRIDA SIMPLIFICADA
            print(f"🚀 PASO 8: Iniciando entrenamiento HÍBRIDO...")
            print(f"📊 Configuración HÍBRIDA:")
            print(f"   - Epochs: 100 (probado)")
            print(f"   - Batch size: 64 (estable)")
            print(f"   - Early stopping: patience=15")
            print(f"   - Reduce LR: patience=8")
            print(f"   - Learning rate: 0.0005 (conservador)")
            print(f"   - Etiquetado: ATR dinámico")
            print(f"   - Arquitectura: Híbrida simplificada")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,  # Como en el híbrido original
                batch_size=64,  # Tamaño estable
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                shuffle=True
            )

            # 10. Evaluar
            print(f"📊 PASO 9: Evaluando modelo...")
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ Resultados {timeframe}:")
            print(f"   - Accuracy: {test_acc:.3f}")
            print(f"   - Loss: {test_loss:.3f}")

            # 11. Guardar componentes
            print(f"💾 PASO 10: Guardando componentes del modelo...")
            model.save(f'{model_dir}/model.h5')
            print(f"   ✅ Modelo guardado")

            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            print(f"   ✅ Scaler guardado")

            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)
            print(f"   ✅ Feature columns guardadas")

            with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)
            print(f"   ✅ Class weights guardados")

            # Guardar métricas del ensemble
            print(f"📊 PASO 11: Guardando métricas...")
            ensemble_metrics = {
                'timeframe': timeframe,
                'accuracy': test_acc,
                'loss': test_loss,
                'symbol': symbol,
                'lookback_window': self.timeframes[timeframe]['lookback_window'],
                'prediction_horizon': self.timeframes[timeframe]['prediction_horizon'],
                'features_count': len(feature_columns),
                'training_days': self.days,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None
            }

            with open(f'{model_dir}/ensemble_metrics.pkl', 'wb') as f:
                pickle.dump(ensemble_metrics, f)
            print(f"   ✅ Ensemble metrics guardadas")

            results[timeframe] = True
            print(f"✅ Modelo {timeframe} guardado en {model_dir}/")

        except Exception as e:
            print(f"❌ Error entrenando {timeframe}: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Resumen final
        print(f"\n🎯 RESUMEN ENSEMBLE {symbol}:")
        print("=" * 50)
        for timeframe, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"   {timeframe}: {status}")

        successful = sum(results.values())
        print(f"\n🏆 Modelos exitosos: {successful}/1")

        return successful == 1


def get_user_configuration():
    """🎯 Obtener configuración del usuario desde consola"""

    print("\n🎯 CONFIGURACIÓN DEL ENSEMBLE TRAINER")
    print("=" * 50)

    # 1. Símbolo
    available_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
    print(f"\n📊 Símbolos disponibles: {', '.join(available_symbols)}")
    symbol = input("🎯 Ingresa el símbolo (ej: BTCUSDT): ").upper().strip()
    if symbol not in available_symbols:
        print(f"⚠️ Símbolo no válido, usando XRPUSDT")
        symbol = 'XRPUSDT'

    # 2. Timeframe
    available_timeframes = ['1m', '5m', '15m', '1h', '4h']
    print(f"\n⏰ Timeframes disponibles: {', '.join(available_timeframes)}")
    timeframe = input("⏰ Ingresa el timeframe (ej: 5m): ").lower().strip()
    if timeframe not in available_timeframes:
        print(f"⚠️ Timeframe no válido, usando 5m")
        timeframe = '5m'

    # 3. Días de entrenamiento
    print(f"\n📅 Configuración de datos:")
    days_option = input("📅 ¿Usar días específicos o fechas específicas? (días/fechas): ").lower().strip()

    if days_option == 'fechas':
        # Fechas específicas
        print("📅 Ingresa fechas en formato YYYY-MM-DD")
        start_date = input("📅 Fecha inicio (ej: 2024-01-01): ").strip()
        end_date = input("📅 Fecha fin (ej: 2024-12-31): ").strip()

        try:
            start_time = datetime.strptime(start_date, '%Y-%m-%d')
            end_time = datetime.strptime(end_date, '%Y-%m-%d')
            days = None
        except ValueError:
            print("⚠️ Formato de fecha incorrecto, usando 30 días")
            days = 30
            start_time = None
            end_time = None
    else:
        # Días específicos
        try:
            days = int(input("📅 Número de días (ej: 30): ").strip())
            if days <= 0 or days > 365:
                print("⚠️ Días no válidos, usando 30")
                days = 30
        except ValueError:
            print("⚠️ Días no válidos, usando 30")
            days = 30
        start_time = None
        end_time = None

    # 4. Configuración avanzada
    print(f"\n🔧 Configuración avanzada:")
    advanced = input("🔧 ¿Configurar parámetros avanzados? (s/n): ").lower().strip()

    lookback_window = None
    prediction_horizon = None

    if advanced == 's':
        try:
            lookback_window = int(input("🔧 Lookback window (ej: 24): ").strip())
            if lookback_window <= 0:
                print("⚠️ Lookback window no válido, usando default")
                lookback_window = None
        except ValueError:
            print("⚠️ Lookback window no válido, usando default")
            lookback_window = None

        try:
            prediction_horizon = int(input("🔧 Prediction horizon (ej: 12): ").strip())
            if prediction_horizon <= 0:
                print("⚠️ Prediction horizon no válido, usando default")
                prediction_horizon = None
        except ValueError:
            print("⚠️ Prediction horizon no válido, usando default")
            prediction_horizon = None

    # Crear configuración
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'start_time': start_time,
        'end_time': end_time,
        'lookback_window': lookback_window,
        'prediction_horizon': prediction_horizon
    }

    print(f"\n✅ CONFIGURACIÓN FINAL:")
    print(f"   📊 Símbolo: {symbol}")
    print(f"   ⏰ Timeframe: {timeframe}")
    if days:
        print(f"   📅 Días: {days}")
    else:
        print(f"   📅 Fechas: {start_time.date()} a {end_time.date()}")
    if lookback_window:
        print(f"   🔧 Lookback window: {lookback_window}")
    if prediction_horizon:
        print(f"   🔧 Prediction horizon: {prediction_horizon}")

    return config


def get_optimized_configuration():
    """🎯 Configuraciones optimizadas predefinidas"""

    print("\n🎯 CONFIGURACIONES OPTIMIZADAS")
    print("=" * 50)
    print("1. 🚀 Configuración Rápida (5m, 30 días)")
    print("2. 📊 Configuración Estándar (5m, 60 días)")
    print("3. 📈 Configuración Extensa (5m, 90 días)")
    print("4. ⚡ Configuración Alta Frecuencia (1m, 30 días)")
    print("5. 🎯 Configuración Personalizada")

    choice = input("\n🎯 Selecciona configuración (1-5): ").strip()

    if choice == '1':
        return {
            'symbol': 'XRPUSDT',
            'timeframe': '5m',
            'days': 30,
            'start_time': None,
            'end_time': None
        }
    elif choice == '2':
        return {
            'symbol': 'XRPUSDT',
            'timeframe': '5m',
            'days': 60,
            'start_time': None,
            'end_time': None
        }
    elif choice == '3':
        return {
            'symbol': 'XRPUSDT',
            'timeframe': '5m',
            'days': 90,
            'start_time': None,
            'end_time': None
        }
    elif choice == '4':
        return {
            'symbol': 'XRPUSDT',
            'timeframe': '1m',
            'days': 30,
            'start_time': None,
            'end_time': None
        }
    else:
        return get_user_configuration()


async def main():
    """🎯 Entrenar sistema de ensamble configurable"""

    print("🎯 TCN ENSEMBLE TRAINER - CONFIGURABLE")
    print("=" * 80)
    print("🔄 Sistema de entrenamiento configurable para modelos TCN")
    print("🎯 Características:")
    print("   ✅ Timeframes configurables (1m, 5m, 15m, 1h, 4h)")
    print("   ✅ Días de entrenamiento personalizables")
    print("   ✅ Fechas específicas de inicio/fin")
    print("   ✅ Lookback window y prediction horizon configurables")
    print("   ✅ Nombres de modelo con timeframe incluido")
    print("=" * 80)

    # Seleccionar tipo de configuración
    print("\n🎯 TIPO DE CONFIGURACIÓN:")
    print("1. 🚀 Configuración Optimizada (Recomendado)")
    print("2. ⚙️ Configuración Personalizada")

    config_type = input("\n🎯 Selecciona tipo (1-2): ").strip()

    if config_type == '1':
        config = get_optimized_configuration()
    else:
        config = get_user_configuration()

    # Crear trainer con configuración
    trainer = TCNEnsembleTrainer(config)

    # Entrenar modelo
    symbol = config['symbol']
    timeframe = config['timeframe']

    print(f"\n🚀 Entrenando modelo {timeframe} para {symbol}...")
    print(f"📊 Configuración:")
    print(f"   - Símbolo: {symbol}")
    print(f"   - Timeframe: {timeframe}")
    if config['days']:
        print(f"   - Días: {config['days']}")
    else:
        print(f"   - Fechas: {config['start_time'].date()} a {config['end_time'].date()}")
    if config.get('lookback_window'):
        print(f"   - Lookback window: {config['lookback_window']}")
    if config.get('prediction_horizon'):
        print(f"   - Prediction horizon: {config['prediction_horizon']}")

    success = await trainer.train_ensemble_models(symbol)

    if success:
        print(f"\n✅ {symbol}: MODELO {timeframe} COMPLETADO EXITOSAMENTE")
        print(f"🎯 Modelo guardado en: models/definitivo_{timeframe}_{symbol.lower()}/")
        print(f"📁 Archivos incluidos:")
        print(f"   - best_model.h5 (modelo entrenado)")
        print(f"   - scaler.pkl (normalización)")
        print(f"   - feature_columns.pkl (features)")
        print(f"   - class_weights.pkl (pesos de clases)")
        print(f"   - ensemble_metrics.pkl (métricas)")
    else:
        print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO")

    # Preguntar si entrenar más modelos
    train_more = input("\n🤔 ¿Entrenar otro modelo? (s/n): ").lower().strip()

    while train_more == 's':
        config = get_user_configuration()
        trainer = TCNEnsembleTrainer(config)
        symbol = config['symbol']
        timeframe = config['timeframe']

        print(f"\n🚀 Entrenando modelo {timeframe} para {symbol}...")
        success = await trainer.train_ensemble_models(symbol)

        if success:
            print(f"\n✅ {symbol}: MODELO {timeframe} COMPLETADO")
            print(f"🎯 Guardado en: models/definitivo_{timeframe}_{symbol.lower()}/")
        else:
            print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO")

        train_more = input("\n🤔 ¿Entrenar otro modelo? (s/n): ").lower().strip()

    print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print(f"🎯 Revisa los modelos en el directorio models/")


if __name__ == "__main__":
    asyncio.run(main())
