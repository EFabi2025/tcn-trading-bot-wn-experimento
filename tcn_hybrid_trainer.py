#!/usr/bin/env python3
"""
🎯 SISTEMA PROFESIONAL DE TRADING DE CRIPTOMONEDAS
Versión sin look-ahead bias, validación temporal correcta y métricas reales de trading
INTEGRADO CON MOTOR CENTRALIZADO DE FEATURES
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import talib
import warnings
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging

# Importar motor centralizado de features
from centralized_features_engine2 import CentralizedFeaturesEngine

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Configuración profesional de trading con opciones expandidas"""

    # === CONFIGURACIÓN BÁSICA ===
    symbol: str = "BTCUSDT"

    # === CONFIGURACIÓN TEMPORAL ===
    timeframe: str = "5m"  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    lookback_periods: int = 48  # Ventana de datos históricos para predicción
    prediction_horizon: int = 12  # Períodos hacia adelante a predecir

    # === CONFIGURACIÓN DE DATOS ===
    training_days: Optional[int] = 90  # Días de entrenamiento (si None, usar fechas específicas)
    start_date: Optional[str] = None  # Formato: "2024-01-01" (sobrescribe training_days)
    end_date: Optional[str] = None    # Formato: "2024-12-31" (si None, usar fecha actual)

    # === CONFIGURACIÓN DEL MODELO ===
    model_type: str = "tcn_advanced"  # tcn_basic, tcn_advanced, lstm_tcn, transformer_tcn
    feature_set: str = "tcn_definitivo"  # tcn_definitivo, tcn_final, full_set

    # === CONFIGURACIÓN DE ESCALADO ===
    scaler_type: str = "robust"  # robust, standard, minmax

    # === CONFIGURACIÓN DE ENTRENAMIENTO ===
    test_size: float = 0.2  # Proporción para test temporal
    validation_size: float = 0.15  # Proporción para validación
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0005
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 10

    # === COSTOS DE TRADING ===
    commission_rate: float = 0.001  # 0.1% comisión
    spread_cost: float = 0.0005     # 0.05% spread
    slippage_cost: float = 0.0005   # 0.05% slippage

    # === CONFIGURACIÓN DE SEÑALES ===
    buy_threshold: float = 0.7  # Umbral de confianza para BUY
    sell_threshold: float = 0.7  # Umbral de confianza para SELL
    min_signal_strength: int = 4  # Mínimo número de condiciones para señal

    # === TIMEFRAMES VÁLIDOS ===
    VALID_TIMEFRAMES: List[str] = field(default_factory=lambda: [
        "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    ])

    # === MODELOS DISPONIBLES ===
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: [
        "tcn_basic", "tcn_advanced", "lstm_tcn", "transformer_tcn"
    ])

    # === FEATURE SETS DISPONIBLES ===
    AVAILABLE_FEATURE_SETS: List[str] = field(default_factory=lambda: [
        "tcn_definitivo", "tcn_final", "full_set"
    ])

    # === ESCALADORES DISPONIBLES ===
    AVAILABLE_SCALERS: List[str] = field(default_factory=lambda: [
        "robust", "standard", "minmax"
    ])

    def __post_init__(self):
        """Validación post-inicialización"""
        self.validate_config()

    def validate_config(self):
        """Validar configuración"""
        # Validar timeframe
        if self.timeframe not in self.VALID_TIMEFRAMES:
            raise ValueError(f"Timeframe '{self.timeframe}' no válido. Opciones: {self.VALID_TIMEFRAMES}")

        # Validar modelo
        if self.model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modelo '{self.model_type}' no disponible. Opciones: {self.AVAILABLE_MODELS}")

        # Validar feature set
        if self.feature_set not in self.AVAILABLE_FEATURE_SETS:
            raise ValueError(f"Feature set '{self.feature_set}' no disponible. Opciones: {self.AVAILABLE_FEATURE_SETS}")

        # Validar escalador
        if self.scaler_type not in self.AVAILABLE_SCALERS:
            raise ValueError(f"Escalador '{self.scaler_type}' no disponible. Opciones: {self.AVAILABLE_SCALERS}")

        # Validar fechas si están especificadas
        if self.start_date:
            try:
                datetime.strptime(self.start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date debe tener formato YYYY-MM-DD")

        if self.end_date:
            try:
                datetime.strptime(self.end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end_date debe tener formato YYYY-MM-DD")

        # Validar rangos
        if not 0 < self.test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size debe estar entre 0 y 1")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size debe ser < 1")

        logger.info("✅ Configuración validada correctamente")

    @property
    def total_trading_cost(self) -> float:
        """Costo total de trading"""
        return self.commission_rate + self.spread_cost + self.slippage_cost

    @property
    def min_profitable_move(self) -> float:
        """Movimiento mínimo rentable"""
        return self.total_trading_cost * 2.5  # 250% margin de seguridad

    @property
    def timeframe_minutes(self) -> int:
        """Convertir timeframe a minutos"""
        timeframe_map = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
            "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
        }
        return timeframe_map.get(self.timeframe, 5)

    def get_training_period_info(self) -> Dict:
        """Obtener información del período de entrenamiento"""
        if self.start_date and self.end_date:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            days = (end - start).days
            return {
                "mode": "dates",
                "start_date": self.start_date,
                "end_date": self.end_date,
                "total_days": days
            }
        elif self.start_date:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.now()
            days = (end - start).days
            return {
                "mode": "start_to_now",
                "start_date": self.start_date,
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "total_days": days
            }
        else:
            end = datetime.now()
            start = end - timedelta(days=self.training_days)
            return {
                "mode": "days_back",
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "total_days": self.training_days
            }

    def to_dict(self) -> Dict:
        """Convertir configuración a diccionario"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not key.isupper()
        }


class ProfessionalCryptoTrader:
    """Sistema profesional de trading de criptomonedas con motor centralizado de features"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.scaler = None
        self.model = None
        self.feature_columns = []

        # Inicializar motor centralizado de features
        self.features_engine = CentralizedFeaturesEngine()

        # Inicializar escalador según configuración
        self.scaler = self._create_scaler()

        # Mostrar información de inicialización
        logger.info(f"🎯 Inicializando trader profesional")
        logger.info(f"   📈 Símbolo: {config.symbol}")
        logger.info(f"   ⏰ Timeframe: {config.timeframe} ({config.timeframe_minutes} min)")
        logger.info(f"   🧠 Modelo: {config.model_type}")
        logger.info(f"   🔧 Features: {config.feature_set}")
        logger.info(f"   📊 Escalador: {config.scaler_type}")
        logger.info(f"   🕐 Lookback: {config.lookback_periods} períodos")
        logger.info(f"   🔮 Horizonte: {config.prediction_horizon} períodos")
        logger.info(f"   💰 Costo total trading: {config.total_trading_cost:.3f}")
        logger.info(f"   📊 Movimiento mínimo rentable: {config.min_profitable_move:.3f}")

        # Mostrar información del período de entrenamiento
        period_info = config.get_training_period_info()
        logger.info(f"   📅 Período: {period_info['start_date']} a {period_info['end_date']}")
        logger.info(f"   📆 Total días: {period_info['total_days']}")

    def _create_scaler(self):
        """Crear escalador según configuración"""
        scaler_map = {
            "robust": RobustScaler(),
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
        }
        return scaler_map[self.config.scaler_type]

    async def fetch_market_data(self) -> pd.DataFrame:
        """Obtener datos de mercado con configuración flexible de fechas"""

        # Determinar fechas según configuración
        period_info = self.config.get_training_period_info()

        if self.config.end_date:
            end_time = int(datetime.strptime(self.config.end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_time = int(datetime.now().timestamp() * 1000)

        if self.config.start_date:
            start_time = int(datetime.strptime(self.config.start_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            start_time = int((datetime.now() - timedelta(days=self.config.training_days)).timestamp() * 1000)

        logger.info(f"📥 Descargando datos de {self.config.symbol}")
        logger.info(f"   📅 Desde: {period_info['start_date']}")
        logger.info(f"   📅 Hasta: {period_info['end_date']}")
        logger.info(f"   ⏰ Timeframe: {self.config.timeframe}")
        logger.info(f"   📊 Total días: {period_info['total_days']}")

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.config.symbol,
            'interval': self.config.timeframe,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        all_data = []
        async with aiohttp.ClientSession() as session:
            current_start = start_time

            while current_start < end_time:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"API Error: {response.status}")
                        break

                    data = await response.json()
                    if not data:
                        break

                    all_data.extend(data)
                    current_start = data[-1][6] + 1

                await asyncio.sleep(0.1)  # Rate limiting

        # Convertir a DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                  'taker_buy_quote', 'ignore']

        df = pd.DataFrame(all_data, columns=columns)

        # Limpiar datos
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        # Remover outliers extremos
        for col in numeric_cols:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)

        # Verificar integridad
        df = df.dropna()

        logger.info(f"✅ Datos obtenidos: {len(df)} registros")
        logger.info(f"   📅 Período: {df.index.min()} a {df.index.max()}")

        return df

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features técnicos usando motor centralizado"""

        logger.info(f"🔧 Calculando features técnicos con motor centralizado")
        logger.info(f"   📊 Feature set: {self.config.feature_set}")

        # Usar motor centralizado de features
        try:
            features = self.features_engine.calculate_features(df, self.config.feature_set)

            # Verificar que tenemos las features esperadas
            feature_info = self.features_engine.get_feature_info(self.config.feature_set)
            expected_features = feature_info['features']
            actual_features = list(features.columns)

            logger.info(f"   ✅ Features calculadas: {len(actual_features)} de {len(expected_features)} esperadas")

            # Almacenar columnas de features para usar en predicción
            self.feature_columns = actual_features

            # Mostrar primeras y últimas features calculadas para verificación
            if len(actual_features) > 0:
                logger.info(f"   📋 Primeras features: {actual_features[:5]}")
                if len(actual_features) > 5:
                    logger.info(f"   📋 Últimas features: {actual_features[-5:]}")

            return features

        except Exception as e:
            logger.error(f"❌ Error calculando features con motor centralizado: {e}")
            logger.info("🔄 Fallback: calculando features básicos manualmente")
            return self._calculate_basic_features_fallback(df)

    def _calculate_basic_features_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features básicos como fallback si falla el motor centralizado"""

        features = pd.DataFrame(index=df.index)

        try:
            # Features básicos esenciales
            features['returns_1'] = df['close'].pct_change(1)
            features['returns_5'] = df['close'].pct_change(5)
            features['sma_20'] = df['close'].rolling(20).mean()
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['volatility'] = df['close'].pct_change().rolling(20).std()
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # RSI básico
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss.clip(lower=1e-8)
            features['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD básico
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            # Limpiar
            features = features.fillna(method='ffill').fillna(0)

            self.feature_columns = list(features.columns)
            logger.info(f"   ✅ Features fallback calculadas: {len(self.feature_columns)}")

            return features

        except Exception as e:
            logger.error(f"❌ Error en features fallback: {e}")
            # Último recurso: features mínimas
            features['returns_1'] = df['close'].pct_change(1).fillna(0)
            features['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
            self.feature_columns = ['returns_1', 'sma_20']
        return features

    def create_labels_without_lookahead(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Crear etiquetas SIN look-ahead bias usando solo features disponibles"""

        logger.info("🏷️ Creando etiquetas sin look-ahead bias...")
        logger.info(f"   📊 Features disponibles: {list(features.columns)}")

        # Mapear features disponibles a features necesarias
        feature_mapping = self._get_safe_feature_mapping(features.columns)

        labels = []

        for i in range(len(df)):
            # Solo usar información disponible hasta el momento i
            if i < 50:  # Necesitamos historia mínima
                labels.append(1)  # HOLD
                continue

            # Información histórica hasta el momento i
            hist_data = df.iloc[:i+1]
            hist_features = features.iloc[:i+1]

            try:
                # Calcular contexto del mercado usando features disponibles
                volatility_feature = feature_mapping.get('volatility', None)
                trend_feature = feature_mapping.get('trend', None)
                rsi_feature = feature_mapping.get('rsi', None)
                macd_feature = feature_mapping.get('macd', None)
                bb_feature = feature_mapping.get('bb_position', None)
                volume_feature = feature_mapping.get('volume', None)

                # Volatilidad (usar feature disponible o calcular básica)
                if volatility_feature:
                    recent_volatility = hist_features[volatility_feature].iloc[-14:].mean()
                else:
                    # Calcular volatilidad básica
                    returns = hist_data['close'].pct_change().iloc[-14:]
                    recent_volatility = returns.std()

                # Tendencia (usar feature disponible o calcular básica)
                if trend_feature:
                    trend_strength = hist_features[trend_feature].iloc[-1]
                else:
                    # Calcular tendencia básica
                    price_change = (hist_data['close'].iloc[-1] - hist_data['close'].iloc[-20]) / hist_data['close'].iloc[-20]
                    trend_strength = price_change

                # RSI
                if rsi_feature:
                    rsi_current = hist_features[rsi_feature].iloc[-1]
                else:
                    rsi_current = 50  # Neutral si no disponible

                # MACD
                if macd_feature:
                    macd_momentum = hist_features[macd_feature].iloc[-1]
                else:
                    macd_momentum = 0  # Neutral si no disponible

                # Bollinger position
                if bb_feature:
                    bb_position = hist_features[bb_feature].iloc[-1]
                else:
                    bb_position = 0.5  # Neutral si no disponible

                # Volume strength
                if volume_feature:
                    volume_strength = hist_features[volume_feature].iloc[-1]
                else:
                    # Calcular ratio de volumen básico
                    avg_volume = hist_data['volume'].iloc[-20:].mean()
                    current_volume = hist_data['volume'].iloc[-1]
                    volume_strength = current_volume / avg_volume if avg_volume > 0 else 1.0

                # Condiciones para señales (basadas solo en información disponible)
                volatility_acceptable = recent_volatility < 0.03 if recent_volatility is not None else True

                # Señales de BUY
                buy_conditions = [
                    trend_strength > self.config.min_profitable_move if trend_strength is not None else False,
                    rsi_current < 65 if rsi_current is not None else True,
                    macd_momentum > 0 if macd_momentum is not None else False,
                    bb_position < 0.8 if bb_position is not None else True,
                    volume_strength > 1.2 if volume_strength is not None else False,
                    volatility_acceptable
                ]

                # Señales de SELL
                sell_conditions = [
                    trend_strength < -self.config.min_profitable_move if trend_strength is not None else False,
                    rsi_current > 35 if rsi_current is not None else True,
                    macd_momentum < 0 if macd_momentum is not None else False,
                    bb_position > 0.2 if bb_position is not None else True,
                    volume_strength > 1.2 if volume_strength is not None else False,
                    volatility_acceptable
                ]

                # Decisión final
                buy_score = sum(buy_conditions)
                sell_score = sum(sell_conditions)

                if buy_score >= 3 and buy_score > sell_score:
                    labels.append(2)  # BUY
                elif sell_score >= 3 and sell_score > buy_score:
                    labels.append(0)  # SELL
                else:
                    labels.append(1)  # HOLD

            except Exception as e:
                logger.warning(f"Error procesando etiqueta en índice {i}: {e}")
                labels.append(1)  # HOLD como fallback

        # Crear DataFrame con labels
        df_labeled = df.copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        logger.info("📊 Distribución de etiquetas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            logger.info(f"   {name}: {count} ({pct:.1f}%)")

        return df_labeled

    def _get_safe_feature_mapping(self, available_features: List[str]) -> Dict[str, str]:
        """Mapear features conceptuales a features disponibles"""

        mapping = {}

        # Volatilidad features (en orden de preferencia)
        volatility_options = ['atr_14', 'atr_20', 'volatility', 'natr_14', 'true_range', 'volatility_20', 'price_volatility_10']
        for feature in volatility_options:
            if feature in available_features:
                mapping['volatility'] = feature
                break

        # Tendencia features
        trend_options = ['returns_20', 'returns_10', 'returns_5', 'price_change_10', 'momentum_10', 'momentum_20']
        for feature in trend_options:
            if feature in available_features:
                mapping['trend'] = feature
                break

        # RSI features
        rsi_options = ['rsi_14', 'rsi_21', 'rsi_7']
        for feature in rsi_options:
            if feature in available_features:
                mapping['rsi'] = feature
                break

        # MACD features
        macd_options = ['macd_histogram', 'macd', 'macd_signal']
        for feature in macd_options:
            if feature in available_features:
                mapping['macd'] = feature
                break

        # Bollinger position
        bb_options = ['bb_position']
        for feature in bb_options:
            if feature in available_features:
                mapping['bb_position'] = feature
                break

        # Volume features
        volume_options = ['volume_ratio', 'volume_sma_20', 'mfi_14', 'ad']
        for feature in volume_options:
            if feature in available_features:
                mapping['volume'] = feature
                break

        logger.info(f"🔗 Mapeo de features: {mapping}")
        return mapping

    def prepare_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar secuencias temporales para el modelo"""

        logger.info(f"🔄 Preparando secuencias temporales")
        logger.info(f"   📊 Escalador: {self.config.scaler_type}")
        logger.info(f"   🕐 Lookback: {self.config.lookback_periods} períodos")

        # Seleccionar features numéricas
        numeric_features = features.select_dtypes(include=[np.number])

        # Actualizar feature_columns si no se estableció anteriormente
        if not self.feature_columns:
            self.feature_columns = list(numeric_features.columns)

        logger.info(f"   🔧 Features utilizadas: {len(self.feature_columns)}")

        # Normalizar features usando el escalador configurado
        features_scaled = self.scaler.fit_transform(numeric_features)

        # Crear secuencias
        X, y = [], []

        for i in range(self.config.lookback_periods, len(features_scaled)):
            # Secuencia de features
            sequence = features_scaled[i-self.config.lookback_periods:i]
            X.append(sequence)

            # Label correspondiente
            y.append(labels.iloc[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"   ✅ Secuencias creadas: X={X.shape}, y={y.shape}")

        # Verificar calidad de los datos
        if len(X) == 0:
            raise ValueError("No se pudieron crear secuencias. Verifique lookback_periods y cantidad de datos.")

        # Verificar distribución de clases
        unique_labels, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_labels, counts))
        logger.info(f"   📊 Distribución de clases: {class_distribution}")

        return X, y

    def create_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Crear modelo según configuración"""

        logger.info(f"🧠 Creando modelo: {self.config.model_type}")
        logger.info(f"   📐 Input shape: {input_shape}")
        logger.info(f"   🎯 Learning rate: {self.config.learning_rate}")

        # Crear modelo según tipo configurado
        if self.config.model_type == "tcn_basic":
            model = self._create_tcn_basic_model(input_shape)
        elif self.config.model_type == "tcn_advanced":
            model = self._create_tcn_advanced_model(input_shape)
        elif self.config.model_type == "lstm_tcn":
            model = self._create_lstm_tcn_model(input_shape)
        elif self.config.model_type == "transformer_tcn":
            model = self._create_transformer_tcn_model(input_shape)
        else:
            logger.warning(f"Modelo '{self.config.model_type}' no reconocido, usando tcn_advanced")
            model = self._create_tcn_advanced_model(input_shape)

        # Compilar con configuración personalizada
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                weight_decay=0.0001
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"   ✅ Modelo creado: {model.count_params():,} parámetros")

        return model

    def _create_tcn_basic_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Modelo TCN básico para experimentos rápidos"""

        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),

            # Bloques TCN básicos
            tf.keras.layers.Conv1D(32, 3, padding='causal', activation='relu'),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Conv1D(64, 3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.Dropout(0.2),

            # Pooling y clasificación
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

    def _create_tcn_advanced_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Modelo TCN avanzado con múltiples bloques de dilatación"""

        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),

            # Bloques TCN con dilataciones progresivas
            tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Conv1D(128, 3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(256, 3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(128, 3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Capas densas con regularización
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

    def _create_lstm_tcn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Modelo híbrido LSTM + TCN"""

        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),

            # Bloque LSTM
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),

            # Bloques TCN
            tf.keras.layers.Conv1D(128, 3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(256, 3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Pooling y clasificación
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

    def _create_transformer_tcn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Modelo híbrido Transformer + TCN"""

        # Capa de atención simplificada
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Normalización inicial
        x = tf.keras.layers.LayerNormalization()(inputs)

        # Bloque de atención multi-head simplificado
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=64, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)

        # Bloques TCN
        x = tf.keras.layers.Conv1D(128, 3, padding='causal', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv1D(256, 3, dilation_rate=2, padding='causal', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Pooling y clasificación
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)

    def temporal_train_test_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split temporal para entrenamiento, validación y test sin data leakage"""

        total_size = len(X)
        test_size = self.config.test_size
        validation_size = self.config.validation_size
        train_size = 1 - test_size - validation_size

        # Calcular índices
        train_idx = int(total_size * train_size)
        val_idx = int(total_size * (train_size + validation_size))

        # Split temporal
        X_train = X[:train_idx]
        X_val = X[train_idx:val_idx]
        X_test = X[val_idx:]

        y_train = y[:train_idx]
        y_val = y[train_idx:val_idx]
        y_test = y[val_idx:]

        logger.info(f"📊 Split temporal configurado:")
        logger.info(f"   🔧 Train: {len(X_train)} muestras ({train_size:.1%})")
        logger.info(f"   🔧 Validation: {len(X_val)} muestras ({validation_size:.1%})")
        logger.info(f"   🔧 Test: {len(X_test)} muestras ({test_size:.1%})")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray) -> Dict:
        """Calcular métricas específicas de trading"""

        logger.info("Calculando métricas de trading...")

        # Métricas básicas
        accuracy = np.mean(y_true == y_pred)

        # Simular trading
        portfolio_value = 1.0
        trades = []
        position = 0  # 0=cash, 1=long, -1=short

        for i in range(len(y_pred)):
            signal = y_pred[i]
            price = prices[i]

            if signal == 2 and position != 1:  # BUY signal
                if position == -1:  # Close short
                    portfolio_value *= (1 - self.config.total_trading_cost)
                portfolio_value *= (1 - self.config.total_trading_cost)  # Enter long
                position = 1
                trades.append(('BUY', price, portfolio_value))

            elif signal == 0 and position != -1:  # SELL signal
                if position == 1:  # Close long
                    portfolio_value *= (1 - self.config.total_trading_cost)
                portfolio_value *= (1 - self.config.total_trading_cost)  # Enter short
                position = -1
                trades.append(('SELL', price, portfolio_value))

            elif signal == 1 and position != 0:  # HOLD signal - close position
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = 0
                trades.append(('HOLD', price, portfolio_value))

            # Update portfolio value based on price movement
            if i < len(prices) - 1:
                price_change = (prices[i+1] - price) / price
                if position == 1:  # Long position
                    portfolio_value *= (1 + price_change)
                elif position == -1:  # Short position
                    portfolio_value *= (1 - price_change)

        # Calcular métricas
        total_return = (portfolio_value - 1.0) * 100
        num_trades = len(trades)

        # Sharpe ratio simplificado
        if len(trades) > 1:
            returns = [trade[2] for trade in trades]
            returns_pct = np.diff(returns) / returns[:-1]
            sharpe_ratio = np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0
        else:
            sharpe_ratio = 0

        # Win rate
        profitable_trades = sum(1 for i in range(1, len(trades)) if trades[i][2] > trades[i-1][2])
        win_rate = (profitable_trades / max(1, num_trades - 1)) * 100

        metrics = {
            'accuracy': accuracy,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'final_portfolio': portfolio_value
        }

        logger.info("Métricas de trading:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")

        return metrics

    async def train(self) -> Dict:
        """Entrenamiento completo del sistema"""

        logger.info("Iniciando entrenamiento profesional...")

        # 1. Obtener datos
        df = await self.fetch_market_data()

        # 2. Calcular features
        features = self.calculate_technical_features(df)

        # 3. Crear labels sin look-ahead bias
        df_labeled = self.create_labels_without_lookahead(df, features)

        # 4. Preparar secuencias
        X, y = self.prepare_sequences(features, df_labeled['label'])

        # 5. Split temporal con configuración personalizada
        X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_train_test_split(X, y)

        # 6. Calcular class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        logger.info(f"📊 Class weights: {class_weight_dict}")

        # 7. Crear y entrenar modelo
        self.model = self.create_model((X.shape[1], X.shape[2]))

        # Callbacks con configuración personalizada
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=self.config.reduce_lr_patience,
                factor=0.5,
                monitor='val_loss'
            ),
        ]

        logger.info(f"🚀 Iniciando entrenamiento")
        logger.info(f"   🔄 Épocas máximas: {self.config.epochs}")
        logger.info(f"   📦 Batch size: {self.config.batch_size}")
        logger.info(f"   ⏰ Early stopping: {self.config.early_stopping_patience} épocas")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # 8. Evaluación
        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        # Obtener precios para métricas de trading
        test_prices = df['close'].iloc[-len(y_test):].values

        metrics = self.calculate_trading_metrics(y_test, y_pred, test_prices)

        # 9. Reporte final
        logger.info("\n" + "="*50)
        logger.info("RESULTADOS FINALES")
        logger.info("="*50)
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Retorno total: {metrics['total_return']:.2f}%")
        logger.info(f"Número de trades: {metrics['num_trades']}")
        logger.info(f"Win rate: {metrics['win_rate']:.1f}%")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")

        return metrics

    def save_model(self, path: str):
        """Guardar modelo y componentes"""

        os.makedirs(path, exist_ok=True)

        # Guardar modelo
        self.model.save(f"{path}/model.h5")

        # Guardar scaler
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Guardar configuración
        with open(f"{path}/config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Guardar feature columns
        with open(f"{path}/features.json", "w") as f:
            json.dump(self.feature_columns, f, indent=2)

        logger.info(f"Modelo guardado en: {path}")

    def predict(self, market_data: pd.DataFrame) -> int:
        """Hacer predicción en tiempo real"""

        if self.model is None or self.scaler is None:
            raise ValueError("Modelo no entrenado")

        # Calcular features
        features = self.calculate_technical_features(market_data)

        # Tomar últimos períodos
        recent_features = features[self.feature_columns].iloc[-self.config.lookback_periods:]

        # Normalizar
        features_scaled = self.scaler.transform(recent_features)

        # Predecir
        X = features_scaled.reshape(1, self.config.lookback_periods, len(self.feature_columns))
        prediction = self.model.predict(X, verbose=0)

        return np.argmax(prediction[0])


def create_sample_configurations() -> List[TradingConfig]:
    """Crear configuraciones de ejemplo para diferentes estrategias"""

    configs = []

    # Configuración básica - trading rápido
    configs.append(TradingConfig(
        symbol="BTCUSDT",
        timeframe="5m",
        lookback_periods=24,
        prediction_horizon=6,
        training_days=30,
        model_type="tcn_basic",
        feature_set="tcn_final",
        scaler_type="robust",
        epochs=50,
        batch_size=64
    ))

    # Configuración avanzada - trading medium-term
    configs.append(TradingConfig(
        symbol="ETHUSDT",
        timeframe="15m",
        lookback_periods=48,
        prediction_horizon=12,
        training_days=90,
        model_type="tcn_advanced",
        feature_set="tcn_definitivo",
        scaler_type="robust",
        epochs=100,
        batch_size=32
    ))

    # Configuración híbrida - trading con dates específicas
    configs.append(TradingConfig(
        symbol="BNBUSDT",
        timeframe="1h",
        lookback_periods=72,
        prediction_horizon=24,
        start_date="2024-01-01",
        end_date="2024-06-01",
        model_type="lstm_tcn",
        feature_set="full_set",
        scaler_type="standard",
        epochs=150,
        batch_size=16
    ))

    # Configuración XRP - trading medium-term
    configs.append(TradingConfig(
        symbol="XRPUSDT",
        timeframe="30m",
        lookback_periods=36,
        prediction_horizon=8,
        training_days=60,
        model_type="tcn_advanced",
        feature_set="tcn_definitivo",
        scaler_type="robust",
        epochs=80,
        batch_size=32
    ))

    return configs

async def run_configuration(config: TradingConfig) -> Dict:
    """Ejecutar una configuración específica"""

    print(f"\n🔥 EJECUTANDO CONFIGURACIÓN: {config.symbol} - {config.model_type}")
    print("=" * 80)

    # Mostrar configuración completa
    period_info = config.get_training_period_info()
    print(f"📈 Símbolo: {config.symbol}")
    print(f"⏰ Timeframe: {config.timeframe} ({config.timeframe_minutes} min)")
    print(f"🧠 Modelo: {config.model_type}")
    print(f"🔧 Features: {config.feature_set}")
    print(f"📊 Escalador: {config.scaler_type}")
    print(f"🕐 Lookback: {config.lookback_periods} períodos")
    print(f"🔮 Horizonte: {config.prediction_horizon} períodos")
    print(f"📅 Período: {period_info['start_date']} a {period_info['end_date']} ({period_info['total_days']} días)")
    print(f"🔄 Entrenamiento: {config.epochs} épocas, batch {config.batch_size}")
    print(f"📊 Split: Train {1-config.test_size-config.validation_size:.1%}, Val {config.validation_size:.1%}, Test {config.test_size:.1%}")
    print(f"💰 Costos: {config.total_trading_cost:.3f} | Min rentable: {config.min_profitable_move:.3f}")

    # Crear y entrenar trader
    trader = ProfessionalCryptoTrader(config)

    try:
        metrics = await trader.train()

        # Mostrar resumen de resultados
        print(f"\n🎯 RESULTADOS FINALES - {config.symbol}")
        print("=" * 50)
        print(f"💡 Accuracy: {metrics['accuracy']:.3f}")
        print(f"💰 Retorno total: {metrics['total_return']:.2f}%")
        print(f"🔄 Número de trades: {metrics['num_trades']}")
        print(f"🎯 Win rate: {metrics['win_rate']:.1f}%")
        print(f"📊 Sharpe ratio: {metrics['sharpe_ratio']:.3f}")

        # Guardar modelo si es rentable
        model_path = f"models/{config.model_type}_{config.symbol.lower()}_{config.timeframe}"
        if metrics['total_return'] > 0:
            trader.save_model(model_path)
            print(f"✅ Modelo rentable guardado en: {model_path}")
        else:
            print(f"⚠️ Modelo no rentable - no guardado")

        # Ejemplo de predicción
        print(f"\n🔮 Ejemplo de predicción en tiempo real:")
        df_example = await trader.fetch_market_data()
        prediction = trader.predict(df_example)
        signal_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        print(f"Señal actual: {signal_names[prediction]}")

        return metrics

    except Exception as e:
        logger.error(f"❌ Error durante entrenamiento de {config.symbol}: {e}")
        return {"error": str(e)}

def get_user_input(prompt: str, options: List[str] = None, default: str = None) -> str:
    """Obtener input del usuario de manera segura"""
    try:
        if options:
            print(f"\n{prompt}")
            for i, option in enumerate(options, 1):
                print(f"   {i}. {option}")
            if default:
                print(f"   (Presiona Enter para default: {default})")

            while True:
                try:
                    choice = input("\n👉 Selecciona una opción: ").strip()
                except EOFError:
                    # Si no hay input disponible, usar default o primera opción
                    if default:
                        print(f"📝 Usando opción por defecto: {default}")
                        return default
                    else:
                        print(f"📝 Usando primera opción: {options[0]}")
                        return options[0]

                if not choice and default:
                    return default

                # Intentar como número primero
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                    else:
                        print(f"❌ Por favor selecciona un número entre 1 y {len(options)}")
                        continue
                except ValueError:
                    pass

                # Intentar como texto directo (coincidencia exacta)
                if choice in options:
                    return choice

                # Buscar coincidencia parcial (case insensitive)
                for option in options:
                    if choice.lower() == option.lower():
                        return option

                print("❌ Opción no válida. Ingresa un número o el texto de la opción")
        else:
            try:
                if default:
                    result = input(f"\n{prompt} (default: {default}): ").strip()
                    return result if result else default
                else:
                    return input(f"\n{prompt}: ").strip()
            except EOFError:
                if default:
                    print(f"📝 Usando valor por defecto: {default}")
                    return default
                else:
                    print("❌ Se requiere entrada del usuario")
                    return ""
    except KeyboardInterrupt:
        print(f"\n\n👋 Operación cancelada por el usuario")
        exit(0)

def create_interactive_config() -> TradingConfig:
    """Crear configuración de manera interactiva"""

    print("\n🎛️ CONFIGURACIÓN INTERACTIVA DEL SISTEMA")
    print("=" * 50)

    # Symbol
    symbol = get_user_input(
        "💰 Símbolo de criptomoneda",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT", "AVAXUSDT"],
        "BTCUSDT"
    )

    # Timeframe
    timeframe = get_user_input(
        "⏰ Timeframe",
        ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        "5m"
    )

    # Model type
    model_type = get_user_input(
        "🧠 Tipo de modelo",
        ["tcn_basic", "tcn_advanced", "lstm_tcn", "transformer_tcn"],
        "tcn_advanced"
    )

    # Feature set
    feature_set = get_user_input(
        "🔧 Conjunto de features",
        ["tcn_final", "tcn_definitivo", "full_set"],
        "tcn_definitivo"
    )

    # Scaler type
    scaler_type = get_user_input(
        "📊 Tipo de escalador",
        ["robust", "standard", "minmax"],
        "robust"
    )

    # Training period
    period_type = get_user_input(
        "📅 Tipo de período de entrenamiento",
        ["dias_atras", "fecha_especifica", "rango_fechas"],
        "dias_atras"
    )

    start_date = None
    end_date = None
    training_days = None

    if period_type == "dias_atras":
        training_days = int(get_user_input(
            "📆 Días de entrenamiento hacia atrás",
            default="90"
        ))
    elif period_type == "fecha_especifica":
        start_date = get_user_input(
            "📅 Fecha de inicio (YYYY-MM-DD)",
            default="2024-01-01"
        )
    elif period_type == "rango_fechas":
        start_date = get_user_input(
            "📅 Fecha de inicio (YYYY-MM-DD)",
            default="2024-01-01"
        )
        end_date = get_user_input(
            "📅 Fecha de fin (YYYY-MM-DD)",
            default="2024-06-01"
        )

    # Advanced settings
    print("\n⚙️ CONFIGURACIÓN AVANZADA (opcional)")
    advanced = get_user_input(
        "¿Configurar parámetros avanzados?",
        ["si", "no"],
        "no"
    )

    epochs = 100
    batch_size = 32
    lookback_periods = 48

    if advanced == "si":
        epochs = int(get_user_input("🔄 Épocas de entrenamiento", default="100"))
        batch_size = int(get_user_input("📦 Batch size", default="32"))
        lookback_periods = int(get_user_input("🕐 Períodos de lookback", default="48"))

    # Crear configuración
    config = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        model_type=model_type,
        feature_set=feature_set,
        scaler_type=scaler_type,
        start_date=start_date,
        end_date=end_date,
        training_days=training_days,
        epochs=epochs,
        batch_size=batch_size,
        lookback_periods=lookback_periods
    )

    return config

async def main():
    """Función principal interactiva"""

    print("🎯 SISTEMA PROFESIONAL DE TRADING DE CRIPTOMONEDAS")
    print("🔥 VERSIÓN ARMONIZADA CON MOTOR CENTRALIZADO DE FEATURES")
    print("=" * 80)

    # Modo de operación
    mode = get_user_input(
        "🚀 Modo de operación",
        ["interactivo", "ejemplos_predefinidos", "prueba_rapida"],
        "interactivo"
    )

    if mode == "interactivo":
        # Configuración interactiva
        config = create_interactive_config()

        print(f"\n📋 CONFIGURACIÓN CREADA:")
        print("=" * 50)
        period_info = config.get_training_period_info()
        print(f"📈 Símbolo: {config.symbol}")
        print(f"⏰ Timeframe: {config.timeframe}")
        print(f"🧠 Modelo: {config.model_type}")
        print(f"🔧 Features: {config.feature_set}")
        print(f"📊 Escalador: {config.scaler_type}")
        print(f"📅 Período: {period_info['start_date']} a {period_info['end_date']} ({period_info['total_days']} días)")
        print(f"🔄 Entrenamiento: {config.epochs} épocas, batch {config.batch_size}")

        confirm = get_user_input(
            "\n¿Proceder con esta configuración?",
            ["si", "no"],
            "si"
        )

        if confirm == "si":
            print(f"\n🚀 INICIANDO ENTRENAMIENTO...")
            results = await run_configuration(config)

            if "error" not in results:
                print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            else:
                print(f"\n❌ Error durante el entrenamiento: {results['error']}")
        else:
            print(f"\n👋 Operación cancelada")

    elif mode == "ejemplos_predefinidos":
        # Configuraciones predefinidas
        configs = create_sample_configurations()

        print(f"\n📋 Configuraciones predefinidas disponibles:")
        config_descriptions = []
        for i, config in enumerate(configs):
            desc = f"{config.symbol} - {config.model_type} - {config.timeframe} - {config.feature_set}"
            config_descriptions.append(desc)
            print(f"   {i+1}. {desc}")

        choice = get_user_input(
            "Selecciona una configuración",
            ["todas"] + config_descriptions,
            "todas"
        )

        if choice == "todas":
            print(f"\n🚀 Ejecutando todas las configuraciones...")
            all_results = {}

            for i, config in enumerate(configs):
                try:
                    print(f"\n{'='*20} CONFIGURACIÓN {i+1}/{len(configs)} {'='*20}")
                    results = await run_configuration(config)
                    all_results[f"{config.symbol}_{config.model_type}"] = results
                except Exception as e:
                    logger.error(f"Error en configuración {i+1}: {e}")
                    all_results[f"{config.symbol}_{config.model_type}"] = {"error": str(e)}

            # Resumen final
            print(f"\n🏆 RESUMEN FINAL DE TODAS LAS CONFIGURACIONES")
            print("=" * 80)

            successful_configs = []
            for name, results in all_results.items():
                if "error" not in results:
                    successful_configs.append((name, results['total_return']))
                    print(f"✅ {name}: {results['total_return']:.2f}% retorno")
                else:
                    print(f"❌ {name}: Error - {results['error']}")

            if successful_configs:
                best_config = max(successful_configs, key=lambda x: x[1])
                print(f"\n🥇 MEJOR CONFIGURACIÓN: {best_config[0]} con {best_config[1]:.2f}% retorno")
        else:
            # Ejecutar configuración específica
            selected_idx = config_descriptions.index(choice)
            selected_config = configs[selected_idx]

            print(f"\n🚀 Ejecutando configuración seleccionada...")
            results = await run_configuration(selected_config)

            if "error" not in results:
                print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            else:
                print(f"\n❌ Error durante el entrenamiento: {results['error']}")

    elif mode == "prueba_rapida":
        # Configuración rápida para pruebas
        print(f"\n🧪 MODO PRUEBA RÁPIDA")
        config = TradingConfig(
            symbol="BTCUSDT",
            timeframe="5m",
            model_type="tcn_basic",
            feature_set="tcn_final",
            training_days=7,
            epochs=5,
            batch_size=16,
            lookback_periods=12
        )

        print("⚡ Usando configuración rápida para pruebas...")
        results = await run_configuration(config)

        if "error" not in results:
            print(f"\n🎉 ¡PRUEBA COMPLETADA!")
        else:
            print(f"\n❌ Error en prueba: {results['error']}")

    print(f"\n🎯 Proceso completo finalizado!")


if __name__ == "__main__":
    asyncio.run(main())
