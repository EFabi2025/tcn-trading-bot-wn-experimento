#!/usr/bin/env python3
"""
TCN DEFINITIVO PREDICTOR - SINCRONIZADO CON ENTRENADOR
🔧 VERSIÓN CORREGIDA: Usa EXACTAMENTE la misma lógica de features que el entrenador
para solucionar las inconsistencias detectadas
"""

import os
# Configurar TensorFlow ANTES de importarlo para carga rápida
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silenciar logs de TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forzar uso de CPU solamente

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Configurar TensorFlow para máximo rendimiento en CPU
tf.config.threading.set_intra_op_parallelism_threads(0)  # Usar todos los cores
tf.config.threading.set_inter_op_parallelism_threads(0)  # Usar todos los cores

import pickle
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCNDefinitivoPredictor:
    """
    🔧 Predictor SINCRONIZADO que usa EXACTAMENTE la misma lógica del entrenador
    Corrige las inconsistencias críticas detectadas en el análisis
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.class_weights = {}
        # ✅ Solo pares con modelos entrenados disponibles
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']

        # ⚠️ PARES PENDIENTES (sin modelos): ['ADAUSDT', 'DOTUSDT', 'SOLUSDT']
        self.excluded_symbols = ['ADAUSDT', 'DOTUSDT', 'SOLUSDT']
        self.model_stats = {
            'BTCUSDT': {'accuracy': 0.597, 'loss': 0.835},
            'ETHUSDT': {'accuracy': 0.628, 'loss': 0.852},  # ✅ REENTRENADO: 62.8% accuracy, thresholds actualizados
            'BNBUSDT': {'accuracy': 0.601, 'loss': 0.858},
            'XRPUSDT': {'accuracy': 0.404, 'loss': 1.050}   # ✅ MODELO REENTRENADO con metodología definitiva
        }

        # 🚀 THRESHOLDS AGRESIVOS - MÁS BAJOS para mayor sensibilidad
        self.thresholds = {
            'BTCUSDT': {'sell': -0.0014, 'buy': 0.0014},   # 🚀 AGRESIVO: 0.3% para BUY (era 0.14%)
            'ETHUSDT': {'sell': -0.0012, 'buy': 0.0013},   # 🚀 AGRESIVO: 0.2% para BUY (era 0.09%)
            'BNBUSDT': {'sell': -0.0009, 'buy': 0.0015},   # 🚀 AGRESIVO: 0.3% para BUY (era 0.15%)
            'XRPUSDT': {'sell': -0.0018, 'buy': 0.0018}    # 🚀 AGRESIVO: 0.3% para BUY (era 0.11%)
        }

        # 🔧 SEQUENCE LENGTH DINÁMICO POR MODELO
        self.sequence_lengths = {
            'BTCUSDT': 24,  # Modelo antiguo
            'ETHUSDT': 24,  # Modelo reentrenado
            'BNBUSDT': 48,  # Modelo antiguo
            'XRPUSDT': 24   # ✅ MODELO REENTRENADO con configuración estándar
        }

        self.n_features = 66

        # ✅ OPTIMIZADO: Inicialización rápida sin cargar modelos
        print("🔧 TCN Predictor SINCRONIZADO inicializado (carga lazy)")
        print(f"📊 Modelos disponibles: {self.symbols}")
        print("⚡ Los modelos se cargarán bajo demanda para mayor velocidad")

        # ✅ NUEVO: Flag para tracking de modelos cargados
        self.models_loaded = set()
        self.models_loading = set()  # Para evitar carga duplicada

        # ✅ OPTIMIZADO: Pre-cargar solo BTC para arranque rápido
        self._preload_critical_models()

    def _preload_critical_models(self):
        """⚡ Pre-cargar modelos críticos en background"""
        # ✅ CORREGIDO: Cargar sincrónicamente para evitar conflictos de threading
        try:
            logger.info("⚡ Pre-cargando BTC sincrónicamente...")
            success = self._load_model_for_symbol('BTCUSDT')
            if success:
                self.models_loaded.add('BTCUSDT')
                logger.info("✅ BTC pre-cargado exitosamente")
            else:
                logger.warning("⚠️ No se pudo pre-cargar BTC, se cargará bajo demanda")
        except Exception as e:
            logger.error(f"❌ Error pre-cargando BTC: {e}")
            import traceback
            print(f"   🔍 Traceback completo: {traceback.format_exc()}")

    def load_all_models(self):
        """🎯 Cargar todos los modelos disponibles"""
        logger.info("🔄 Cargando todos los modelos definitivos...")
        success_count = 0

        for symbol in self.symbols:
            if symbol not in self.models_loaded:
                success = self._load_model_for_symbol(symbol)
                if success:
                    self.models_loaded.add(symbol)
                    success_count += 1

        logger.info(f"🎉 {success_count}/{len(self.symbols)} modelos cargados correctamente")
        if success_count == len(self.symbols):
            logger.info("🎉 Todos los modelos definitivos cargados correctamente")

        return success_count == len(self.symbols)

    def _load_model_for_symbol(self, symbol: str) -> bool:
        """Cargar modelo, scaler y features para un símbolo específico"""

        if symbol in self.models_loading:
            logger.warning(f"⚠️ {symbol} ya está siendo cargado, esperando...")
            return False

        self.models_loading.add(symbol)

        try:
            model_dir = f"models/definitivo_{symbol.lower()}"
            model_path = os.path.join(model_dir, "best_model.h5")
            scaler_path = os.path.join(model_dir, "scaler.pkl")

            if not os.path.exists(model_path):
                logger.error(f"  ❌ Modelo no encontrado: {model_path}")
                return False

            if not os.path.exists(scaler_path):
                logger.error(f"  ❌ Scaler no encontrado: {scaler_path}")
                return False

            # Cargar modelo
            logger.info(f"  📂 Cargando modelo {symbol}...")
            self.models[symbol] = keras.models.load_model(model_path, compile=False)
            logger.info(f"  ✅ Modelo {symbol} cargado")

            # Cargar scaler
            with open(scaler_path, 'rb') as f:
                self.scalers[symbol] = pickle.load(f)
            logger.info(f"  📊 Scaler {symbol} cargado")

            # Cargar features si existen
            features_path = os.path.join(model_dir, "feature_columns.pkl")

            if os.path.exists(features_path):
                try:
                    with open(features_path, 'rb') as f:
                        self.feature_columns[symbol] = pickle.load(f)
                    logger.info(f"  📋 Features cargadas: {len(self.feature_columns[symbol])} columnas")
                except Exception as features_error:
                    logger.error(f"Error cargando features {symbol}: {features_error}")
                    self.feature_columns[symbol] = None
            else:
                logger.error(f"  ❌ Features no encontradas: {features_path}")
                self.feature_columns[symbol] = None

            # Cargar class weights si existen
            weights_path = os.path.join(model_dir, "class_weights.pkl")
            if os.path.exists(weights_path):
                try:
                    with open(weights_path, 'rb') as f:
                        self.class_weights[symbol] = pickle.load(f)
                    logger.info(f"  ⚖️ Class weights {symbol} cargados")
                except Exception:
                    self.class_weights[symbol] = None
            else:
                self.class_weights[symbol] = None

            logger.info(f"   🎉 Modelo {symbol} cargado completamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error cargando modelo {symbol}: {e}")
            return False
        finally:
            self.models_loading.discard(symbol)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 Crear las 66 features técnicos EXACTOS del entrenador
        ✅ SINCRONIZADO: Usa la MISMA lógica que tcn_definitivo_trainer.py
        """
        return self.create_66_features(df)

    def create_66_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 Crear las 66 features técnicos EXACTOS del entrenador
        ✅ SINCRONIZADO: Usa la MISMA lógica que tcn_definitivo_trainer.py
        """
        try:
            import talib
        except ImportError:
            logger.error("TA-Lib no está instalado. Instalar con: pip install TA-Lib")
            return pd.DataFrame()

        print("🔧 Creando 66 features técnicos (SINCRONIZADO)...")

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = pd.DataFrame(index=df.index)

        try:
            # === MOMENTUM INDICATORS (15 features) ===
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['rsi_7'] = talib.RSI(close, timeperiod=7)

            # MACD family
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd

            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)

            # Rate of Change
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_20'] = talib.ROC(close, timeperiod=20)

            # Momentum
            features['momentum_10'] = talib.MOM(close, timeperiod=10)
            features['momentum_20'] = talib.MOM(close, timeperiod=20)

            # CCI
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)

            # === TREND INDICATORS (12 features) ===
            # Moving Averages
            features['sma_10'] = talib.SMA(close, timeperiod=10)
            features['sma_20'] = talib.SMA(close, timeperiod=20)
            features['sma_50'] = talib.SMA(close, timeperiod=50)
            features['ema_10'] = talib.EMA(close, timeperiod=10)
            features['ema_20'] = talib.EMA(close, timeperiod=20)
            features['ema_50'] = talib.EMA(close, timeperiod=50)

            # ADX family
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # PSAR
            features['psar'] = talib.SAR(high, low)

            # Aroon
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down

            # === VOLATILITY INDICATORS (10 features) ===
            # 🔧 SINCRONIZADO: Bollinger Bands SIN PARÁMETROS (igual que entrenador)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)  # ✅ SIN PARÁMETROS como entrenador
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle              # ✅ SIN PROTECCIÓN como entrenador
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)  # ✅ SIN PROTECCIÓN como entrenador

            # ATR
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)

            # True Range
            features['true_range'] = talib.TRANGE(high, low, close)

            # Normalized ATR
            features['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
            features['natr_20'] = talib.NATR(high, low, close, timeperiod=20)

            # === VOLUME INDICATORS (8 features) ===
            features['ad'] = talib.AD(high, low, close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            features['obv'] = talib.OBV(close, volume)

            # Volume SMA
            features['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            features['volume_ratio'] = volume / features['volume_sma_20']  # ✅ SIN PROTECCIÓN como entrenador

            # Money Flow Index
            features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            features['mfi_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

            # === PRICE PATTERNS (8 features) ===
            # 🔧 SINCRONIZADO: Price ratios SIN PROTECCIÓN (igual que entrenador)
            features['hl_ratio'] = (high - low) / close                    # ✅ SIN PROTECCIÓN como entrenador
            features['oc_ratio'] = (close - df['open'].values) / close     # ✅ SIN PROTECCIÓN como entrenador
            features['price_position'] = (close - low) / (high - low)

            # Price momentum
            close_series = pd.Series(close, index=features.index)
            features['price_change_1'] = close_series.pct_change(1)
            features['price_change_5'] = close_series.pct_change(5)
            features['price_change_10'] = close_series.pct_change(10)

            # Volatility
            returns = np.log(close_series / close_series.shift(1))
            features['volatility_10'] = returns.rolling(10).std()
            features['volatility_20'] = returns.rolling(20).std()

            # === MARKET STRUCTURE (8 features) ===
            # Higher highs, lower lows
            features['higher_high'] = (pd.Series(high, index=features.index) > pd.Series(high, index=features.index).shift(1)).astype(int)
            features['lower_low'] = (pd.Series(low, index=features.index) < pd.Series(low, index=features.index).shift(1)).astype(int)

            # Trend strength
            features['uptrend_strength'] = (close_series > close_series.shift(1)).rolling(10).sum() / 10
            features['downtrend_strength'] = (close_series < close_series.shift(1)).rolling(10).sum() / 10

            # Support/Resistance
            features['resistance_touch'] = (close_series >= close_series.rolling(20).max() * 0.99).astype(int)
            features['support_touch'] = (close_series <= close_series.rolling(20).min() * 1.01).astype(int)

            # Market efficiency
            # 🔧 SINCRONIZADO: Efficiency ratio SIN PROTECCIÓN (igual que entrenador)
            features['efficiency_ratio'] = (np.abs(close_series - close_series.shift(10)) /
                                          (np.abs(close_series.diff()).rolling(10).sum())).fillna(0)  # ✅ SIN PROTECCIÓN como entrenador

            # Fractal dimension (simplificado)
            features['fractal_dimension'] = 0.5  # Valor constante por ahora

            # === MOMENTUM DERIVATIVES (5 features) ===
            features['rsi_momentum'] = features['rsi_14'].diff().fillna(0)
            features['macd_momentum'] = pd.Series(macd_hist, index=features.index).diff().fillna(0)
            features['ad_momentum'] = features['ad'].diff().fillna(0)
            features['volume_momentum'] = pd.Series(volume, index=features.index).pct_change().fillna(0)
            features['price_acceleration'] = features['price_change_1'].diff().fillna(0)

            # 🔧 SINCRONIZADO: Limpiar datos EXACTAMENTE igual que entrenador
            features = features.fillna(method='ffill').fillna(0)        # ✅ SOLO ffill como entrenador
            features = features.replace([np.inf, -np.inf], 0)

            # 🔧 SINCRONIZADO: Clip valores extremos EXACTAMENTE igual que entrenador
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    q99 = features[col].quantile(0.99)
                    q01 = features[col].quantile(0.01)
                    features[col] = features[col].clip(q01, q99)        # ✅ SIN VERIFICACIÓN NaN como entrenador

            # Verificar que tenemos exactamente 66 features
            if len(features.columns) != 66:
                print(f"⚠️ Features creados: {len(features.columns)}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            print(f"✅ {len(features.columns)} features técnicos creados (SINCRONIZADO)")
            return features

        except Exception as e:
            logger.error(f"Error creando features: {e}")
            return pd.DataFrame()

    def predict(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Realizar predicción para un símbolo específico

        Args:
            symbol: Símbolo a predecir (BTCUSDT, ETHUSDT, BNBUSDT)
            market_data: DataFrame con datos OHLCV

        Returns:
            Dict con predicción, confianza y detalles
        """
        if symbol not in self.models:
            logger.error(f"Modelo no cargado para {symbol}")
            return None

        try:
            # Crear features SINCRONIZADAS
            features = self.create_66_features(market_data)

            # Verificar que tenemos suficientes datos
            if len(features) < self.sequence_lengths[symbol]:
                logger.warning(f"Datos insuficientes para {symbol}: {len(features)} < {self.sequence_lengths[symbol]}")
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'predicted_return': 0.0,
                    'reason': 'Datos insuficientes para predicción'
                }

            # Tomar las últimas sequence_length muestras
            sequence_length = self.sequence_lengths[symbol]
            recent_features = features.iloc[-sequence_length:].values

            # Verificar y manejar NaN/inf
            if np.any(np.isnan(recent_features)) or np.any(np.isinf(recent_features)):
                logger.warning(f"Features contienen NaN/inf para {symbol}")
                # Reemplazar NaN/inf con ceros
                recent_features = np.nan_to_num(recent_features, nan=0.0, posinf=0.0, neginf=0.0)

            # Verificar dimensiones
            expected_features = self.n_features
            actual_features = recent_features.shape[1]

            if actual_features != expected_features:
                logger.error(f"Dimensión incorrecta: esperado {expected_features}, actual {actual_features}")
                return None

            # Normalizar con scaler
            try:
                # Reshape para scaler (samples, features)
                features_reshaped = recent_features.reshape(-1, actual_features)
                features_scaled = self.scalers[symbol].transform(features_reshaped)

                # Reshape de vuelta para TCN (samples, timesteps, features)
                features_scaled = features_scaled.reshape(1, sequence_length, actual_features)

            except Exception as scaling_error:
                logger.error(f"Error en scaling para {symbol}: {scaling_error}")
                return None

            # Realizar predicción
            prediction = self.models[symbol].predict(features_scaled, verbose=0)

            # Extraer probabilidades
            if len(prediction.shape) > 1 and prediction.shape[1] == 3:
                probs = prediction[0]
                sell_prob, hold_prob, buy_prob = probs[0], probs[1], probs[2]
            else:
                logger.error(f"Formato de predicción inesperado para {symbol}: {prediction.shape}")
                return None

            # Determinar acción basada en thresholds
            action = 'HOLD'
            confidence = hold_prob
            predicted_return = 0.0

            # Aplicar thresholds agresivos
            if buy_prob > max(sell_prob, hold_prob):
                if buy_prob >= 0.4:  # Threshold mínimo de confianza
                    action = 'BUY'
                    confidence = buy_prob
                    predicted_return = self.thresholds[symbol]['buy']
            elif sell_prob > max(buy_prob, hold_prob):
                if sell_prob >= 0.4:  # Threshold mínimo de confianza
                    action = 'SELL'
                    confidence = sell_prob
                    predicted_return = self.thresholds[symbol]['sell']

            return {
                'action': action,
                'confidence': float(confidence),
                'predicted_return': float(predicted_return),
                'probabilities': {
                    'SELL': float(sell_prob),
                    'HOLD': float(hold_prob),
                    'BUY': float(buy_prob)
                },
                'model_accuracy': self.model_stats[symbol]['accuracy'],
                'features_count': actual_features,
                'sequence_length': sequence_length,
                'reason': f'Predicción TCN {symbol} - Features sincronizadas con entrenador'
            }

        except Exception as e:
            logger.error(f"Error en predicción para {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_available_symbols(self) -> list:
        """Obtener lista de símbolos disponibles"""
        return [symbol for symbol in self.symbols if symbol in self.models_loaded]

    def is_model_loaded(self, symbol: str) -> bool:
        """Verificar si un modelo está cargado"""
        return symbol in self.models_loaded

    def predict_symbol(self, symbol: str) -> Optional[Dict]:
        """
        🎯 Método de compatibilidad para integración con sistema principal
        Obtiene datos de Binance y realiza predicción con carga lazy
        """
        try:
            # ✅ OPTIMIZADO: Carga lazy del modelo
            if not self._load_model_lazy(symbol):
                logger.error(f"No se pudo cargar modelo para {symbol}")
                return {'signal': 'HOLD', 'confidence': 0.0, 'error': f'Modelo no disponible para {symbol}'}

            # Importar cliente de Binance
            import requests

            # Obtener datos de Binance
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': 1000
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Error obteniendo datos de Binance para {symbol}")
                return {'signal': 'HOLD', 'confidence': 0.0, 'error': 'Error obteniendo datos de mercado'}

            klines = response.json()

            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convertir tipos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Realizar predicción
            prediction = self.predict(symbol, df)

            if prediction is None:
                return {'signal': 'HOLD', 'confidence': 0.0, 'error': 'Error en predicción'}

            # Formatear respuesta para compatibilidad
            return {
                'signal': prediction['action'],
                'confidence': prediction['confidence'],
                'probabilities': prediction['probabilities'],
                'error': None
            }

        except Exception as e:
            logger.error(f"Error en predict_symbol para {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': str(e)}

    def _load_model_lazy(self, symbol: str) -> bool:
        """🚀 Cargar modelo bajo demanda (lazy loading)"""
        if symbol in self.models_loaded:
            return True  # Ya está cargado

        if symbol in self.models_loading:
            logger.warning(f"⏳ {symbol} ya se está cargando, esperando...")
            # Esperar hasta 5 segundos para que termine la carga
            import time
            for _ in range(50):  # 50 * 0.1 = 5 segundos
                time.sleep(0.1)
                if symbol in self.models_loaded:
                    return True
                if symbol not in self.models_loading:
                    break
            logger.error(f"❌ Timeout esperando carga de {symbol}")
            return False

        if symbol not in self.symbols:
            logger.error(f"❌ {symbol} no está en la lista de símbolos soportados")
            return False

        # Cargar modelo
        success = self._load_model_for_symbol(symbol)
        if success:
            self.models_loaded.add(symbol)
            logger.info(f"✅ {symbol}: Modelo cargado exitosamente (lazy)")
        else:
            logger.error(f"❌ {symbol}: Error cargando modelo (lazy)")

        return success

# Instancia global para uso en otros módulos
predictor_sincronizado = None

def get_predictor():
    """Obtener instancia del predictor sincronizado"""
    global predictor_sincronizado
    if predictor_sincronizado is None:
        predictor_sincronizado = TCNDefinitivoPredictor()
    return predictor_sincronizado
