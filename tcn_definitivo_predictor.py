#!/usr/bin/env python3
"""
TCN DEFINITIVO PREDICTOR - Integración de Modelos al Sistema Principal
Predictor unificado que utiliza los 3 modelos definitivos entrenados
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
    Predictor definitivo que integra los 3 modelos TCN entrenados
    - BTCUSDT: 59.7% accuracy, distribución balanceada
    - ETHUSDT: ~60% accuracy, distribución balanceada
    - BNBUSDT: 60.1% accuracy, distribución balanceada
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

        # Thresholds específicos utilizados en entrenamiento
        self.thresholds = {
            'BTCUSDT': {'sell': -0.0014, 'buy': 0.0014},  # -0.14%/+0.14%
            'ETHUSDT': {'sell': -0.0008, 'buy': 0.0009},  # ✅ ACTUALIZADOS: -0.08%/+0.09% (corrige drift 68.7%)
            'BNBUSDT': {'sell': -0.0015, 'buy': 0.0015},  # -0.15%/+0.15%
            'XRPUSDT': {'sell': -0.0010, 'buy': 0.0011}   # ✅ BASADO EN ANÁLISIS REAL DE VOLATILIDAD
        }

        # 🔧 SEQUENCE LENGTH DINÁMICO POR MODELO
        self.sequence_lengths = {
            'BTCUSDT': 48,  # Modelo antiguo
            'ETHUSDT': 24,  # Modelo reentrenado
            'BNBUSDT': 48,  # Modelo antiguo
            'XRPUSDT': 24   # ✅ MODELO REENTRENADO con configuración estándar
        }

        self.n_features = 66

        # ✅ OPTIMIZADO: Inicialización rápida sin cargar modelos
        print("🚀 TCN Definitivo Predictor inicializado (carga lazy)")
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

    def load_all_models(self) -> bool:
        """Cargar todos los modelos definitivos"""
        logger.info("🔄 Cargando modelos definitivos...")

        success_count = 0
        for symbol in self.symbols:
            if self._load_model_for_symbol(symbol):
                success_count += 1
                logger.info(f"✅ {symbol}: Modelo cargado exitosamente")
            else:
                logger.error(f"❌ {symbol}: Error cargando modelo")

        if success_count == len(self.symbols):
            logger.info("🎉 Todos los modelos definitivos cargados correctamente")
            return True
        else:
            logger.warning(f"⚠️ Solo {success_count}/{len(self.symbols)} modelos cargados")
            return False

    def _load_model_for_symbol(self, symbol: str) -> bool:
        """Cargar modelo, scaler y features para un símbolo específico"""
        try:
            model_dir = f"models/definitivo_{symbol.lower()}"

            # Verificar que el directorio existe
            if not os.path.exists(model_dir):
                logger.error(f"Directorio no encontrado: {model_dir}")
                return False

            # Cargar modelo
            model_path = os.path.join(model_dir, "best_model.h5")
            if os.path.exists(model_path):
                self.models[symbol] = tf.keras.models.load_model(model_path)
                logger.info(f"  📊 Modelo cargado: {model_path}")
            else:
                logger.error(f"  ❌ Modelo no encontrado: {model_path}")
                return False

            # Cargar scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
                logger.info(f"  🔧 Scaler cargado: {scaler_path}")
            else:
                logger.error(f"  ❌ Scaler no encontrado: {scaler_path}")
                return False

            # Cargar feature columns
            features_path = os.path.join(model_dir, "feature_columns.pkl")
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.feature_columns[symbol] = pickle.load(f)
                logger.info(f"  📋 Features cargadas: {len(self.feature_columns[symbol])} columnas")
            else:
                logger.error(f"  ❌ Features no encontradas: {features_path}")
                return False

            # Cargar class weights (opcional)
            weights_path = os.path.join(model_dir, "class_weights.pkl")
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    self.class_weights[symbol] = pickle.load(f)
                logger.info(f"  ⚖️ Class weights cargados")

            return True

        except Exception as e:
            logger.error(f"Error cargando modelo {symbol}: {e}")
            return False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear las 66 features técnicos EXACTOS utilizados en entrenamiento
        ✅ CORREGIDO: Parámetros consistentes con CentralizedFeaturesEngine
        """
        try:
            import talib
        except ImportError:
            logger.error("TA-Lib no está instalado. Instalar con: pip install TA-Lib")
            return pd.DataFrame()

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
            # ✅ CORREGIDO: Bollinger Bands con parámetros explícitos (igual que CentralizedFeaturesEngine)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower

            # ✅ CORREGIDO: Manejo seguro de división por cero
            bb_range = bb_upper - bb_lower
            bb_range = np.where(bb_range == 0, 1e-8, bb_range)
            bb_middle_safe = np.where(bb_middle == 0, 1e-8, bb_middle)

            features['bb_width'] = bb_range / bb_middle_safe
            features['bb_position'] = (close - bb_lower) / bb_range

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

            # ✅ CORREGIDO: Manejo seguro de división por cero
            volume_sma_20_safe = np.where(features['volume_sma_20'] == 0, 1e-8, features['volume_sma_20'])
            features['volume_ratio'] = volume / volume_sma_20_safe

            # Money Flow Index
            features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            features['mfi_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

            # === PRICE PATTERNS (8 features) ===
            # ✅ CORREGIDO: Manejo seguro de división por cero
            close_safe = np.where(close == 0, 1e-8, close)
            hl_range = high - low
            hl_range_safe = np.where(hl_range == 0, 1e-8, hl_range)

            features['hl_ratio'] = hl_range / close_safe
            features['oc_ratio'] = (close - df['open'].values) / close_safe
            features['price_position'] = (close - low) / hl_range_safe

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
            price_diff_abs = np.abs(close_series.diff()).rolling(10).sum()
            price_diff_abs_safe = price_diff_abs.replace(0, 1e-8)
            features['efficiency_ratio'] = (np.abs(close_series - close_series.shift(10)) / price_diff_abs_safe).fillna(0)

            # Fractal dimension (simplificado)
            features['fractal_dimension'] = 0.5  # Valor constante por ahora

            # === MOMENTUM DERIVATIVES (5 features) ===
            features['rsi_momentum'] = features['rsi_14'].diff().fillna(0)
            features['macd_momentum'] = pd.Series(macd_hist, index=features.index).diff().fillna(0)
            features['ad_momentum'] = features['ad'].diff().fillna(0)
            features['volume_momentum'] = pd.Series(volume, index=features.index).pct_change().fillna(0)
            features['price_acceleration'] = features['price_change_1'].diff().fillna(0)

            # ✅ MEJORADO: Limpiar datos de forma más robusta
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # ✅ MEJORADO: Clip valores extremos de forma más conservadora
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    q99 = features[col].quantile(0.99)
                    q01 = features[col].quantile(0.01)
                    if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                        features[col] = features[col].clip(q01, q99)

            # Verificar que tenemos exactamente 66 features
            if len(features.columns) != 66:
                logger.warning(f"Features creados: {len(features.columns)}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            logger.info(f"✅ Features calculadas (legacy corregido): {len(features.columns)} features")
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
            # Crear features
            features = self.create_features(market_data)

            # Verificar que tenemos suficientes datos
            if len(features) < self.sequence_lengths[symbol]:  # Secuencia mínima requerida
                logger.warning(f"Datos insuficientes para {symbol}: {len(features)} < {self.sequence_lengths[symbol]}")
                return None

            # Seleccionar features utilizadas en entrenamiento
            feature_cols = self.feature_columns[symbol]
            features_selected = features[feature_cols].iloc[-self.sequence_lengths[symbol]:]  # Últimas observaciones

            # Normalizar con el scaler entrenado
            features_scaled = self.scalers[symbol].transform(features_selected)

            # Crear secuencia para el modelo
            sequence = features_scaled.reshape(1, self.sequence_lengths[symbol], len(feature_cols))

            # Realizar predicción
            prediction = self.models[symbol].predict(sequence, verbose=0)
            probabilities = prediction[0]

            # Interpretar resultado
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))

            # ✅ CORREGIDO: Unificar interpretación de clases a [SELL, HOLD, BUY] para coincidir con el entrenamiento
            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]

            # Información adicional
            current_price = float(market_data['close'].iloc[-1])
            model_accuracy = self.model_stats[symbol]['accuracy']

            result = {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'current_price': current_price,
                'model_accuracy': model_accuracy,
                'threshold_used': self.thresholds[symbol],
                'timestamp': datetime.now().isoformat(),
                'features_count': len(feature_cols)
            }

            logger.info(f"🎯 {symbol}: {signal} (conf: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error en predicción {symbol}: {e}")
            return None

    def predict_all_symbols(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Realizar predicciones para todos los símbolos

        Args:
            market_data_dict: Dict con datos de mercado por símbolo

        Returns:
            Dict con predicciones por símbolo
        """
        predictions = {}

        for symbol in self.symbols:
            if symbol in market_data_dict:
                prediction = self.predict(symbol, market_data_dict[symbol])
                if prediction:
                    predictions[symbol] = prediction
            else:
                logger.warning(f"Datos no disponibles para {symbol}")

        return predictions

    def predict_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Método de compatibilidad para integración con sistema principal
        Obtiene datos de Binance y realiza predicción con carga lazy
        """
        try:
            # ✅ OPTIMIZADO: Carga lazy del modelo
            if not self._load_model_lazy(symbol):
                logger.error(f"No se pudo cargar modelo para {symbol}")
                return None

            # Importar cliente de Binance
            import requests

            # Obtener datos de Binance
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': 100
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                logger.error(f"Error obteniendo datos de Binance para {symbol}")
                return None

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
            return self.predict(symbol, df)

        except Exception as e:
            logger.error(f"Error en predict_symbol para {symbol}: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Obtener información de los modelos cargados"""
        info = {
            'models_loaded': len(self.models),
            'symbols': list(self.models.keys()),
            'model_stats': self.model_stats,
            'thresholds': self.thresholds,
            'total_parameters': sum([model.count_params() for model in self.models.values()]),
            'load_timestamp': datetime.now().isoformat()
        }
        return info

    def predict_signal(self, symbol: str) -> Dict:
        """🎯 Predecir señal de trading usando TCN"""
        try:
            # Verificar que el modelo existe
            if symbol not in self.models:
                return {'signal': 'HOLD', 'confidence': 0.0, 'error': f'Modelo no disponible para {symbol}'}

            # Obtener datos de mercado
            market_data = self._get_market_data(symbol)
            if market_data is None or len(market_data) < self.sequence_lengths[symbol]:
                return {'signal': 'HOLD', 'confidence': 0.0, 'error': 'Datos insuficientes'}

            # Calcular features
            features_df = self._calculate_features_legacy_corrected(market_data, symbol)
            if features_df is None or len(features_df) < self.sequence_lengths[symbol]:
                return {'signal': 'HOLD', 'confidence': 0.0, 'error': 'Features insuficientes'}

            # Preparar datos para predicción
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            feature_columns = self.feature_columns[symbol]

            # Seleccionar y escalar features
            X = features_df[feature_columns].values
            X_scaled = scaler.transform(X)
            X_sequence = X_scaled[-self.sequence_lengths[symbol]:].reshape(1, self.sequence_lengths[symbol], len(feature_columns))

            # Hacer predicción
            prediction = model.predict(X_sequence, verbose=0)[0]

            # Aplicar class weights si están disponibles
            if symbol in self.class_weights:
                class_weights = self.class_weights[symbol]
                weighted_prediction = prediction * np.array([class_weights.get(i, 1.0) for i in range(len(prediction))])
                weighted_prediction = weighted_prediction / np.sum(weighted_prediction)
            else:
                weighted_prediction = prediction

            # Determinar señal y confianza
            signal_idx = np.argmax(weighted_prediction)
            confidence = float(weighted_prediction[signal_idx])

            # ✅ CORREGIDO: El orden de salida del modelo es [SELL, HOLD, BUY] para coincidir con el entrenamiento
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal = signal_map[signal_idx]

            # 🔧 FILTRO DE CORDURA: Validar predicción contra indicadores técnicos básicos
            sanity_check_result = self._sanity_check_prediction(features_df, signal, confidence, symbol)
            if sanity_check_result['override']:
                logger.warning(f"⚠️ FILTRO DE CORDURA: {sanity_check_result['reason']}")
                signal = sanity_check_result['corrected_signal']
                confidence = sanity_check_result['corrected_confidence']

            # Log de la predicción
            # ✅ CORREGIDO: El orden de las probabilidades debe coincidir con el signal_map
            probabilities = {
                'SELL': float(weighted_prediction[0]),
                'HOLD': float(weighted_prediction[1]),
                'BUY': float(weighted_prediction[2])
            }

            logger.info(f"🎯 {symbol}: {signal} (conf: {confidence:.3f})")

            return {
                'signal': signal,
                'confidence': confidence,
                'probabilities': probabilities,
                'current_price': float(market_data['close'].iloc[-1]),
                'rsi': float(features_df['rsi_14'].iloc[-1]),
                'macd': float(features_df['macd'].iloc[-1])
            }

        except Exception as e:
            logger.error(f"❌ Error en predicción para {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': str(e)}

    def _sanity_check_prediction(self, features_df: pd.DataFrame, signal: str, confidence: float, symbol: str) -> Dict:
        """🔍 Filtro de cordura para validar predicciones contra indicadores técnicos básicos"""
        try:
            # Obtener valores de indicadores técnicos de la última fila
            last_row = features_df.iloc[-1]

            rsi = last_row['rsi_14']
            macd = last_row['macd']
            stoch_k = last_row['stoch_k']
            uptrend = last_row.get('uptrend_strength', 0.5)
            downtrend = last_row.get('downtrend_strength', 0.5)

            # Contadores de señales técnicas
            buy_signals = 0
            sell_signals = 0

            # Análisis RSI
            if rsi < 30:
                buy_signals += 2  # RSI oversold = fuerte señal de compra
            elif rsi > 70:
                sell_signals += 2  # RSI overbought = fuerte señal de venta
            elif 30 <= rsi <= 45:
                buy_signals += 1  # RSI bajo-neutral = señal débil de compra
            elif 55 <= rsi <= 70:
                sell_signals += 1  # RSI alto-neutral = señal débil de venta

            # Análisis MACD (más estricto)
            if macd > 0.1:
                buy_signals += 2  # MACD fuertemente positivo
            elif macd > 0:
                buy_signals += 1  # MACD ligeramente positivo
            elif macd < -0.2:
                sell_signals += 2  # MACD fuertemente negativo
            else:
                sell_signals += 1  # MACD ligeramente negativo

            # Análisis Stochastic
            if stoch_k < 20:
                buy_signals += 1
            elif stoch_k > 80:
                sell_signals += 2  # Stoch overbought = señal fuerte de venta

            # Análisis de tendencia
            if uptrend > downtrend + 0.2:  # Uptrend dominante
                buy_signals += 1
            elif downtrend > uptrend + 0.2:  # Downtrend dominante
                sell_signals += 1

            # Determinar señal técnica dominante
            if buy_signals > sell_signals + 2:
                technical_signal = 'BUY'
            elif sell_signals > buy_signals + 2:
                technical_signal = 'SELL'
            else:
                technical_signal = 'HOLD'

            # Verificar contradicciones graves (umbral más bajo para ETH)
            contradiction_threshold = 0.65 if symbol == 'ETHUSDT' else 0.75

            # Caso 1: Modelo dice BUY fuerte pero indicadores dicen SELL
            if (signal == 'BUY' and confidence > contradiction_threshold and
                technical_signal == 'SELL' and sell_signals >= 3):
                return {
                    'override': True,
                    'reason': f"BUY {confidence:.1%} contradice indicadores técnicos (RSI:{rsi:.1f}, MACD:{macd:.3f}, Stoch:{stoch_k:.1f})",
                    'corrected_signal': 'HOLD',
                    'corrected_confidence': 0.55
                }

            # Caso 2: MACD fuertemente negativo + BUY fuerte (específico para ETH)
            if (symbol == 'ETHUSDT' and signal == 'BUY' and confidence > 0.8 and macd < -0.25):
                return {
                    'override': True,
                    'reason': f"BUY {confidence:.1%} con MACD negativo ({macd:.3f}) - señal contradictoria",
                    'corrected_signal': 'HOLD',
                    'corrected_confidence': 0.6
                }

            # Caso 3: Modelo dice SELL fuerte pero indicadores dicen BUY
            if (signal == 'SELL' and confidence > contradiction_threshold and
                technical_signal == 'BUY' and buy_signals >= 4):
                return {
                    'override': True,
                    'reason': f"SELL {confidence:.1%} contradice indicadores técnicos (RSI:{rsi:.1f}, MACD:{macd:.3f}, Stoch:{stoch_k:.1f})",
                    'corrected_signal': 'HOLD',
                    'corrected_confidence': 0.6
                }

            # Caso 4: Confianza extrema (>85%) que contradice indicadores básicos
            if confidence > 0.85:
                if ((signal == 'BUY' and sell_signals > buy_signals) or
                    (signal == 'SELL' and buy_signals > sell_signals)):
                    return {
                        'override': True,
                        'reason': f"Confianza extrema {confidence:.1%} para {signal} no justificada por indicadores",
                        'corrected_signal': technical_signal if technical_signal != 'HOLD' else 'HOLD',
                        'corrected_confidence': min(0.7, confidence * 0.8)
                    }

            # No hay contradicción grave
            return {'override': False}

        except Exception as e:
            logger.warning(f"⚠️ Error en filtro de cordura para {symbol}: {e}")
            return {'override': False}

    def _get_market_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """📊 Obtener datos de mercado de Binance"""
        try:
            import requests
            from datetime import datetime, timedelta

            # URL de la API de Binance
            url = "https://api.binance.com/api/v3/klines"

            # Parámetros para obtener datos de 1 minuto
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': limit
            }

            # Hacer la petición
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Procesar datos
            data = response.json()

            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convertir tipos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado para {symbol}: {e}")
            return None

    def _calculate_features_legacy_corrected(self, market_data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """🔧 Calcular features usando el método legacy corregido"""
        try:
            # Usar el método create_features existente
            features_df = self.create_features(market_data)

            if features_df.empty:
                logger.error(f"No se pudieron calcular features para {symbol}")
                return None

            # Limpiar datos
            features_df = features_df.dropna()


            return features_df

        except Exception as e:
            logger.error(f"Error calculando features para {symbol}: {e}")
            return None

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

        try:
            self.models_loading.add(symbol)
            logger.info(f"⚡ Carga lazy iniciada para {symbol}...")

            success = self._load_model_for_symbol(symbol)

            if success:
                self.models_loaded.add(symbol)
                logger.info(f"✅ {symbol} cargado exitosamente (lazy)")
                return True
            else:
                logger.error(f"❌ Error cargando {symbol}")
                return False

        except Exception as e:
            logger.error(f"❌ Error en carga lazy de {symbol}: {e}")
            return False
        finally:
            self.models_loading.discard(symbol)

# Función de utilidad para testing
def test_definitivo_predictor():
    """Test del predictor definitivo"""
    print("🧪 Testing TCN Definitivo Predictor...")

    predictor = TCNDefinitivoPredictor()

    # Cargar modelos
    if predictor.load_all_models():
        print("✅ Todos los modelos cargados correctamente")

        # Mostrar información
        info = predictor.get_model_info()
        print(f"📊 Modelos cargados: {info['models_loaded']}")
        print(f"🎯 Símbolos: {info['symbols']}")
        print(f"🧠 Parámetros totales: {info['total_parameters']:,}")

        # Generar datos de prueba
        print("\n🔄 Generando datos de prueba...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')

        test_data = {}
        for symbol in predictor.symbols:
            # Simular datos OHLCV
            base_price = {'BTCUSDT': 45000, 'ETHUSDT': 3000, 'BNBUSDT': 400}[symbol]
            returns = np.random.normal(0, 0.01, 100)
            prices = base_price * np.exp(np.cumsum(returns))

            test_data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, 100)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, 100)
            }, index=dates)

        # Realizar predicciones
        predictions = predictor.predict_all_symbols(test_data)

        print(f"\n🎯 Predicciones realizadas: {len(predictions)}")
        for symbol, result in predictions.items():
            print(f"  {symbol}: {result['signal']} (conf: {result['confidence']:.3f})")

        return True
    else:
        print("❌ Error cargando modelos")
        return False

if __name__ == "__main__":
    test_definitivo_predictor()
