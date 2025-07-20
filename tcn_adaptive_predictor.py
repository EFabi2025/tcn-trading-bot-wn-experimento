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

# ✅ NUEVO: Importar motor de features centralizado (MISMO que entrenador)
from centralized_features_engine2 import CentralizedFeaturesEngine

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
        
        # ✅ NUEVO: Motor de features centralizado (MISMO que entrenador)
        self.features_engine = CentralizedFeaturesEngine()

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
            'BTCUSDT': {'sell': -0.004, 'buy': 0.004},   # 🚀 AGRESIVO: 0.3% para BUY (era 0.14%)
            'ETHUSDT': {'sell': -0.0012, 'buy': 0.0013},   # 🚀 AGRESIVO: 0.2% para BUY (era 0.09%)
            'BNBUSDT': {'sell': -0.0009, 'buy': 0.0015},   # 🚀 AGRESIVO: 0.3% para BUY (era 0.15%)
            'XRPUSDT': {'sell': -0.0018, 'buy': 0.0018}    # 🚀 AGRESIVO: 0.3% para BUY (era 0.11%)
        }

        # 🔧 SEQUENCE LENGTH DINÁMICO POR MODELO
        self.sequence_lengths = {
            'BTCUSDT': 48,  # Modelo antiguo
            'ETHUSDT': 24,  # Modelo reentrenado
            'BNBUSDT': 24,  # Modelo antiguo
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
            model_dir = f"models/adaptive_{symbol.lower()}"
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
        🔧 Crear features usando EXACTAMENTE el mismo motor que el entrenador
        ✅ SINCRONIZADO: Usa CentralizedFeaturesEngine2 (IGUAL que tcn_adaptive_trainer.py)
        """
        print("🔧 Creando 66 features técnicos (SINCRONIZADO)...")
        
        try:
            # ✅ USAR EXACTAMENTE EL MISMO MOTOR QUE EL ENTRENADOR
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
            
            if features.empty:
                logger.error("❌ Error: Motor de features devolvió DataFrame vacío")
                return pd.DataFrame()
            
            print(f"✅ 66 features técnicos creados (SINCRONIZADO)")
            return features

        except Exception as e:
            logger.error(f"❌ Error usando motor de features centralizado: {e}")
            return pd.DataFrame()

    def create_66_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 Alias para compatibilidad - usa el motor centralizado
        ✅ SINCRONIZADO: Usa CentralizedFeaturesEngine2 (IGUAL que tcn_adaptive_trainer.py)
        """
        return self.create_features(df)

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
