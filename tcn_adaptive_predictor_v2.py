#!/usr/bin/env python3
"""
🎯 TCN ADAPTIVE PREDICTOR V2 - COMPATIBLE CON ENTRENADOR V2
Predictor mejorado compatible con thresholds adaptativos y nuevos símbolos
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
import talib
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Importar ambos motores de features
from centralized_features_engine2 import CentralizedFeaturesEngine
from centralized_features_engine_optimized import OptimizedFeaturesEngine

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCNAdaptivePredictorV2:
    """
    🎯 Predictor V2 HÍBRIDO - Compatible con ambos tipos de modelos
    - Modelos existentes: Motor centralizado (66 features)
    - Modelos direccionales: Motor optimizado (12 features)
    - Detección automática del tipo de modelo
    - Thresholds adaptativos basados en ATR
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.class_weights = {}
        self.model_types = {}  # Nuevo: Tipo de modelo (legacy/directional)
        
        # ✅ NUEVOS SÍMBOLOS SOPORTADOS (igual que entrenador v2)
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'ADAUSDT', 'DOTUSDT']
        
        # ✅ AMBOS MOTORES DE FEATURES
        self.features_engine_legacy = CentralizedFeaturesEngine()
        self.features_engine_optimized = OptimizedFeaturesEngine()

        # 🎯 THRESHOLDS FIJOS MÁS CONSERVADORES (compatibilidad con entrenador v2)
        self.fixed_thresholds = {
            'BTCUSDT': {'sell': -0.0010, 'buy': 0.0010},  # Reducido de ±0.0014 a ±0.0010
            'ETHUSDT': {'sell': -0.0006, 'buy': 0.0006},  # Reducido de ±0.0009 a ±0.0006
            'BNBUSDT': {'sell': -0.0006, 'buy': 0.0006},  # Reducido de ±0.0009 a ±0.0006
            'XRPUSDT': {'sell': -0.0006, 'buy': 0.0006},  # Reducido de ±0.0009 a ±0.0006
            'SOLUSDT': {'sell': -0.0012, 'buy': 0.0012},  # Reducido de ±0.0018 a ±0.0012
            'DOGEUSDT': {'sell': -0.0012, 'buy': 0.0012}, # Reducido de ±0.0018 a ±0.0012
            'ADAUSDT': {'sell': -0.0012, 'buy': 0.0012},  # Reducido de ±0.0018 a ±0.0012
            'DOTUSDT': {'sell': -0.0012, 'buy': 0.0012},  # Reducido de ±0.0018 a ±0.0012
        }

        # 🔧 CONFIGURACIÓN DE MODELOS (igual que entrenador v2)
        self.sequence_lengths = {
            'BTCUSDT': 24, 'ETHUSDT': 24, 'BNBUSDT': 24, 'XRPUSDT': 24,
            'SOLUSDT': 24, 'DOGEUSDT': 24, 'ADAUSDT': 24, 'DOTUSDT': 24
        }

        self.n_features = 66

        # ✅ NUEVO: Configuración de thresholds adaptativos
        self.use_adaptive_thresholds = True

        # ✅ Tracking de modelos cargados
        self.models_loaded = set()
        self.models_loading = set()

        print("🎯 TCN Adaptive Predictor V2 HÍBRIDO inicializado")
        print(f"📊 Símbolos soportados: {self.symbols}")
        print("🔧 Motores de features:")
        print("   • Motor Legacy: 66 features (modelos existentes)")
        print("   • Motor Optimizado: 12 features direccionales (modelos nuevos)")
        print("🤖 Detección automática del tipo de modelo")
        print("🛡️ Configuración conservadora (relajada 5%)")
        print("⚡ Modelos se cargarán bajo demanda")

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        🎯 Calcular thresholds adaptativos basados en ATR
        MISMA LÓGICA que entrenador v2
        """
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]
        
        try:
            # Calcular ATR para volatilidad adaptativa
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            
            # ATR de 14 períodos
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Promedio de ATR reciente (últimas 50 velas)
            avg_atr = np.nanmean(atr_14[-50:]) if len(atr_14) > 50 else np.nanmean(atr_14)
            avg_price = np.mean(close_prices[-50:]) if len(close_prices) > 50 else np.mean(close_prices)
            
            # ATR como porcentaje del precio
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02
            
            # Thresholds adaptativos basados en ATR (MÁS CONSERVADORES)
            base_threshold = atr_percent * 0.3  # Factor más conservador (reducido de 0.5 a 0.3)
            
            adaptive_thresholds = {
                'sell': -base_threshold * 1.2,  # Reducido de 1.5 a 1.2
                'buy': base_threshold * 1.2     # Reducido de 1.5 a 1.2
            }
            
            logger.info(f"🎯 {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%)")
            logger.info(f"   📊 Thresholds: Buy {adaptive_thresholds['buy']:.4f}, Sell {adaptive_thresholds['sell']:.4f}")
            
            return adaptive_thresholds
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando thresholds adaptativos para {symbol}: {e}")
            logger.warning(f"   🔄 Usando thresholds fijos como fallback")
            return self.fixed_thresholds[symbol]

    def load_all_models(self):
        """🎯 Cargar todos los modelos disponibles"""
        logger.info("🔄 Cargando todos los modelos adaptativos v2...")
        success_count = 0

        for symbol in self.symbols:
            if symbol not in self.models_loaded:
                success = self._load_model_for_symbol(symbol)
                if success:
                    self.models_loaded.add(symbol)
                    success_count += 1

        logger.info(f"🎉 {success_count}/{len(self.symbols)} modelos cargados correctamente")
        return success_count > 0

    def _load_model_for_symbol(self, symbol: str) -> bool:
        """Cargar modelo con fallback automático: Direccionales -> Definitivos -> Legacy"""

        # 🔧 SIMPLIFICAR: Evitar problemas de concurrencia
        if symbol in self.models_loading:
            logger.warning(f"⚠️ {symbol} ya está siendo cargado por otro proceso")
            # Limpiar estado para permitir nueva carga
            self.models_loading.discard(symbol)

        self.models_loading.add(symbol)

        try:
            # 🎯 ESTRATEGIA DE FALLBACK HÍBRIDA
            model_directories = [
                # PRIORIDAD 1: Modelos direccionales optimizados (NUEVOS)
                f"models/directional_v1_{symbol.lower()}",
                
                # PRIORIDAD 2: Modelos definitivos recientes
                f"models/definitivo_v3_{symbol.lower()}",
                
                # PRIORIDAD 3: Modelos adaptativos mejorados
                f"models/adaptive_v3_improved_{symbol.lower()}",
                f"models/adaptive_v2_optimized_{symbol.lower()}",
                f"models/adaptive_v2_{symbol.lower()}",
                
                # PRIORIDAD 4: Modelos legacy (fallback)
                f"models/adaptive_{symbol.lower()}",
                f"models/definitivo_{symbol.lower()}",
            ]
            
            model_loaded = False
            used_directory = None

            for model_dir in model_directories:
                model_path = os.path.join(model_dir, "best_model.h5")
                scaler_path = os.path.join(model_dir, "scaler.pkl")

                # Verificar si esta versión existe
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    logger.info(f"  📂 Intentando cargar desde: {model_dir}")
                    used_directory = model_dir
                    
                    try:
                        # Cargar modelo
                        logger.info(f"  📂 Cargando modelo {symbol}...")
                        self.models[symbol] = keras.models.load_model(model_path, compile=False)
                        logger.info(f"  ✅ Modelo {symbol} cargado")

                        # Cargar scaler
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)
                        logger.info(f"  📊 Scaler {symbol} cargado")

                        # 🎯 DETECTAR TIPO DE MODELO
                        if "directional_v1" in model_dir:
                            self.model_types[symbol] = "directional"
                            logger.info(f"  🎯 Tipo: DIRECCIONAL (motor optimizado)")
                        else:
                            self.model_types[symbol] = "legacy"
                            logger.info(f"  🔧 Tipo: LEGACY (motor centralizado)")

                        model_loaded = True
                        break

                    except Exception as load_error:
                        logger.warning(f"  ⚠️ Error cargando desde {model_dir}: {load_error}")
                        continue

                else:
                    logger.debug(f"  ⏭️ No encontrado: {model_dir}")

            if not model_loaded:
                logger.error(f"  ❌ No se encontraron modelos para {symbol} en ninguna ubicación")
                return False

            # ✅ MODELO CARGADO EXITOSAMENTE - Cargar componentes adicionales
            if used_directory is not None:
                logger.info(f"  🎉 Modelo {symbol} cargado desde: {used_directory}")

                # Cargar features si existen
                features_path = os.path.join(used_directory, "feature_columns.pkl")
                if os.path.exists(features_path):
                    try:
                        with open(features_path, 'rb') as f:
                            self.feature_columns[symbol] = pickle.load(f)
                        logger.info(f"  📋 Features cargadas: {len(self.feature_columns[symbol])} columnas")
                    except Exception as features_error:
                        logger.warning(f"  ⚠️ Error cargando features {symbol}: {features_error}")
                        self.feature_columns[symbol] = None
                else:
                    logger.info(f"  ℹ️ Features no encontradas en {used_directory}")
                    self.feature_columns[symbol] = None

                # Cargar class weights si existen
                weights_path = os.path.join(used_directory, "class_weights.pkl")
                if os.path.exists(weights_path):
                    try:
                        with open(weights_path, 'rb') as f:
                            self.class_weights[symbol] = pickle.load(f)
                        logger.info(f"  ⚖️ Class weights {symbol} cargados")
                    except Exception:
                        self.class_weights[symbol] = None
                else:
                    self.class_weights[symbol] = None

                # 🎯 INFORMACIÓN DE VERSIÓN DETECTADA
                if "directional_v1" in used_directory:
                    logger.info(f"  🌟 {symbol}: Usando modelo DIRECCIONAL V1 (optimizado)")
                elif "adaptive_v3_improved" in used_directory:
                    logger.info(f"  🚀 {symbol}: Usando modelo V3 MEJORADO")
                elif "adaptive_v2_optimized" in used_directory:
                    logger.info(f"  🎯 {symbol}: Usando modelo V2 OPTIMIZADO")
                elif "adaptive_v2" in used_directory:
                    logger.info(f"  🔧 {symbol}: Usando modelo V2")
                elif "adaptive" in used_directory:
                    logger.info(f"  🔄 {symbol}: Usando modelo V1 (fallback)")

            return True

        except Exception as e:
            logger.error(f"❌ Error general cargando modelo {symbol}: {e}")
            return False
        finally:
            self.models_loading.discard(symbol)

    def create_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        🔧 Crear features usando el motor apropiado según el tipo de modelo
        """
        try:
            # Determinar qué motor usar basado en el tipo de modelo
            model_type = self.model_types.get(symbol, "legacy")
            
            if model_type == "directional":
                # Usar motor optimizado para modelos direccionales
                features = self.features_engine_optimized.calculate_features(df, feature_set='directional_only')
                feature_count = 12
                motor_type = "DIRECCIONAL"
            else:
                # Usar motor legacy para modelos existentes
                features = self.features_engine_legacy.calculate_features(df, feature_set='tcn_definitivo')
                feature_count = 66
                motor_type = "LEGACY"
            
            if features.empty:
                logger.error(f"❌ Error: Motor {motor_type} devolvió DataFrame vacío")
                return pd.DataFrame()
            
            logger.info(f"✅ {len(features.columns)} features {motor_type} creados para {symbol}")
            return features

        except Exception as e:
            logger.error(f"❌ Error usando motor de features para {symbol}: {e}")
            return pd.DataFrame()

    def predict(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        🎯 Realizar predicción con thresholds adaptativos
        COMPATIBLE con entrenador v2
        """
        if symbol not in self.models:
            logger.error(f"Modelo no cargado para {symbol}")
            return None

        try:
            # Crear features usando el motor apropiado
            features = self.create_features(market_data, symbol)

            # Verificar que tenemos suficientes datos
            sequence_length = self.sequence_lengths[symbol]
            if len(features) < sequence_length:
                logger.warning(f"Datos insuficientes para {symbol}: {len(features)} < {sequence_length}")
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'predicted_return': 0.0,
                    'reason': 'Datos insuficientes para predicción'
                }

            # Tomar las últimas sequence_length muestras
            recent_features = features.iloc[-sequence_length:].values

            # Verificar y manejar NaN/inf
            if np.any(np.isnan(recent_features)) or np.any(np.isinf(recent_features)):
                logger.warning(f"Features contienen NaN/inf para {symbol}")
                recent_features = np.nan_to_num(recent_features, nan=0.0, posinf=0.0, neginf=0.0)

            # 🎯 VERIFICAR DIMENSIONES SEGÚN TIPO DE MODELO
            model_type = self.model_types.get(symbol, "legacy")
            expected_features = 12 if model_type == "directional" else self.n_features
            actual_features = recent_features.shape[1]

            if actual_features != expected_features:
                logger.error(f"Dimensión incorrecta para modelo {model_type}: esperado {expected_features}, actual {actual_features}")
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

            # ✅ NUEVO: Calcular thresholds adaptativos
            thresholds = self.calculate_adaptive_thresholds(market_data, symbol)

            # 🎯 LÓGICA CORREGIDA: Respetar la mayor probabilidad con filtros de calidad
            max_prob = max(buy_prob, sell_prob, hold_prob)
            second_max = sorted([buy_prob, sell_prob, hold_prob])[-2]
            prob_margin = max_prob - second_max

            # Determinar qué acción tiene la mayor probabilidad
            if buy_prob == max_prob:
                action = 'BUY'
                confidence = buy_prob
                predicted_return = thresholds['buy']
                
                # ✅ FILTROS DE CALIDAD OPCIONALES (relajados un 5%)
                if buy_prob < 0.43:  # Confianza mínima relajada a 43% (era 45%)
                    logger.info(f"⚠️ {symbol}: Señal BUY con baja confianza {buy_prob:.1%}")
                if prob_margin < 0.047:  # Margen mínimo relajado a 4.7% (era 5%)
                    logger.info(f"⚠️ {symbol}: Señal BUY con margen pequeño {prob_margin:.1%}")
                    
            elif sell_prob == max_prob:
                action = 'SELL'
                confidence = sell_prob
                predicted_return = thresholds['sell']
                
                # ✅ FILTROS DE CALIDAD OPCIONALES (relajados un 5%)
                if sell_prob < 0.43:  # Confianza mínima relajada a 43% (era 45%)
                    logger.info(f"⚠️ {symbol}: Señal SELL con baja confianza {sell_prob:.1%}")
                if prob_margin < 0.047:  # Margen mínimo relajado a 4.7% (era 5%)
                    logger.info(f"⚠️ {symbol}: Señal SELL con margen pequeño {prob_margin:.1%}")
                    
            else:  # hold_prob == max_prob
                action = 'HOLD'
                confidence = hold_prob
                predicted_return = 0.0
                logger.info(f"📊 {symbol}: HOLD tiene mayor probabilidad {hold_prob:.1%}")

            return {
                'action': action,
                'confidence': float(confidence),
                'predicted_return': float(predicted_return),
                'probabilities': {
                    'SELL': float(sell_prob),
                    'HOLD': float(hold_prob),
                    'BUY': float(buy_prob)
                },
                'thresholds': thresholds,
                'adaptive_thresholds': self.use_adaptive_thresholds,
                'features_count': actual_features,
                'sequence_length': sequence_length,
                'model_type': model_type,
                'motor_features': "direccional" if model_type == "directional" else "legacy",
                'reason': f'Predicción TCN V2 {symbol} ({model_type}) - Thresholds {"adaptativos" if self.use_adaptive_thresholds else "fijos"}'
            }

        except Exception as e:
            logger.error(f"Error en predicción para {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def predict_symbol(self, symbol: str) -> Optional[Dict]:
        """
        🎯 Método de compatibilidad para integración con sistema principal
        Obtiene datos de Binance y realiza predicción con carga lazy
        """
        try:
            # ✅ Carga lazy del modelo
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
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            df = pd.DataFrame(klines, columns=columns)

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
                'thresholds': prediction.get('thresholds', {}),
                'adaptive': prediction.get('adaptive_thresholds', False),
                'error': None
            }

        except Exception as e:
            logger.error(f"Error en predict_symbol para {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': str(e)}

    def _load_model_lazy(self, symbol: str) -> bool:
        """🚀 Cargar modelo bajo demanda (lazy loading) - MEJORADO"""
        if symbol in self.models_loaded:
            return True  # Ya está cargado

        # 🔧 MEJORAR: Evitar deadlocks con mejor manejo de concurrencia
        if symbol in self.models_loading:
            logger.warning(f"⏳ {symbol} ya se está cargando por otro proceso...")
            # En lugar de esperar indefinidamente, intentar cargar nosotros mismos
            # Esto evita deadlocks cuando múltiples procesos intentan cargar simultáneamente
            logger.info(f"🔄 {symbol}: Intentando carga directa en lugar de esperar...")
            # Quitar de la lista de carga para intentar nosotros
            self.models_loading.discard(symbol)

        if symbol not in self.symbols:
            logger.error(f"❌ {symbol} no está en la lista de símbolos soportados")
            return False

        # Verificar si el directorio del modelo existe antes de intentar cargar
        model_found = False
        model_directories = [
            f"models/definitivo_v3_{symbol.lower()}",
            f"models/adaptive_v3_improved_{symbol.lower()}",
            f"models/adaptive_v2_optimized_{symbol.lower()}",
            f"models/adaptive_v2_{symbol.lower()}",
            f"models/adaptive_{symbol.lower()}",
            f"models/definitivo_{symbol.lower()}",
        ]
        
        for model_dir in model_directories:
            model_path = os.path.join(model_dir, "best_model.h5")
            if os.path.exists(model_path):
                model_found = True
                logger.info(f"🔍 {symbol}: Modelo encontrado en {model_dir}")
                break
        
        if not model_found:
            logger.error(f"❌ {symbol}: No se encontró ningún modelo en las ubicaciones esperadas")
            return False

        # Cargar modelo
        success = self._load_model_for_symbol(symbol)
        if success:
            self.models_loaded.add(symbol)
            logger.info(f"✅ {symbol}: Modelo cargado exitosamente (lazy)")
        else:
            logger.error(f"❌ {symbol}: Error cargando modelo (lazy)")

        return success

    def get_available_symbols(self) -> list:
        """Obtener lista de símbolos disponibles"""
        return [symbol for symbol in self.symbols if symbol in self.models_loaded]

    def is_model_loaded(self, symbol: str) -> bool:
        """Verificar si un modelo está cargado"""
        return symbol in self.models_loaded

    def set_adaptive_thresholds(self, enabled: bool):
        """🎯 Activar/desactivar thresholds adaptativos"""
        self.use_adaptive_thresholds = enabled
        logger.info(f"🎯 Thresholds adaptativos: {'ACTIVADOS' if enabled else 'DESACTIVADOS'}")

    def get_model_info(self, symbol: str) -> Dict:
        """🔍 Obtener información detallada del modelo cargado"""
        if symbol not in self.models_loaded:
            return {'error': f'Modelo {symbol} no está cargado'}
        
        model_type = self.model_types.get(symbol, "unknown")
        return {
            'symbol': symbol,
            'model_type': model_type,
            'motor_features': "direccional" if model_type == "directional" else "legacy",
            'features_count': 12 if model_type == "directional" else 66,
            'is_loaded': symbol in self.models_loaded,
            'has_scaler': symbol in self.scalers,
            'has_feature_columns': symbol in self.feature_columns,
            'sequence_length': self.sequence_lengths.get(symbol, 24)
        }

    def list_loaded_models(self) -> Dict:
        """📋 Listar todos los modelos cargados con su tipo"""
        loaded_models = {}
        for symbol in self.models_loaded:
            loaded_models[symbol] = self.get_model_info(symbol)
        return loaded_models

    def get_directional_models(self) -> List[str]:
        """🎯 Obtener lista de modelos direccionales cargados"""
        return [symbol for symbol, model_type in self.model_types.items() 
                if model_type == "directional" and symbol in self.models_loaded]

    def get_legacy_models(self) -> List[str]:
        """🔧 Obtener lista de modelos legacy cargados"""
        return [symbol for symbol, model_type in self.model_types.items() 
                if model_type == "legacy" and symbol in self.models_loaded]

    def compare_model_types(self) -> Dict:
        """📊 Comparar estadísticas de ambos tipos de modelos"""
        directional_count = len(self.get_directional_models())
        legacy_count = len(self.get_legacy_models())
        total_count = len(self.models_loaded)
        
        return {
            'total_models': total_count,
            'directional_models': {
                'count': directional_count,
                'symbols': self.get_directional_models(),
                'percentage': (directional_count / total_count * 100) if total_count > 0 else 0
            },
            'legacy_models': {
                'count': legacy_count,
                'symbols': self.get_legacy_models(),
                'percentage': (legacy_count / total_count * 100) if total_count > 0 else 0
            }
        }

# Instancia global para uso en otros módulos
predictor_v2 = None

def get_predictor_v2():
    """Obtener instancia del predictor v2"""
    global predictor_v2
    if predictor_v2 is None:
        predictor_v2 = TCNAdaptivePredictorV2()
    return predictor_v2

if __name__ == "__main__":
    # Test del predictor v2 híbrido
    print("🧪 TESTING TCN ADAPTIVE PREDICTOR V2 HÍBRIDO...")
    predictor = TCNAdaptivePredictorV2()
    
    # Test carga de modelos
    predictor.load_all_models()
    
    # Mostrar estadísticas de modelos cargados
    comparison = predictor.compare_model_types()
    print(f"\n📊 ESTADÍSTICAS DE MODELOS:")
    print(f"   Total cargados: {comparison['total_models']}")
    print(f"   Direccionales: {comparison['directional_models']['count']} ({comparison['directional_models']['percentage']:.1f}%)")
    print(f"   Legacy: {comparison['legacy_models']['count']} ({comparison['legacy_models']['percentage']:.1f}%)")
    
    if comparison['directional_models']['symbols']:
        print(f"   🎯 Direccionales: {', '.join(comparison['directional_models']['symbols'])}")
    if comparison['legacy_models']['symbols']:
        print(f"   🔧 Legacy: {', '.join(comparison['legacy_models']['symbols'])}")
    
    # Test con BTC
    print(f"\n🧪 Test predicción BTCUSDT...")
    result = predictor.predict_symbol('BTCUSDT')
    if result:
        print(f"📊 Resultado: {result['signal']} (confianza: {result['confidence']:.3f})")
        print(f"🎯 Tipo modelo: {result.get('model_type', 'unknown')}")
        print(f"🔧 Motor features: {result.get('motor_features', 'unknown')}")
        print(f"📈 Features count: {result.get('features_count', 'unknown')}")
        print(f"🎯 Adaptativos: {result.get('adaptive', False)}")
    else:
        print("❌ Error en predicción")
    
    # Test info detallada
    if 'BTCUSDT' in predictor.models_loaded:
        info = predictor.get_model_info('BTCUSDT')
        print(f"\n🔍 Info detallada BTCUSDT:")
        for key, value in info.items():
            print(f"   {key}: {value}") 