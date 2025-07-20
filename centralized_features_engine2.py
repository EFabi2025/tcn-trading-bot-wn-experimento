#!/usr/bin/env python3
"""
🎯 CENTRALIZED FEATURES ENGINE
=============================

Motor centralizado para cálculo de features técnicas.
Unifica todas las implementaciones del sistema usando TA-Lib.

Características:
- ✅ Implementación única y centralizada
- ✅ Usa TA-Lib para precisión matemática
- ✅ Compatible con entrenamiento y trading en vivo
- ✅ Soporte para múltiples conjuntos de features
- ✅ Validación automática de datos
"""

import numpy as np
import pandas as pd
try:
    import talib
except ImportError:
    print("⚠️ TA-Lib no disponible, usando implementaciones alternativas")
    talib = None

from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CentralizedFeaturesEngine:
    """
    Motor centralizado de features técnicas usando TA-Lib
    """

    def __init__(self):
        """Inicializar el motor de features"""
        self.feature_sets = {
            'tcn_definitivo': self._get_tcn_definitivo_features(),
            'tcn_final': self._get_tcn_final_features(),
            'full_set': self._get_full_features_set()
        }

        print("🎯 Centralized Features Engine inicializado")
        print(f"   📊 Conjuntos disponibles: {list(self.feature_sets.keys())}")
        for name, features in self.feature_sets.items():
            print(f"   🔧 {name}: {len(features)} features")

    def _get_tcn_definitivo_features(self) -> List[str]:
        """Features para modelos TCN definitivos (66 features EXACTAS del entrenador)"""
        return [
            # === MOMENTUM INDICATORS (15 features) ===
            'rsi_14', 'rsi_21', 'rsi_7',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
            'cci_14', 'cci_20',

            # === TREND INDICATORS (12 features) ===
            'sma_10', 'sma_20', 'sma_50',
            'ema_10', 'ema_20', 'ema_50',
            'adx_14', 'plus_di', 'minus_di',
            'psar', 'aroon_up', 'aroon_down',

            # === VOLATILITY INDICATORS (10 features) ===
            'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_position',
            'atr_14', 'atr_20', 'true_range',
            'natr_14', 'natr_20',

            # === VOLUME INDICATORS (8 features) ===
            'ad', 'adosc', 'obv',
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'mfi_14', 'mfi_20',

            # === PRICE PATTERNS (8 features) ===
            'hl_ratio', 'oc_ratio', 'price_position',
            'price_change_1', 'price_change_5', 'price_change_10',
            'volatility_10', 'volatility_20',

            # === MARKET STRUCTURE (8 features) ===
            'higher_high', 'lower_low',
            'uptrend_strength', 'downtrend_strength',
            'resistance_touch', 'support_touch',
            'efficiency_ratio', 'fractal_dimension',

            # === MOMENTUM DERIVATIVES (5 features) ===
            'rsi_momentum', 'macd_momentum', 'ad_momentum',
            'volume_momentum', 'price_acceleration'
        ]

    def _get_tcn_final_features(self) -> List[str]:
        """Features para modelos tcn_final (21 features simplificadas)"""
        return [
            # 1. OHLCV básicos (5 features)
            'open', 'high', 'low', 'close', 'volume',
            # 2. Returns múltiples períodos (5 features)
            'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
            # 3. Moving Averages (3 features)
            'sma_5', 'sma_20', 'ema_12',
            # 4. RSI (1 feature)
            'rsi_14',
            # 5. MACD completo (3 features)
            'macd', 'macd_signal', 'macd_histogram',
            # 6. Bollinger Bands (2 features)
            'bb_position', 'bb_width',
            # 7. Volume analysis (1 feature)
            'volume_ratio',
            # 8. Volatilidad (1 feature)
            'volatility'
        ]

    def _get_full_features_set(self) -> List[str]:
        """Conjunto completo de features disponibles"""
        tcn_def = self._get_tcn_definitivo_features()
        tcn_final = self._get_tcn_final_features()
        additional = ['returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
                     'sma_5', 'ema_12', 'bb_position', 'bb_width', 'volume_ratio', 'volatility']
        return list(set(tcn_def + tcn_final + additional))

    def calculate_features(self, df: pd.DataFrame, feature_set: str = 'tcn_definitivo') -> pd.DataFrame:
        """
        Calcular features técnicas usando TA-Lib
        
        Args:
            df: DataFrame con columnas OHLCV
            feature_set: Conjunto de features a calcular ('tcn_definitivo', 'tcn_final', 'full_set')
            
        Returns:
            DataFrame con features calculadas
        """
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set}' no disponible. Opciones: {list(self.feature_sets.keys())}")

        # Validar datos de entrada
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener columnas: {required_columns}")

        # Crear copia para trabajar
        features_df = df.copy()

        # Extraer arrays para TA-Lib
        open_prices = df['open'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume_data = df['volume'].values.astype(float)

        # Calcular todas las features disponibles
        features_df = self._calculate_all_talib_features(
            features_df, open_prices, high_prices, low_prices, close_prices, volume_data
        )

        # Calcular features adicionales no disponibles en TA-Lib
        features_df = self._calculate_additional_features(features_df)

        # Seleccionar solo las features del conjunto solicitado
        requested_features = self.feature_sets[feature_set]
        available_features = [f for f in requested_features if f in features_df.columns]

        if len(available_features) != len(requested_features):
            missing = set(requested_features) - set(available_features)
            print(f"⚠️ Features faltantes: {missing}")

        # Retornar solo las features solicitadas
        result_df = features_df[available_features].copy()

        # Limpiar datos
        result_df = self._clean_features_data(result_df)

        print(f"✅ Features calculadas: {len(result_df.columns)} de {len(requested_features)} solicitadas")
        return result_df

    def _calculate_all_talib_features(self, df: pd.DataFrame, open_arr: np.ndarray,
                                    high_arr: np.ndarray, low_arr: np.ndarray,
                                    close_arr: np.ndarray, volume_arr: np.ndarray) -> pd.DataFrame:
        """Calcular todas las features usando TA-Lib"""

        if talib is None:
            print("⚠️ TA-Lib no disponible, usando implementaciones manuales")
            return self._calculate_manual_features(df)

        try:
            # === MOMENTUM INDICATORS ===
            df['rsi_14'] = talib.RSI(close_arr, timeperiod=14)
            df['rsi_21'] = talib.RSI(close_arr, timeperiod=21)
            df['rsi_7'] = talib.RSI(close_arr, timeperiod=7)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_arr)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['williams_r'] = talib.WILLR(high_arr, low_arr, close_arr)

            # Rate of Change
            df['roc_10'] = talib.ROC(close_arr, timeperiod=10)
            df['roc_20'] = talib.ROC(close_arr, timeperiod=20)

            # Momentum
            df['momentum_10'] = talib.MOM(close_arr, timeperiod=10)
            df['momentum_20'] = talib.MOM(close_arr, timeperiod=20)

            # CCI
            df['cci_14'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)
            df['cci_20'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=20)

            # === TREND INDICATORS ===
            # Moving Averages
            df['sma_10'] = talib.SMA(close_arr, timeperiod=10)
            df['sma_20'] = talib.SMA(close_arr, timeperiod=20)
            df['sma_50'] = talib.SMA(close_arr, timeperiod=50)
            df['sma_5'] = talib.SMA(close_arr, timeperiod=5)

            df['ema_10'] = talib.EMA(close_arr, timeperiod=10)
            df['ema_20'] = talib.EMA(close_arr, timeperiod=20)
            df['ema_50'] = talib.EMA(close_arr, timeperiod=50)
            df['ema_12'] = talib.EMA(close_arr, timeperiod=12)

            # ADX
            df['adx_14'] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14)

            # PSAR
            df['psar'] = talib.SAR(high_arr, low_arr)

            # Aroon
            aroon_down, aroon_up = talib.AROON(high_arr, low_arr, timeperiod=14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down

            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower

            # ATR
            df['atr_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
            df['atr_20'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=20)
            df['natr_14'] = talib.NATR(high_arr, low_arr, close_arr, timeperiod=14)
            df['natr_20'] = talib.NATR(high_arr, low_arr, close_arr, timeperiod=20)
            df['true_range'] = talib.TRANGE(high_arr, low_arr, close_arr)

            # === VOLUME INDICATORS ===
            df['ad'] = talib.AD(high_arr, low_arr, close_arr, volume_arr)
            df['adosc'] = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr)
            df['obv'] = talib.OBV(close_arr, volume_arr)
            df['volume_sma_10'] = talib.SMA(volume_arr, timeperiod=10)
            df['volume_sma_20'] = talib.SMA(volume_arr, timeperiod=20)
            df['mfi_14'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)
            df['mfi_20'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=20)

            # === CYCLE INDICATORS ===
            df['ht_dcperiod'] = talib.HT_DCPERIOD(close_arr)
            df['ht_dcphase'] = talib.HT_DCPHASE(close_arr)
            inphase, quadrature = talib.HT_PHASOR(close_arr)
            df['ht_phasor_inphase'] = inphase
            df['ht_phasor_quadrature'] = quadrature

            # === STATISTICAL INDICATORS ===
            df['beta'] = talib.BETA(high_arr, low_arr, timeperiod=5)
            df['correl'] = talib.CORREL(high_arr, low_arr, timeperiod=30)
            df['linearreg'] = talib.LINEARREG(close_arr, timeperiod=14)
            df['linearreg_angle'] = talib.LINEARREG_ANGLE(close_arr, timeperiod=14)
            df['linearreg_intercept'] = talib.LINEARREG_INTERCEPT(close_arr, timeperiod=14)
            df['linearreg_slope'] = talib.LINEARREG_SLOPE(close_arr, timeperiod=14)

        except Exception as e:
            print(f"⚠️ Error calculando features TA-Lib: {e}")

        return df

    def _calculate_manual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implementaciones manuales básicas cuando TA-Lib no está disponible"""
        # RSI manual
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # SMA/EMA básicos
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # MACD básico
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df

    def _calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features adicionales no disponibles en TA-Lib"""

        try:
            # Returns múltiples períodos
            df['returns_1'] = df['close'].pct_change(periods=1)
            df['returns_3'] = df['close'].pct_change(periods=3)
            df['returns_5'] = df['close'].pct_change(periods=5)
            df['returns_10'] = df['close'].pct_change(periods=10)
            df['returns_20'] = df['close'].pct_change(periods=20)

            # Bollinger Bands adicionales
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_range = df['bb_upper'] - df['bb_lower']
                bb_range = bb_range.replace(0, 1e-8)
                df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range
                df['bb_width'] = bb_range / df['bb_middle'] if 'bb_middle' in df.columns else bb_range / df['close']

            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
            
            # Volume ratio (usando volume_sma_20 si está disponible)
            if 'volume_sma_20' in df.columns:
                volume_sma_safe = df['volume_sma_20'].replace(0, 1e-8)
                df['volume_ratio'] = df['volume'] / volume_sma_safe
            else:
                volume_sma_safe = df['volume_sma'].replace(0, 1e-8)
                df['volume_ratio'] = df['volume'] / volume_sma_safe
                
            df['volume_price_trend'] = df['volume'] * df['close'].pct_change()

            # Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std()

            # === NUEVAS FEATURES DEL TCN DEFINITIVO ===
            
            # PRICE PATTERNS (8 features)
            hl_range = df['high'] - df['low']
            hl_range = hl_range.replace(0, 1e-8)
            
            df['hl_ratio'] = hl_range / df['close']
            df['oc_ratio'] = (df['close'] - df['open']) / df['close']
            df['price_position'] = (df['close'] - df['low']) / hl_range
            
            # Price changes
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            
            # Volatility windows
            returns = pd.Series(np.log(df['close'] / df['close'].shift(1)), index=df.index)
            df['volatility_10'] = returns.rolling(10).std()
            df['volatility_20'] = returns.rolling(20).std()

            # MARKET STRUCTURE (8 features)
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            df['uptrend_strength'] = (df['close'] > df['close'].shift(1)).rolling(10).sum() / 10
            df['downtrend_strength'] = (df['close'] < df['close'].shift(1)).rolling(10).sum() / 10
            
            df['resistance_touch'] = (df['close'] >= df['close'].rolling(20).max() * 0.99).astype(int)
            df['support_touch'] = (df['close'] <= df['close'].rolling(20).min() * 1.01).astype(int)
            
            # Market efficiency
            close_diff_abs = pd.Series(np.abs(df['close'].diff()), index=df.index)
            df['efficiency_ratio'] = (np.abs(df['close'] - df['close'].shift(10)) /
                                      close_diff_abs.rolling(10).sum()).fillna(0)
            
            # Fractal dimension (simplified)
            df['fractal_dimension'] = 0.5  # Valor constante por ahora

            # MOMENTUM DERIVATIVES (5 features)
            if 'rsi_14' in df.columns:
                df['rsi_momentum'] = df['rsi_14'].diff().fillna(0)
            if 'macd_histogram' in df.columns:
                df['macd_momentum'] = df['macd_histogram'].diff().fillna(0)
            if 'ad' in df.columns:
                df['ad_momentum'] = df['ad'].diff().fillna(0)
                
            df['volume_momentum'] = df['volume'].pct_change().fillna(0)
            df['price_acceleration'] = df['price_change_1'].diff().fillna(0)

            # Keltner Channels (aproximación)
            if 'ema_20' in df.columns and 'atr_14' in df.columns:
                df['keltner_upper'] = df['ema_20'] + (2 * df['atr_14'])
                df['keltner_lower'] = df['ema_20'] - (2 * df['atr_14'])

            # Ease of Movement (aproximación)
            if len(df) > 1:
                distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
                box_height = df['volume'] / (df['high'] - df['low'])
                box_height = box_height.replace([np.inf, -np.inf], 0)
                df['ease_of_movement'] = distance_moved / box_height
                df['ease_of_movement'] = df['ease_of_movement'].replace([np.inf, -np.inf], 0)

            # Pattern recognition (simplificado)
            df['doji'] = ((abs(df['open'] - df['close']) / hl_range) < 0.1).astype(int)
            df['hammer'] = ((df['close'] > df['open']) &
                           ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
            df['shooting_star'] = ((df['open'] > df['close']) &
                                  ((df['high'] - df['open']) > 2 * (df['open'] - df['close']))).astype(int)
            df['engulfing'] = 0  # Placeholder
            df['harami'] = 0     # Placeholder
            df['spinning_top'] = ((abs(df['open'] - df['close']) / hl_range) < 0.3).astype(int)

        except Exception as e:
            print(f"⚠️ Error calculando features adicionales: {e}")

        return df

    def _clean_features_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y validar datos de features"""

        # Reemplazar infinitos
        df = df.replace([np.inf, -np.inf], np.nan)

        # Rellenar NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Verificar que no hay valores extremos
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                # Clip valores extremos
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                    df[col] = df[col].clip(lower=q01, upper=q99)

        return df

    def get_feature_info(self, feature_set: str = None) -> Dict:
        """Obtener información sobre los conjuntos de features"""

        if feature_set and feature_set in self.feature_sets:
            return {
                'feature_set': feature_set,
                'features': self.feature_sets[feature_set],
                'count': len(self.feature_sets[feature_set])
            }

        return {
            'available_sets': list(self.feature_sets.keys()),
            'sets_info': {
                name: {
                    'features': features,
                    'count': len(features)
                }
                for name, features in self.feature_sets.items()
            }
        }

    async def compute_features(self, symbol: str, klines_data: List, feature_set: str = 'tcn_definitivo') -> np.ndarray:
        """
        Computar features desde datos de klines de Binance
        
        Args:
            symbol: Símbolo del par (ej: BTCUSDT)
            klines_data: Lista de klines de Binance
            feature_set: Conjunto de features a calcular
            
        Returns:
            np.ndarray: Features calculadas o None si error
        """
        try:
            print(f"🔄 Calculando {len(self.feature_sets.get(feature_set, []))} features para {symbol}...")
            
            # Convertir klines a DataFrame
            df = self._klines_to_dataframe(klines_data)
            if df is None or df.empty:
                print(f"❌ Error: DataFrame vacío para {symbol}")
                return None
            
            # Calcular features
            df_features = self.calculate_features(df, feature_set)
            
            if df_features is None or df_features.empty:
                print(f"❌ Error: No se calcularon features para {symbol}")
                return None
            
            # Seleccionar solo las features del conjunto solicitado
            feature_columns = self.feature_sets.get(feature_set, [])
            available_columns = [col for col in feature_columns if col in df_features.columns]
            
            if not available_columns:
                print(f"❌ Error: No hay features disponibles para {symbol}")
                return None
            
            # Obtener datos como numpy array
            features_array = df_features[available_columns].values
            
            print(f"✅ Features calculadas: {len(available_columns)} de {len(feature_columns)} solicitadas")
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error calculando features para {symbol}: {e}")
            return None
    
    def _klines_to_dataframe(self, klines_data: List) -> pd.DataFrame:
        """Convertir datos de klines de Binance a DataFrame"""
        try:
            if not klines_data:
                return None
            
            # Formato esperado de klines de Binance:
            # [timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            
            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir a tipos correctos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ordenar por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Error convirtiendo klines a DataFrame: {e}")
            return None

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validar que el DataFrame tiene el formato correcto"""

        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Verificar columnas
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            print(f"❌ Columnas faltantes: {missing}")
            return False

        # Verificar tipos de datos
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"❌ Columna '{col}' debe ser numérica")
                return False

        # Verificar que no hay valores negativos en precios
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"❌ Columna '{col}' contiene valores no positivos")
                return False

        # Verificar lógica OHLC
        if not ((df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])).all():
            print("❌ Datos OHLC inconsistentes")
            return False

        print("✅ DataFrame validado correctamente")
        return True


# === FUNCIONES DE UTILIDAD ===

def create_features_engine() -> CentralizedFeaturesEngine:
    """Factory function para crear el motor de features"""
    return CentralizedFeaturesEngine()

def calculate_features_for_symbol(df: pd.DataFrame, feature_set: str = 'tcn_definitivo') -> pd.DataFrame:
    """Función de conveniencia para calcular features"""
    engine = create_features_engine()
    return engine.calculate_features(df, feature_set)

def get_available_feature_sets() -> List[str]:
    """Obtener lista de conjuntos de features disponibles"""
    engine = create_features_engine()
    return list(engine.feature_sets.keys())


# === TESTING ===
def test_centralized_features():
    """Test del motor centralizado de features"""
    print("🧪 TESTING CENTRALIZED FEATURES ENGINE")
    print("=" * 50)

    # Crear datos de prueba
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    np.random.seed(42)

    # Simular datos OHLCV realistas
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))

    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 100)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, 100)
    }, index=dates)

    # Crear motor
    engine = create_features_engine()

    # Validar datos
    is_valid = engine.validate_dataframe(test_data)
    print(f"✅ Datos válidos: {is_valid}")

    # Test cada conjunto de features
    for feature_set in engine.feature_sets.keys():
        print(f"\n🔧 Testing feature set: {feature_set}")

        try:
            features = engine.calculate_features(test_data, feature_set)
            print(f"   ✅ Features calculadas: {features.shape}")
            print(f"   📊 Rango de valores: ({features.min().min():.4f}, {features.max().max():.4f})")

            # Verificar que no hay NaN
            nan_count = features.isnull().sum().sum()
            print(f"   🔍 NaN encontrados: {nan_count}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Mostrar información
    info = engine.get_feature_info()
    print(f"\n📋 Información del motor:")
    print(f"   🎯 Conjuntos disponibles: {len(info['available_sets'])}")
    for name, details in info['sets_info'].items():
        print(f"   📊 {name}: {details['count']} features")


if __name__ == "__main__":
    test_centralized_features() 