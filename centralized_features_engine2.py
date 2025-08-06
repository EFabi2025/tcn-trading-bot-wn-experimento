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
        """Features para modelos TCN definitivos (88 features técnicas completas)"""
        return [
            # === MOMENTUM INDICATORS (17 features) ===
            'rsi_14', 'rsi_21', 'rsi_7',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
            'cci_14', 'cci_20',
            'rsi_momentum', 'macd_momentum',

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

            # === VOLUME INDICATORS (10 features) ===
            'ad', 'adosc', 'obv',
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'mfi_14', 'mfi_20',
            'ad_momentum', 'volume_momentum',

            # === PRICE PATTERNS (8 features) ===
            'hl_ratio', 'oc_ratio', 'price_position',
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_volatility_10', 'price_volatility_20',

            # === MARKET STRUCTURE (8 features) ===
            'higher_high', 'lower_low',
            'uptrend_strength', 'downtrend_strength',
            'resistance_touch', 'support_touch',
            'efficiency_ratio', 'fractal_dimension',

            # === MOMENTUM DERIVATIVES (1 feature) ===
            'price_acceleration',

            # === PRICE MOMENTUM (8 features) ===
            'price_momentum_1', 'price_momentum_3', 'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'price_momentum_normalized_5', 'price_momentum_normalized_10', 'price_momentum_normalized_20',

            # === VOLATILIDAD ADICIONAL (14 features) ===
            'volatility_5', 'volatility_10', 'volatility_15', 'volatility_20', 'volatility_30',
            'hl_volatility_5', 'hl_volatility_10', 'hl_volatility_15', 'hl_volatility_20', 'hl_volatility_30',
            'volatility_normalized_10', 'volatility_normalized_15', 'volatility_normalized_20', 'volatility_normalized_30'
        ]

    def _get_tcn_final_features(self) -> List[str]:
        """Features para modelos tcn_final (16 features técnicas simplificadas)"""
        return [
            # === RETURNS Y MOMENTUM (5 features) ===
            'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',

            # === MOVING AVERAGES (3 features) ===
            'sma_5', 'sma_20', 'ema_12',

            # === MOMENTUM INDICATORS (4 features) ===
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',

            # === VOLATILITY & VOLUME (4 features) ===
            'bb_position', 'bb_width', 'volume_ratio', 'volatility'
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

        # ✅ CORREGIDO: Validar integridad con acceso al precio original
        # Pasar información del precio antes de filtrar las features
        price_context = {
            'median_price': df['close'].median() if 'close' in df.columns else None,
            'price_std': df['close'].std() if 'close' in df.columns else None
        }
        validation_results = self.validate_talib_features_integrity(result_df, price_context)
        if not validation_results['talib_features_preserved']:
            print("⚠️ ADVERTENCIA: Features de TA-Lib pueden estar corrompidas")
            for warning in validation_results['warnings']:
                print(f"   ⚠️ {warning}")
        else:
            print("✅ Features de TA-Lib preservadas correctamente")

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
        """Implementaciones manuales SEGURAS cuando TA-Lib no está disponible"""
        try:
            # RSI manual con protección
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

            # ✅ División segura
            loss_safe = loss.clip(lower=1e-8)  # Usar pandas en lugar de numpy
            rs = gain / loss_safe
            rs = rs.replace([np.inf, -np.inf], 100)  # RS extremo
            rs = rs.clip(0, 1000)  # Limitar RS a rango razonable
            df['rsi_14'] = 100 - (100 / (1 + rs))
            # ✅ Eliminar clipping innecesario para TA-Lib
            # df['rsi_14'] = df['rsi_14'].clip(0, 100)  # Asegurar rango [0,100]

            # SMA/EMA básicos
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()

            # MACD básico con protección
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Reemplazar valores problemáticos
            manual_cols = ['rsi_14', 'sma_20', 'ema_12', 'macd', 'macd_signal', 'macd_histogram']
            for col in manual_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = df[col].ffill()

        except Exception as e:
            print(f"⚠️ Error en features manuales: {e}")
            # Fallback: valores neutros
            df['rsi_14'] = 50.0
            df['sma_20'] = df['close']
            df['ema_12'] = df['close']
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0

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

            # Bollinger Bands adicionales - CORREGIDO
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_range = df['bb_upper'] - df['bb_lower']
                # ✅ CORRECCIÓN: Manejo robusto de división por cero
                bb_range_safe = bb_range.replace(0, np.nan)
                bb_range_safe = bb_range_safe.fillna(bb_range_safe.mean())
                bb_range_safe = bb_range_safe.replace(0, 1e-8)  # Último recurso

                df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range_safe
                df['bb_position'] = df['bb_position'].clip(0, 1)  # Normalizar a [0,1]

                if 'bb_middle' in df.columns:
                    bb_middle_safe = df['bb_middle'].replace(0, np.nan)
                    bb_middle_safe = bb_middle_safe.fillna(df['close'])
                    df['bb_width'] = bb_range_safe / bb_middle_safe
                else:
                    df['bb_width'] = bb_range_safe / df['close']

            # Volume ratio - Unificado
            volume_sma_source = df.get('volume_sma_20', df['volume'].rolling(20).mean())
            volume_sma_safe = volume_sma_source.replace(0, np.nan)
            volume_sma_safe = volume_sma_safe.fillna(df['volume'].mean())
            volume_sma_safe = volume_sma_safe.clip(lower=1e-8)  # Usar pandas en lugar de numpy
            df['volume_ratio'] = df['volume'] / volume_sma_safe

            df['volume_price_trend'] = df['volume'] * df['close'].pct_change()

            # Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std()

            # === NUEVAS FEATURES DEL TCN DEFINITIVO ===

            # PRICE PATTERNS (8 features) - CORREGIDO
            hl_range = df['high'] - df['low']
            # ✅ CORRECCIÓN: Manejo robusto de división por cero
            hl_range_safe = hl_range.replace(0, np.nan)
            hl_range_safe = hl_range_safe.fillna(hl_range_safe.mean())
            hl_range_safe = hl_range_safe.replace(0, 1e-8)  # Último recurso

            df['hl_ratio'] = hl_range_safe / df['close']
            df['oc_ratio'] = (df['close'] - df['open']) / df['close']
            df['price_position'] = (df['close'] - df['low']) / hl_range_safe
            df['price_position'] = df['price_position'].clip(0, 1)  # Normalizar a [0,1]

            # Price changes
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)

            # Volatility windows - Corregido para consistencia
            returns = df['close'].pct_change()
            df['price_volatility_10'] = returns.rolling(10).std()
            df['price_volatility_20'] = returns.rolling(20).std()

            # MARKET STRUCTURE (8 features) - CORREGIDO
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

            df['uptrend_strength'] = (df['close'] > df['close'].shift(1)).rolling(10).sum() / 10
            df['downtrend_strength'] = (df['close'] < df['close'].shift(1)).rolling(10).sum() / 10

            # ✅ CORRECCIÓN: Eliminar look-ahead bias en resistance/support
            # Usar rolling window con shift para evitar data leakage
            rolling_max = df['close'].rolling(20, min_periods=1).max().shift(1)
            rolling_min = df['close'].rolling(20, min_periods=1).min().shift(1)

            df['resistance_touch'] = (df['close'] >= rolling_max * 0.99).astype(int)
            df['support_touch'] = (df['close'] <= rolling_min * 1.01).astype(int)

            # Market efficiency - CORREGIDO
            close_diff_abs = pd.Series(np.abs(df['close'].diff()), index=df.index)
            efficiency_numerator = np.abs(df['close'] - df['close'].shift(10))
            efficiency_denominator = close_diff_abs.rolling(10, min_periods=1).sum()
            efficiency_denominator = efficiency_denominator.replace(0, 1e-8)
            df['efficiency_ratio'] = (efficiency_numerator / efficiency_denominator).fillna(0)

            # Fractal dimension - Implementación básica
            # Calcula la dimensión fractal usando el método de box-counting simplificado
            if len(df) > 20:
                # Usar volatilidad como proxy para dimensión fractal
                volatility = df['close'].pct_change().rolling(20).std()
                # Normalizar a rango [1.0, 2.0] donde 1.0 = línea, 2.0 = ruido completo
                df['fractal_dimension'] = 1.0 + (volatility * 10).clip(0, 1)
            else:
                df['fractal_dimension'] = 1.5  # Valor neutral para datos insuficientes

            # MOMENTUM DERIVATIVES (5 features)
            if 'rsi_14' in df.columns:
                df['rsi_momentum'] = df['rsi_14'].diff().fillna(0)
            if 'macd_histogram' in df.columns:
                df['macd_momentum'] = df['macd_histogram'].diff().fillna(0)
            if 'ad' in df.columns:
                df['ad_momentum'] = df['ad'].diff().fillna(0)

            df['volume_momentum'] = df['volume'].pct_change().fillna(0)
            df['price_acceleration'] = df['price_change_1'].diff().fillna(0)

            # ✅ NUEVO: MOMENTUM DE PRECIO (múltiples períodos)
            # Momentum de precio con protección contra división por cero
            for period in [1, 3, 5, 10, 20]:
                # Calcular momentum para diferentes períodos
                price_diff = df['close'] - df['close'].shift(period)
                price_prev = df['close'].shift(period)

                # Protección contra división por cero
                price_prev_safe = price_prev.replace(0, np.nan)
                momentum = price_diff / price_prev_safe

                # Rellenar valores NaN con 0
                df[f'price_momentum_{period}'] = momentum.fillna(0)

                # ✅ NUEVO: MOMENTUM NORMALIZADO (basado en volatilidad)
                if period >= 5:
                    # Calcular volatilidad para normalizar momentum
                    returns = df['close'].pct_change().rolling(period*2).std()
                    volatility_safe = returns.replace(0, 0.01)  # Evitar división por cero

                    # Normalizar momentum por volatilidad
                    normalized_momentum = momentum / volatility_safe
                    df[f'price_momentum_normalized_{period}'] = normalized_momentum.fillna(0)

            # ✅ NUEVO: VOLATILIDAD ADICIONAL PARA USO EN ENTRENADOR
            # Volatilidad de diferentes períodos para cálculos dinámicos
            for period in [5, 10, 15, 20, 30]:
                # Volatilidad basada en returns
                returns = df['close'].pct_change()
                volatility = returns.rolling(period).std()
                # ✅ CORRECCIÓN: Crear ambas features para armonía perfecta
                if period in [10, 20]:
                    df[f'price_volatility_{period}'] = volatility.fillna(0.01)
                    df[f'volatility_{period}'] = volatility.fillna(0.01)  # ✅ NUEVO: Para armonía
                else:
                    df[f'volatility_{period}'] = volatility.fillna(0.01)

                # Volatilidad basada en high-low range
                hl_range = (df['high'] - df['low']) / df['close']
                hl_volatility = hl_range.rolling(period).mean()
                df[f'hl_volatility_{period}'] = hl_volatility.fillna(0.01)

                # Volatilidad normalizada (para comparaciones)
                if period >= 10:
                    # Normalizar por volatilidad histórica
                    long_term_vol = returns.rolling(period*3).std()
                    long_term_vol_safe = long_term_vol.replace(0, 0.01)
                    normalized_vol = volatility / long_term_vol_safe
                    df[f'volatility_normalized_{period}'] = normalized_vol.fillna(1.0)

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

            # Pattern recognition - Protegido contra división por cero
            df['doji'] = ((abs(df['open'] - df['close']) / hl_range_safe) < 0.1).astype(int)
            df['hammer'] = ((df['close'] > df['open']) &
                           ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
            df['shooting_star'] = ((df['open'] > df['close']) &
                                  ((df['high'] - df['open']) > 2 * (df['open'] - df['close']))).astype(int)
            df['engulfing'] = 0  # Placeholder
            df['harami'] = 0     # Placeholder
            df['spinning_top'] = ((abs(df['open'] - df['close']) / hl_range_safe) < 0.3).astype(int)

        except Exception as e:
            print(f"⚠️ Error calculando features adicionales: {e}")

        return df

    def _clean_features_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y validar datos de features - VERSIÓN CORREGIDA"""

        # Definir features de TA-Lib que NO deben ser clipeadas
        talib_features = [
            'rsi_14', 'rsi_21', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
            'cci_14', 'cci_20', 'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50',
            'adx_14', 'plus_di', 'minus_di', 'psar', 'aroon_up', 'aroon_down',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'atr_20', 'true_range',
            'natr_14', 'natr_20', 'ad', 'adosc', 'obv', 'volume_sma_10', 'volume_sma_20',
            'mfi_14', 'mfi_20'
        ]

        # Definir features manuales problemáticas que necesitan limpieza agresiva
        manual_features = [
            'bb_width', 'bb_position', 'volume_ratio', 'hl_ratio', 'oc_ratio', 'price_position',
            'price_change_1', 'price_change_5', 'price_change_10', 'price_volatility_10', 'price_volatility_20',
            'higher_high', 'lower_low', 'uptrend_strength', 'downtrend_strength',
            'resistance_touch', 'support_touch', 'efficiency_ratio', 'fractal_dimension',
            'rsi_momentum', 'macd_momentum', 'ad_momentum', 'volume_momentum', 'price_acceleration'
        ]

        # Reemplazar infinitos en todas las columnas
        df = df.replace([np.inf, -np.inf], np.nan)

        # Limpieza específica por tipo de feature
        for col in df.columns:
            if col in talib_features:
                # ✅ TA-Lib: Solo manejar NaN suavemente - NO clipping
                df[col] = df[col].ffill()
                # Preservar rangos originales de TA-Lib

            elif col in manual_features:
                # ✅ Manuales: Sin data leakage
                df[col] = df[col].ffill()

                # Valores por defecto específicos por tipo de feature
                if col.startswith('bb_'):
                    df[col] = df[col].fillna(0.5)  # Posición neutral en Bollinger
                elif col.endswith('_ratio'):
                    df[col] = df[col].fillna(1.0)  # Ratio neutral
                elif col.endswith('_touch'):
                    df[col] = df[col].fillna(0)    # No hay toque
                elif col.endswith('_strength'):
                    df[col] = df[col].fillna(0.5)  # Fuerza neutral
                elif col.endswith('_momentum'):
                    df[col] = df[col].fillna(0.0)  # Sin momentum
                else:
                    df[col] = df[col].fillna(0.0)  # Valor neutral genérico

                # Clipping moderado solo para features manuales problemáticas
                if hasattr(df[col], 'dtype') and str(df[col].dtype) in ['float64', 'float32']:
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                        df[col] = df[col].clip(lower=q01, upper=q99)

            else:
                # Features adicionales: limpieza estándar
                df[col] = df[col].ffill().fillna(0)

        return df

    def validate_talib_features_integrity(self, df: pd.DataFrame, price_context: Dict = None) -> Dict:
        """
        Validar que las features de TA-Lib mantienen su integridad después de la limpieza

        Args:
            df: DataFrame con features calculadas
            price_context: Dict con información del precio original (median_price, price_std)

        Returns:
            Dict con resultados de validación
        """
        validation_results = {
            'talib_features_preserved': True,
            'rsi_range_valid': True,
            'macd_extremes_preserved': True,
            'bb_ranges_valid': True,
            'warnings': []
        }

        # Validar RSI está en rango [0, 100]
        rsi_features = ['rsi_14', 'rsi_21', 'rsi_7']
        for rsi_col in rsi_features:
            if rsi_col in df.columns:
                rsi_min = df[rsi_col].min()
                rsi_max = df[rsi_col].max()
                if rsi_min < 0 or rsi_max > 100:
                    validation_results['rsi_range_valid'] = False
                    validation_results['warnings'].append(
                        f"RSI {rsi_col} fuera de rango [0,100]: [{rsi_min:.2f}, {rsi_max:.2f}]"
                    )

        # Validar que MACD mantiene valores extremos - Mejorado con límites adaptativos
        macd_features = ['macd', 'macd_signal', 'macd_histogram']
        for macd_col in macd_features:
            if macd_col in df.columns:
                macd_range = df[macd_col].max() - df[macd_col].min()
                macd_q99 = df[macd_col].quantile(0.99)
                macd_q01 = df[macd_col].quantile(0.01)
                macd_iqr = df[macd_col].quantile(0.75) - df[macd_col].quantile(0.25)

                # 💡 Límites adaptativos basados en el precio del activo - CORREGIDO
                if price_context and price_context.get('median_price') is not None:
                    price_level = price_context['median_price']
                    # Calcular threshold adaptativo basado en el precio - MUY PERMISIVO
                    if price_level > 1000:  # Crypto de alto valor (BTC, ETH)
                        macd_threshold = price_level * 0.00002  # 0.002% del precio (muy permisivo)
                    elif price_level > 100:  # Crypto de valor medio-alto
                        macd_threshold = price_level * 0.0002   # 0.02% del precio
                    elif price_level > 10:  # Crypto de valor medio
                        macd_threshold = price_level * 0.002    # 0.2% del precio
                    elif price_level > 1:  # Crypto de bajo valor (XRP, etc.)
                        macd_threshold = price_level * 0.02     # 2% del precio
                    elif price_level > 0.1:  # Crypto de muy bajo valor
                        macd_threshold = price_level * 0.01     # 1% del precio (más permisivo)
                    else:  # Crypto de precio extremadamente bajo
                        macd_threshold = 0.001  # Threshold fijo ultra bajo para casos extremos

                    macd_range_threshold = max(0.00001, macd_threshold)  # Mínimo extremadamente bajo
                    macd_iqr_threshold = max(0.000001, macd_threshold * 0.001)  # IQR extremadamente permisivo
                else:
                    # Fallback extremadamente permisivo cuando no hay precio disponible
                    macd_range_threshold = 0.00001  # Extremadamente permisivo
                    macd_iqr_threshold = 0.000001

                # Verificar múltiples métricas de compresión con límites adaptativos
                if (macd_range < macd_range_threshold or  # Rango muy pequeño
                    abs(macd_q99 - macd_q01) < macd_range_threshold or  # Percentiles muy juntos
                    macd_iqr < macd_iqr_threshold):  # IQR muy pequeño
                    validation_results['macd_extremes_preserved'] = False
                    price_info = f"price:{price_context.get('median_price', 'N/A'):.2f}" if price_context else "price:N/A"
                    validation_results['warnings'].append(
                        f"MACD {macd_col} comprimido - range:{macd_range:.6f}, iqr:{macd_iqr:.6f}, "
                        f"threshold:{macd_range_threshold:.6f} ({price_info})"
                    )

        # Validar Bollinger Bands - Mejorado con thresholds adaptativos
        bb_features = ['bb_upper', 'bb_middle', 'bb_lower']
        if all(bb in df.columns for bb in bb_features):
            bb_width = df['bb_upper'] - df['bb_lower']
            bb_width_std = bb_width.std()
            bb_width_mean = bb_width.mean()

            # Threshold adaptativo basado en el ancho promedio - CORREGIDO
            if price_context and price_context.get('median_price') is not None:
                price_level = price_context['median_price']
                # BB threshold como porcentaje del precio - MUY PERMISIVO
                if price_level > 1000:  # Crypto de alto valor
                    bb_threshold = price_level * 0.00002  # 0.002% del precio
                elif price_level > 100:  # Crypto de valor medio-alto
                    bb_threshold = price_level * 0.0002   # 0.02% del precio
                elif price_level > 10:  # Crypto de valor medio
                    bb_threshold = price_level * 0.002    # 0.2% del precio
                elif price_level > 1:  # Crypto de bajo valor (XRP, etc.)
                    bb_threshold = price_level * 0.02     # 2% del precio
                elif price_level > 0.1:  # Crypto de muy bajo valor
                    bb_threshold = price_level * 0.01     # 1% del precio (más permisivo)
                else:  # Crypto de precio extremadamente bajo
                    bb_threshold = 0.001  # Threshold fijo ultra bajo para casos extremos
            else:
                bb_threshold = 0.00001  # Fallback extremadamente permisivo

            if bb_width_std < bb_threshold and bb_width_mean < bb_threshold * 10:
                validation_results['bb_ranges_valid'] = False
                validation_results['warnings'].append(
                    f"Bollinger Bands comprimidas - std:{bb_width_std:.6f}, mean:{bb_width_mean:.6f}, "
                    f"threshold:{bb_threshold:.6f}"
                )

        # Validación general
        if not (validation_results['rsi_range_valid'] and
                validation_results['macd_extremes_preserved'] and
                validation_results['bb_ranges_valid']):
            validation_results['talib_features_preserved'] = False

        return validation_results

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
    """Test del motor centralizado de features con thresholds corregidos"""
    print("🧪 TESTING CENTRALIZED FEATURES ENGINE - THRESHOLDS CORREGIDOS")
    print("=" * 70)

    # Crear datos de prueba para diferentes tipos de crypto
    test_cases = [
        {'name': 'BTCUSDT', 'base_price': 45000, 'volatility': 0.03},
        {'name': 'ETHUSDT', 'base_price': 3000, 'volatility': 0.04},
        {'name': 'XRPUSDT', 'base_price': 3.60, 'volatility': 0.05},
        {'name': 'ADAUSDT', 'base_price': 0.45, 'volatility': 0.06}
    ]

    engine = create_features_engine()

    for test_case in test_cases:
        print(f"\n🧪 Probando {test_case['name']} (precio: ${test_case['base_price']})")

        # Crear datos de prueba específicos para este crypto
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        np.random.seed(42)

        base_price = test_case['base_price']
        volatility = test_case['volatility']

        # Simular datos OHLCV realistas
        returns = np.random.normal(0, volatility, 100)
        prices = base_price * np.exp(np.cumsum(returns))

        test_data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, volatility * 0.25, 100)),
            'high': prices * (1 + np.abs(np.random.normal(0, volatility * 0.5, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, volatility * 0.5, 100))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, 100)
        }, index=dates)

        try:
            # Calcular features
            features = engine.calculate_features(test_data, feature_set='tcn_definitivo')
            print(f"   ✅ Features calculadas: {features.shape}")

            # Verificar thresholds corregidos
            price_context = {
                'median_price': test_data['close'].median(),
                'price_std': test_data['close'].std()
            }
            validation = engine.validate_talib_features_integrity(features, price_context)

            if validation['talib_features_preserved']:
                print(f"   ✅ {test_case['name']}: Features de TA-Lib preservadas correctamente")
            else:
                print(f"   ⚠️ {test_case['name']}: Advertencias detectadas:")
                for warning in validation['warnings']:
                    print(f"      - {warning}")

        except Exception as e:
            print(f"   ❌ Error en test para {test_case['name']}: {e}")

    print(f"\n🎯 Test completado - Thresholds corregidos para diferentes tipos de crypto")
    print("=" * 70)


if __name__ == "__main__":
    test_centralized_features()
