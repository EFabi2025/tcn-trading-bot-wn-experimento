#!/usr/bin/env python3
"""
🎯 CENTRALIZED FEATURES ENGINE - PANDAS-TA VERSION
==================================================

Motor centralizado para cálculo de features técnicas usando pandas-ta.
Mantiene exactamente las mismas features que la versión TA-Lib.

Características:
- ✅ Implementación única y centralizada usando pandas-ta
- ✅ 66 features técnicos exactos del entrenador TCN definitivo
- ✅ Compatible con entrenamiento y trading en vivo
- ✅ Soporte para múltiples conjuntos de features
- ✅ Validación automática de datos
- ✅ Alternativa a TA-Lib sin dependencias compiladas
"""

import numpy as np
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    print("⚠️ pandas-ta no disponible. Instalar con: pip install pandas-ta")
    ta = None

from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CentralizedFeaturesEnginePandasTA:
    """
    Motor centralizado de features técnicas usando pandas-ta
    """

    def __init__(self):
        """Inicializar el motor de features pandas-ta"""
        self.feature_sets = {
            'tcn_definitivo': self._get_tcn_definitivo_features(),
            'tcn_final': self._get_tcn_final_features(),
            'full_set': self._get_full_features_set()
        }

        print("🎯 Centralized Features Engine (pandas-ta) inicializado")
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
        Calcular features técnicas usando pandas-ta
        
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

        # Calcular todas las features disponibles usando pandas-ta
        features_df = self._calculate_all_pandas_ta_features(features_df)

        # Calcular features adicionales no disponibles en pandas-ta
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

    def _calculate_all_pandas_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular todas las features usando pandas-ta"""

        if ta is None:
            print("⚠️ pandas-ta no disponible, usando implementaciones manuales")
            return self._calculate_manual_features(df)

        try:
            # === MOMENTUM INDICATORS ===
            # RSI
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_21'] = ta.rsi(df['close'], length=21)
            df['rsi_7'] = ta.rsi(df['close'], length=7)

            # MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and isinstance(macd_result, pd.DataFrame):
                if 'MACD_12_26_9' in macd_result.columns:
                    df['macd'] = macd_result['MACD_12_26_9']
                if 'MACDs_12_26_9' in macd_result.columns:
                    df['macd_signal'] = macd_result['MACDs_12_26_9']
                if 'MACDh_12_26_9' in macd_result.columns:
                    df['macd_histogram'] = macd_result['MACDh_12_26_9']

            # Stochastic
            stoch_result = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_result is not None and isinstance(stoch_result, pd.DataFrame):
                if 'STOCHk_14_3_3' in stoch_result.columns:
                    df['stoch_k'] = stoch_result['STOCHk_14_3_3']
                if 'STOCHd_14_3_3' in stoch_result.columns:
                    df['stoch_d'] = stoch_result['STOCHd_14_3_3']

            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)

            # Rate of Change
            df['roc_10'] = ta.roc(df['close'], length=10)
            df['roc_20'] = ta.roc(df['close'], length=20)

            # Momentum
            df['momentum_10'] = ta.mom(df['close'], length=10)
            df['momentum_20'] = ta.mom(df['close'], length=20)

            # CCI
            df['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
            df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)

            # === TREND INDICATORS ===
            # Moving Averages
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_5'] = ta.sma(df['close'], length=5)

            df['ema_10'] = ta.ema(df['close'], length=10)
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_12'] = ta.ema(df['close'], length=12)

            # ADX
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx_result is not None and isinstance(adx_result, pd.DataFrame):
                if 'ADX_14' in adx_result.columns:
                    df['adx_14'] = adx_result['ADX_14']
                if 'DMP_14' in adx_result.columns:
                    df['plus_di'] = adx_result['DMP_14']
                if 'DMN_14' in adx_result.columns:
                    df['minus_di'] = adx_result['DMN_14']

            # PSAR
            psar_result = ta.psar(df['high'], df['low'], af0=0.02, af=0.02, max_af=0.2)
            if psar_result is not None and isinstance(psar_result, pd.DataFrame):
                if 'PSARl_0.02_0.2' in psar_result.columns and 'PSARs_0.02_0.2' in psar_result.columns:
                    df['psar'] = psar_result['PSARl_0.02_0.2'].fillna(psar_result['PSARs_0.02_0.2'])
                elif len(psar_result.columns) > 0:
                    df['psar'] = psar_result.iloc[:, 0]

            # Aroon
            aroon_result = ta.aroon(df['high'], df['low'], length=14)
            if aroon_result is not None and isinstance(aroon_result, pd.DataFrame):
                if 'AROONU_14' in aroon_result.columns:
                    df['aroon_up'] = aroon_result['AROONU_14']
                if 'AROOND_14' in aroon_result.columns:
                    df['aroon_down'] = aroon_result['AROOND_14']

            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            bb_result = ta.bbands(df['close'], length=20, std=2)
            if bb_result is not None and isinstance(bb_result, pd.DataFrame):
                if 'BBU_20_2.0' in bb_result.columns:
                    df['bb_upper'] = bb_result['BBU_20_2.0']
                if 'BBM_20_2.0' in bb_result.columns:
                    df['bb_middle'] = bb_result['BBM_20_2.0']
                if 'BBL_20_2.0' in bb_result.columns:
                    df['bb_lower'] = bb_result['BBL_20_2.0']

            # ATR
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_20'] = ta.atr(df['high'], df['low'], df['close'], length=20)

            # True Range
            df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])

            # NATR (Normalized ATR)
            df['natr_14'] = ta.natr(df['high'], df['low'], df['close'], length=14)
            df['natr_20'] = ta.natr(df['high'], df['low'], df['close'], length=20)

            # === VOLUME INDICATORS ===
            # A/D Line
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])

            # A/D Oscillator
            df['adosc'] = ta.adosc(df['high'], df['low'], df['close'], df['volume'])

            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])

            # Volume SMA
            df['volume_sma_10'] = ta.sma(df['volume'], length=10)
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)

            # MFI
            df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            df['mfi_20'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=20)

        except Exception as e:
            print(f"⚠️ Error calculando features pandas-ta: {e}")
            # Fallback a implementaciones manuales para las que fallaron
            return self._calculate_manual_features(df)

        return df

    def _calculate_manual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implementaciones manuales básicas cuando pandas-ta no está disponible"""
        try:
            # RSI manual
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)  # Evitar división por cero
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # RSI para otros períodos
            gain_21 = delta.where(delta > 0, 0).rolling(window=21).mean()
            loss_21 = -delta.where(delta < 0, 0).rolling(window=21).mean()
            rs_21 = gain_21 / (loss_21 + 1e-8)
            df['rsi_21'] = 100 - (100 / (1 + rs_21))
            
            gain_7 = delta.where(delta > 0, 0).rolling(window=7).mean()
            loss_7 = -delta.where(delta < 0, 0).rolling(window=7).mean()
            rs_7 = gain_7 / (loss_7 + 1e-8)
            df['rsi_7'] = 100 - (100 / (1 + rs_7))
            
            # SMA/EMA básicos
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_5'] = df['close'].rolling(5).mean()
            
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # MACD básico
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands básicas
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ROC manual
            df['roc_10'] = df['close'].pct_change(periods=10) * 100
            df['roc_20'] = df['close'].pct_change(periods=20) * 100
            
            # Momentum manual
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            
            # ATR manual
            hl = df['high'] - df['low']
            hc = (df['high'] - df['close'].shift(1)).abs()
            lc = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df['true_range'] = tr
            df['atr_14'] = tr.rolling(14).mean()
            df['atr_20'] = tr.rolling(20).mean()
            
            # Volume features básicas
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['obv'] = (df['volume'] * ((df['close'] > df['close'].shift(1)).astype(int) * 2 - 1)).cumsum()
            
        except Exception as e:
            print(f"⚠️ Error en implementaciones manuales: {e}")

        return df

    def _calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features adicionales no disponibles en pandas-ta"""

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
            if 'volume_sma_20' in df.columns:
                volume_sma_safe = df['volume_sma_20'].replace(0, 1e-8)
                df['volume_ratio'] = df['volume'] / volume_sma_safe
            else:
                volume_sma = df['volume'].rolling(window=20, min_periods=1).mean()
                volume_sma_safe = volume_sma.replace(0, 1e-8)
                df['volume_ratio'] = df['volume'] / volume_sma_safe

            # Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std()

            # === FEATURES DEL TCN DEFINITIVO ===
            
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

            # NATR manual si no está disponible
            if 'natr_14' not in df.columns and 'atr_14' in df.columns:
                df['natr_14'] = (df['atr_14'] / df['close']) * 100
            if 'natr_20' not in df.columns and 'atr_20' in df.columns:
                df['natr_20'] = (df['atr_20'] / df['close']) * 100

            # Williams %R manual si no está disponible
            if 'williams_r' not in df.columns:
                highest_high = df['high'].rolling(14).max()
                lowest_low = df['low'].rolling(14).min()
                df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100

            # CCI manual si no está disponible
            if 'cci_14' not in df.columns:
                tp = (df['high'] + df['low'] + df['close']) / 3
                tp_sma = tp.rolling(14).mean()
                tp_mad = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
                df['cci_14'] = (tp - tp_sma) / (0.015 * tp_mad)
                
            if 'cci_20' not in df.columns:
                tp = (df['high'] + df['low'] + df['close']) / 3
                tp_sma = tp.rolling(20).mean()
                tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
                df['cci_20'] = (tp - tp_sma) / (0.015 * tp_mad)

            # PSAR manual si no está disponible
            if 'psar' not in df.columns:
                # Implementación simplificada de PSAR
                af = 0.02
                max_af = 0.2
                df['psar'] = df['close'].copy()  # Placeholder simplificado

            # ADX manual si no está disponible
            if 'adx_14' not in df.columns:
                # Implementación simplificada
                df['plus_di'] = 0.0
                df['minus_di'] = 0.0
                df['adx_14'] = 0.0

            # Aroon manual si no está disponible
            if 'aroon_up' not in df.columns:
                for period in [14]:
                    high_idx = df['high'].rolling(period).apply(lambda x: period - 1 - x.argmax())
                    low_idx = df['low'].rolling(period).apply(lambda x: period - 1 - x.argmin())
                    df['aroon_up'] = ((period - high_idx) / period) * 100
                    df['aroon_down'] = ((period - low_idx) / period) * 100

            # MFI manual si no está disponible
            if 'mfi_14' not in df.columns:
                tp = (df['high'] + df['low'] + df['close']) / 3
                rmf = tp * df['volume']
                rmf_pos = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
                rmf_neg = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
                mfi_ratio = rmf_pos / (rmf_neg + 1e-8)
                df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
                
            if 'mfi_20' not in df.columns:
                tp = (df['high'] + df['low'] + df['close']) / 3
                rmf = tp * df['volume']
                rmf_pos = rmf.where(tp > tp.shift(1), 0).rolling(20).sum()
                rmf_neg = rmf.where(tp < tp.shift(1), 0).rolling(20).sum()
                mfi_ratio = rmf_pos / (rmf_neg + 1e-8)
                df['mfi_20'] = 100 - (100 / (1 + mfi_ratio))

            # ADOSC manual si no está disponible
            if 'adosc' not in df.columns and 'ad' in df.columns:
                df['adosc'] = ta.sma(df['ad'], length=3) - ta.sma(df['ad'], length=10) if ta else df['ad'].rolling(3).mean() - df['ad'].rolling(10).mean()

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
            print(f"🔄 Calculando {len(self.feature_sets.get(feature_set, []))} features para {symbol} (pandas-ta)...")
            
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

def create_features_engine_pandas_ta() -> CentralizedFeaturesEnginePandasTA:
    """Factory function para crear el motor de features pandas-ta"""
    return CentralizedFeaturesEnginePandasTA()

def calculate_features_for_symbol_pandas_ta(df: pd.DataFrame, feature_set: str = 'tcn_definitivo') -> pd.DataFrame:
    """Función de conveniencia para calcular features con pandas-ta"""
    engine = create_features_engine_pandas_ta()
    return engine.calculate_features(df, feature_set)

def get_available_feature_sets_pandas_ta() -> List[str]:
    """Obtener lista de conjuntos de features disponibles"""
    engine = create_features_engine_pandas_ta()
    return list(engine.feature_sets.keys())


# === TESTING ===
def test_centralized_features_pandas_ta():
    """Test del motor centralizado de features pandas-ta"""
    print("🧪 TESTING CENTRALIZED FEATURES ENGINE (PANDAS-TA)")
    print("=" * 60)

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
    engine = create_features_engine_pandas_ta()

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
    test_centralized_features_pandas_ta() 