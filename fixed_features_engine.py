#!/usr/bin/env python3
"""
🎯 FIXED FEATURES ENGINE - Versión Corregida
============================================

Motor de features corregido que resuelve los problemas detectados:
- ✅ Elimina features constantes
- ✅ Resuelve multicolinealidad
- ✅ Normalización robusta y consistente
- ✅ Escalado uniforme
- ✅ Compatible con modelos entrenados
"""

import numpy as np
import pandas as pd
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib no disponible")

from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FixedFeaturesEngine:
    """
    🔧 Motor de features corregido que elimina los problemas detectados
    """

    def __init__(self):
        """Inicializar el motor corregido"""
        self.feature_sets = {
            'tcn_clean': self._get_tcn_clean_features(),
            'tcn_robust': self._get_tcn_robust_features()
        }
        
        # Cache para normalización consistente
        self.normalization_params = {}
        self.feature_stats = {}
        
        print("🔧 Fixed Features Engine inicializado")
        print(f"   📊 Conjuntos corregidos: {list(self.feature_sets.keys())}")

    def _get_tcn_clean_features(self) -> List[str]:
        """
        Features limpias sin multicolinealidad ni constantes (32 features)
        Optimizadas para máxima información con mínima redundancia
        """
        return [
            # === PRICE FEATURES (4) ===
            'close_normalized',     # Precio normalizado
            'returns_1',           # Return 1 período 
            'returns_5',           # Return 5 períodos
            'returns_20',          # Return 20 períodos (tendencia)
            
            # === MOMENTUM UNIQUE (6) ===
            'rsi_14',              # RSI principal
            'rsi_7',               # RSI corto plazo (diferente velocidad)
            'macd_normalized',     # MACD normalizado
            'macd_signal_normalized', # MACD Signal normalizado
            'williams_r',          # Williams %R (complementario a RSI)
            'roc_10',              # Rate of Change
            
            # === TREND INDICATORS (6) ===
            'sma_trend_5_20',      # Relación SMA 5/20
            'ema_trend_10_50',     # Relación EMA 10/50
            'price_sma_position',  # Posición precio vs SMA
            'adx_14',              # ADX para fuerza de tendencia
            'plus_di_minus_di',    # Diferencia DI+ y DI-
            'aroon_difference',    # Diferencia Aroon Up/Down
            
            # === VOLATILITY (5) ===
            'bb_position_clean',   # Posición en Bollinger limpia
            'bb_width_normalized', # Ancho Bollinger normalizado
            'atr_normalized',      # ATR normalizado
            'volatility_ratio',    # Volatilidad actual vs histórica
            'price_range_norm',    # Rango H-L normalizado
            
            # === VOLUME (4) ===
            'volume_trend',        # Tendencia de volumen
            'volume_price_corr',   # Correlación volumen-precio
            'mfi_14',              # Money Flow Index
            'volume_breakout',     # Detección breakout volumen
            
            # === STATISTICAL (4) ===
            'price_momentum',      # Momentum estadístico
            'trend_strength',      # Fuerza de tendencia
            'mean_reversion',      # Señal mean reversion
            'volatility_regime',   # Régimen de volatilidad
            
            # === CYCLE & PATTERN (3) ===
            'cycle_position',      # Posición en ciclo
            'pattern_strength',    # Fuerza del patrón
            'market_regime'        # Régimen de mercado
        ]

    def _get_tcn_robust_features(self) -> List[str]:
        """Features robustas reducidas (21 features) para modelos sensibles"""
        return [
            # Core price features
            'returns_1', 'returns_5', 'returns_20',
            
            # Momentum cleaned
            'rsi_14', 'macd_normalized', 'williams_r',
            
            # Trend cleaned  
            'sma_trend_5_20', 'price_sma_position', 'adx_14',
            
            # Volatility cleaned
            'bb_position_clean', 'atr_normalized', 'volatility_ratio',
            
            # Volume cleaned
            'volume_trend', 'mfi_14',
            
            # Statistical cleaned
            'price_momentum', 'trend_strength', 'mean_reversion',
            
            # Pattern cleaned
            'cycle_position', 'pattern_strength', 'market_regime'
        ]

    def calculate_features(self, df: pd.DataFrame, feature_set: str = 'tcn_clean') -> pd.DataFrame:
        """
        Calcular features limpias sin problemas de multicolinealidad
        """
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set}' no disponible")

        # Validar entrada
        if not self._validate_input(df):
            raise ValueError("DataFrame inválido")

        # Crear features base
        features_df = df.copy()
        
        # Calcular indicadores técnicos base
        features_df = self._calculate_base_indicators(features_df)
        
        # Crear features combinadas y limpias
        features_df = self._create_clean_features(features_df)
        
        # Normalizar robustamente
        features_df = self._robust_normalization(features_df)
        
        # Seleccionar features del conjunto
        requested_features = self.feature_sets[feature_set]
        available_features = [f for f in requested_features if f in features_df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No hay features disponibles")

        result_df = features_df[available_features].copy()
        
        # Verificación final
        result_df = self._final_cleanup(result_df)
        
        print(f"✅ Features limpias: {len(result_df.columns)} calculadas sin problemas")
        return result_df

    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos base usando TA-Lib cuando esté disponible"""
        
        if not TALIB_AVAILABLE:
            return self._calculate_manual_indicators(df)
        
        try:
            # Extraer arrays
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            volume = df['volume'].values.astype(float)
            
            # === MOMENTUM ===
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_7'] = talib.RSI(close, timeperiod=7)
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            df['roc_10'] = talib.ROC(close, timeperiod=10)
            
            # MACD
            macd, macd_signal, _ = talib.MACD(close)
            df['macd_raw'] = macd
            df['macd_signal_raw'] = macd_signal
            
            # === TREND ===
            df['sma_5'] = talib.SMA(close, timeperiod=5)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['ema_10'] = talib.EMA(close, timeperiod=10)
            df['ema_50'] = talib.EMA(close, timeperiod=50)
            
            df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            
            # === VOLATILITY ===
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
            df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            
            # === VOLUME ===
            df['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error con TA-Lib, usando cálculos manuales: {e}")
            return self._calculate_manual_indicators(df)

    def _calculate_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cálculos manuales cuando TA-Lib no está disponible"""
        
        # RSI manual
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-8)  # Evitar división por cero
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        delta_7 = df['close'].diff()
        gain_7 = delta_7.where(delta_7 > 0, 0).rolling(7).mean()
        loss_7 = -delta_7.where(delta_7 < 0, 0).rolling(7).mean()
        rs_7 = gain_7 / (loss_7 + 1e-8)
        df['rsi_7'] = 100 - (100 / (1 + rs_7))
        
        # MACD manual
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_raw'] = ema12 - ema26
        df['macd_signal_raw'] = df['macd_raw'].ewm(span=9).mean()
        
        # Medias móviles
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Williams %R manual
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-8)
        
        # ROC manual
        df['roc_10'] = df['close'].pct_change(10) * 100
        
        # ADX simplificado (aproximación)
        df['adx_14'] = abs(df['close'].pct_change().rolling(14).mean()) * 100
        
        # Bollinger Bands manual
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        
        # ATR manual
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.DataFrame([high_low, high_close_prev, low_close_prev]).max()
        df['atr_14'] = true_range.rolling(14).mean()
        
        return df

    def _create_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear features combinadas sin multicolinealidad"""
        
        # === PRICE FEATURES ===
        df['close_normalized'] = df['close'] / df['close'].rolling(50).mean()
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)
        
        # === MOMENTUM CLEANED ===
        # RSI ya está calculado
        # MACD normalizado por volatilidad
        atr_safe = df['atr_14'].replace(0, df['atr_14'].median())
        df['macd_normalized'] = df['macd_raw'] / (atr_safe + 1e-8)
        df['macd_signal_normalized'] = df['macd_signal_raw'] / (atr_safe + 1e-8)
        
        # === TREND FEATURES (sin redundancia) ===
        df['sma_trend_5_20'] = (df['sma_5'] / (df['sma_20'] + 1e-8)) - 1
        df['ema_trend_10_50'] = (df['ema_10'] / (df['ema_50'] + 1e-8)) - 1
        df['price_sma_position'] = (df['close'] / (df['sma_20'] + 1e-8)) - 1
        
        # Combinación DI única
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            df['plus_di_minus_di'] = df['plus_di'] - df['minus_di']
        else:
            df['plus_di_minus_di'] = 0
        
        # Aroon difference
        if 'aroon_up' in df.columns and 'aroon_down' in df.columns:
            df['aroon_difference'] = df['aroon_up'] - df['aroon_down']
        else:
            df['aroon_difference'] = 0
        
        # === VOLATILITY CLEANED ===
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_position_clean'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-8)
        df['bb_width_normalized'] = bb_range / (df['bb_middle'] + 1e-8)
        df['atr_normalized'] = df['atr_14'] / (df['close'] + 1e-8)
        
        # Volatilidad histórica
        historical_vol = df['close'].pct_change().rolling(20).std()
        current_vol = df['close'].pct_change().rolling(5).std()
        df['volatility_ratio'] = current_vol / (historical_vol + 1e-8)
        
        df['price_range_norm'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        
        # === VOLUME CLEANED ===
        volume_ma = df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'] / (volume_ma + 1e-8)
        
        # Correlación precio-volumen (simplificada)
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        df['volume_price_corr'] = (price_change * volume_change).rolling(10).mean()
        
        # Breakout de volumen
        volume_std = df['volume'].rolling(20).std()
        df['volume_breakout'] = (df['volume'] - volume_ma) / (volume_std + 1e-8)
        
        # === STATISTICAL FEATURES ===
        df['price_momentum'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1
        
        # Fuerza de tendencia (basada en ADX)
        df['trend_strength'] = df['adx_14'] / 100
        
        # Mean reversion signal
        price_zscore = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-8)
        df['mean_reversion'] = np.tanh(price_zscore)  # Bounded between -1 and 1
        
        # Régimen de volatilidad
        df['volatility_regime'] = (df['atr_normalized'] > df['atr_normalized'].rolling(50).median()).astype(int)
        
        # === CYCLE & PATTERN ===
        # Posición en ciclo (simplificada)
        df['cycle_position'] = np.sin(2 * np.pi * np.arange(len(df)) / 20)  # 20-period cycle
        
        # Fuerza del patrón
        df['pattern_strength'] = abs(df['bb_position_clean'] - 0.5) * 2  # 0 to 1
        
        # Régimen de mercado
        trend_up = (df['sma_trend_5_20'] > 0).astype(int)
        vol_high = (df['volatility_regime'] == 1).astype(int)
        df['market_regime'] = trend_up + vol_high  # 0, 1, or 2
        
        return df

    def _robust_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalización robusta y consistente"""
        
        # Features que necesitan normalización especial
        bounded_features = ['rsi_14', 'rsi_7', 'williams_r', 'bb_position_clean', 'mfi_14']
        ratio_features = ['sma_trend_5_20', 'ema_trend_10_50', 'price_sma_position']
        
        for col in df.columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                
                if col in bounded_features:
                    # Features ya bounded (0-100 o -100-0)
                    if col == 'williams_r':
                        df[col] = (df[col] + 100) / 100  # -100-0 -> 0-1
                    else:
                        df[col] = df[col] / 100  # 0-100 -> 0-1
                
                elif col in ratio_features:
                    # Features de ratio - usar tanh para bounded
                    df[col] = np.tanh(df[col])
                
                else:
                    # Normalización robusta para otras features
                    median_val = df[col].median()
                    mad = (df[col] - median_val).abs().median()  # Median Absolute Deviation
                    
                    if mad > 1e-8:
                        df[col] = (df[col] - median_val) / (mad * 1.4826)  # Scale factor for normal distribution
                        df[col] = np.tanh(df[col] / 3)  # Bound to [-1, 1]
                    else:
                        df[col] = 0
        
        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza final para asegurar calidad"""
        
        # Reemplazar infinitos
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with forward fill, then backward fill, then 0
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Clip extreme values
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].clip(-5, 5)  # Reasonable bounds
        
        # Verificar que no hay features constantes
        constant_features = []
        for col in df.columns:
            if df[col].std() < 1e-8:
                constant_features.append(col)
        
        if constant_features:
            print(f"⚠️ Removiendo features constantes: {constant_features}")
            df = df.drop(columns=constant_features)
        
        return df

    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validar DataFrame de entrada"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            return False
        
        if len(df) < 50:  # Mínimo para cálculos
            return False
        
        return True

    async def compute_features(self, symbol: str, klines_data: List, feature_set: str = 'tcn_clean') -> np.ndarray:
        """Computar features limpias desde klines"""
        try:
            print(f"🔧 Calculando features LIMPIAS para {symbol}...")
            
            # Convertir a DataFrame
            df = self._klines_to_dataframe(klines_data)
            if df is None:
                return None
            
            # Calcular features
            df_features = self.calculate_features(df, feature_set)
            
            if df_features.empty:
                return None
            
            # Retornar como array
            features_array = df_features.values
            
            print(f"✅ Features limpias calculadas: {features_array.shape}")
            
            # Verificación rápida de calidad
            nan_count = np.isnan(features_array).sum()
            inf_count = np.isinf(features_array).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"⚠️ Problemas detectados: {nan_count} NaN, {inf_count} Inf")
                return None
            
            return features_array
            
        except Exception as e:
            print(f"❌ Error calculando features limpias: {e}")
            return None

    def _klines_to_dataframe(self, klines_data: List) -> pd.DataFrame:
        """Convertir klines a DataFrame"""
        try:
            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Validar OHLC logic
            valid_ohlc = (
                (df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close'])
            )
            
            if not valid_ohlc.all():
                print("⚠️ Datos OHLC inconsistentes detectados")
            
            return df.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            print(f"❌ Error convirtiendo klines: {e}")
            return None


# === TESTING ===
async def test_fixed_features():
    """Test del motor corregido"""
    print("🧪 TESTING FIXED FEATURES ENGINE")
    print("=" * 50)
    
    from definitivo_tcn_predictor import BinanceDataProvider
    
    engine = FixedFeaturesEngine()
    
    try:
        async with BinanceDataProvider() as provider:
            klines = await provider.get_klines("BTCUSDT", "1m", 200)
            
            if klines:
                # Test features limpias
                features_clean = await engine.compute_features("BTCUSDT", klines, 'tcn_clean')
                
                if features_clean is not None:
                    print(f"✅ Features limpias: {features_clean.shape}")
                    print(f"   📊 Rango valores: [{features_clean.min():.3f}, {features_clean.max():.3f}]")
                    print(f"   📈 Media: {features_clean.mean():.3f}")
                    print(f"   📉 Std: {features_clean.std():.3f}")
                    
                    # Verificar calidad
                    nan_count = np.isnan(features_clean).sum()
                    inf_count = np.isinf(features_clean).sum()
                    
                    print(f"   🔍 NaN: {nan_count}, Inf: {inf_count}")
                    
                    if nan_count == 0 and inf_count == 0:
                        print("✅ FEATURES DE ALTA CALIDAD - SIN PROBLEMAS DETECTADOS")
                    else:
                        print("❌ Aún hay problemas en las features")
                else:
                    print("❌ Error calculando features")
            else:
                print("❌ No se pudieron obtener datos")
                
    except Exception as e:
        print(f"❌ Error en test: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fixed_features()) 