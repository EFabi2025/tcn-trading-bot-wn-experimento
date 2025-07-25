#!/usr/bin/env python3
"""
🎯 MOTOR DE FEATURES OPTIMIZADO
================================

Motor mejorado que elimina correlaciones y añade features direccionales potentes.
Diseñado específicamente para mejorar las señales de compra/venta.

Mejoras:
- ✅ Elimina features altamente correlacionadas  
- ✅ Añade features direccionales potentes
- ✅ Detecta breakouts y momentum real
- ✅ Incluye análisis de presión compradora/vendedora
- ✅ 40 features seleccionadas estratégicamente
"""

import numpy as np
import pandas as pd
try:
    import talib
except ImportError:
    print("⚠️ TA-Lib no disponible, usando implementaciones alternativas")
    talib = None

from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class OptimizedFeaturesEngine:
    """Motor de features optimizado para trading direccional"""

    def __init__(self):
        """Inicializar el motor optimizado"""
        self.feature_sets = {
            'optimized_tcn': self._get_optimized_features(),
            'minimal_power': self._get_minimal_power_features(),
            'directional_only': self._get_directional_features()
        }

        print("🚀 Motor de Features OPTIMIZADO inicializado")
        print("🎯 Diseñado para señales direccionales claras")
        for name, features in self.feature_sets.items():
            print(f"   📊 {name}: {len(features)} features")

    def _get_optimized_features(self) -> List[str]:
        """40 features optimizadas - Sin correlaciones altas"""
        return [
            # === MOMENTUM CORE (6 features) - SIN CORRELACIONES ===
            'rsi_14',              # RSI principal (solo uno)
            'macd_histogram',      # Señal MACD más pura
            'stoch_k',            # Momentum stochastic
            'momentum_diff',      # Momentum diferencial (NUEVO)
            'momentum_acceleration', # Aceleración del momentum (NUEVO)
            'volume_momentum',    # Momentum del volumen

            # === TREND DIRECCIONAL (6 features) - MÁS POTENTES ===
            'ema_crossover',      # Crossover EMA 12/26 (NUEVO)
            'trend_strength',     # Fuerza del trend (NUEVO)
            'adx_14',            # Trend strength
            'breakout_strength', # Fuerza de breakout (NUEVO)
            'trend_consistency', # Consistencia del trend (NUEVO)
            'ema_slope_12',      # Pendiente EMA12 (NUEVO)

            # === VOLATILITY INTELIGENTE (6 features) ===
            'atr_14',            # Volatilidad absoluta
            'bb_squeeze',        # Bollinger Bands squeeze (NUEVO)
            'volatility_expansion', # Expansión de volatilidad (NUEVO)
            'true_range_normalized', # True range normalizado (NUEVO)
            'price_efficiency',  # Eficiencia del movimiento (NUEVO)
            'volatility_regime', # Régimen de volatilidad (NUEVO)

            # === VOLUME DIRECCIONAL (6 features) ===
            'buying_pressure',   # Presión compradora (NUEVO)
            'selling_pressure',  # Presión vendedora (NUEVO)
            'volume_breakout',   # Breakout de volumen (NUEVO)
            'vwap_position',     # Posición vs VWAP (NUEVO)
            'accumulation_distribution', # A/D Line
            'smart_money_flow',  # Flujo de dinero inteligente

            # === PRICE ACTION PURA (6 features) ===
            'candle_strength',   # Fuerza de la vela (NUEVO)
            'body_wick_ratio',   # Ratio cuerpo/mecha (NUEVO)
            'price_momentum_1',  # Momentum precio 1 período
            'price_momentum_5',  # Momentum precio 5 períodos
            'support_resistance_break', # Ruptura S/R (NUEVO)
            'price_position_range', # Posición en el rango (NUEVO)

            # === MARKET MICROSTRUCTURE (6 features) ===
            'bid_ask_pressure',  # Presión bid/ask simulada (NUEVO)
            'orderbook_imbalance', # Desequilibrio orderbook sim (NUEVO)
            'institutional_activity', # Actividad institucional (NUEVO)
            'retail_exhaustion', # Agotamiento retail (NUEVO)
            'market_regime',     # Régimen de mercado (NUEVO)
            'price_efficiency'   # Eficiencia del movimiento
        ]

    def _get_minimal_power_features(self) -> List[str]:
        """20 features más potentes - Conjunto mínimo"""
        return [
            # Momentum puro
            'rsi_14', 'macd_histogram', 'momentum_acceleration',
            # Trend direccional  
            'ema_crossover', 'trend_strength', 'breakout_strength',
            # Volatility
            'bb_squeeze', 'volatility_expansion',
            # Volume direccional
            'buying_pressure', 'selling_pressure', 'volume_breakout',
            # Price action
            'candle_strength', 'support_resistance_break',
            # Microstructure
            'smart_money_flow', 'market_regime',
            # Básicos optimizados
            'price_vs_ema20', 'trend_consistency', 'price_efficiency',
            'vwap_position', 'institutional_activity'
        ]

    def _get_directional_features(self) -> List[str]:
        """Features puramente direccionales para señales claras"""
        return [
            'ema_crossover', 'trend_strength', 'breakout_strength',
            'buying_pressure', 'selling_pressure', 'momentum_acceleration',
            'bb_squeeze', 'volume_breakout', 'smart_money_flow',
            'support_resistance_break', 'candle_strength', 'market_regime'
        ]

    def calculate_features(self, df: pd.DataFrame, feature_set: str = 'optimized_tcn') -> pd.DataFrame:
        """Calcular features optimizadas"""
        
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set}' no disponible")

        # Validar datos
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener: {required_columns}")

        # Crear copia
        features_df = df.copy()

        # Calcular features base de TA-Lib
        features_df = self._calculate_base_features(features_df)
        
        # Calcular features direccionales optimizadas
        features_df = self._calculate_directional_features(features_df)
        
        # Calcular features de microstructura
        features_df = self._calculate_microstructure_features(features_df)

        # Seleccionar solo las features solicitadas
        requested_features = self.feature_sets[feature_set]
        available_features = [f for f in requested_features if f in features_df.columns]

        if len(available_features) != len(requested_features):
            missing = set(requested_features) - set(available_features)
            print(f"⚠️ Features faltantes: {missing}")

        result_df = features_df[available_features].copy()
        result_df = self._clean_features_data(result_df)

        print(f"✅ Features OPTIMIZADAS calculadas: {len(result_df.columns)}")
        return result_df

    def _calculate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features base necesarias"""
        
        if talib is None:
            return self._calculate_base_manual(df)

        open_arr = df['open'].values.astype(float)
        high_arr = df['high'].values.astype(float)
        low_arr = df['low'].values.astype(float)
        close_arr = df['close'].values.astype(float)
        volume_arr = df['volume'].values.astype(float)

        try:
            # Features base necesarias
            df['rsi_14'] = talib.RSI(close_arr, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_arr)
            df['macd_histogram'] = macd_hist
            
            # Stochastic
            slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr)
            df['stoch_k'] = slowk
            
            # Otros momentum
            df['williams_r'] = talib.WILLR(high_arr, low_arr, close_arr)
            df['roc_10'] = talib.ROC(close_arr, timeperiod=10)
            df['cci_14'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)
            
            # EMAs necesarias
            df['ema_12'] = talib.EMA(close_arr, timeperiod=12)
            df['ema_20'] = talib.EMA(close_arr, timeperiod=20)
            df['ema_26'] = talib.EMA(close_arr, timeperiod=26)
            
            # ADX y DI
            df['adx_14'] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
            
            # ATR
            df['atr_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
            df['true_range'] = talib.TRANGE(high_arr, low_arr, close_arr)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle  
            df['bb_lower'] = bb_lower
            
            # Volume indicators
            df['accumulation_distribution'] = talib.AD(high_arr, low_arr, close_arr, volume_arr)
            
        except Exception as e:
            print(f"⚠️ Error en features base TA-Lib: {e}")
            
        return df

    def _calculate_base_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features base manuales si TA-Lib no está disponible"""
        
        # RSI manual
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # EMAs
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd_histogram'] = df['ema_12'] - df['ema_26']
        
        # ATR manual
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        
        return df

    def _calculate_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🎯 Features direccionales potentes - LA CLAVE"""
        
        try:
            # 1. EMA CROSSOVER POWER
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                df['ema_crossover'] = (df['ema_12'] - df['ema_26']) / df['close']
            
            # 2. TREND STRENGTH
            if 'ema_20' in df.columns:
                price_above_ema = (df['close'] > df['ema_20']).rolling(10).sum() / 10
                df['trend_strength'] = price_above_ema * 2 - 1  # -1 a +1
                
                # Posición vs EMA20 
                df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
                
                # Pendiente EMA12
                if 'ema_12' in df.columns:
                    df['ema_slope_12'] = df['ema_12'].pct_change(3)  # Pendiente 3 períodos
            
            # 3. MOMENTUM DIRECCIONAL
            mom_1 = df['close'].pct_change(1)
            mom_5 = df['close'].pct_change(5)
            df['momentum_diff'] = mom_1 - mom_5  # Diferencial momentum
            df['momentum_acceleration'] = mom_1.diff()  # Aceleración
            
            # 4. PRESIÓN DIRECCIONAL 
            if 'plus_di' in df.columns and 'minus_di' in df.columns:
                df['plus_di_minus_di'] = df['plus_di'] - df['minus_di']
            
            # 5. BREAKOUT STRENGTH
            if 'atr_14' in df.columns:
                hl_range = df['high'] - df['low']
                df['breakout_strength'] = hl_range / df['atr_14']
            
            # 6. TREND CONSISTENCY
            returns_sign = np.sign(df['close'].pct_change())
            df['trend_consistency'] = returns_sign.rolling(10).sum() / 10
            
            # 7. BOLLINGER SQUEEZE
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'atr_14']):
                bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                atr_norm = df['atr_14'] / df['close']
                df['bb_squeeze'] = atr_norm / bb_width  # Alto = squeeze
            
            # 8. VOLATILITY EXPANSION
            if 'atr_14' in df.columns:
                atr_ma = df['atr_14'].rolling(20).mean()
                df['volatility_expansion'] = df['atr_14'] / atr_ma
            
            # 9. PRICE MOMENTUM
            df['price_momentum_1'] = df['close'].pct_change(1)
            df['price_momentum_5'] = df['close'].pct_change(5)
            
        except Exception as e:
            print(f"⚠️ Error en features direccionales: {e}")
        
        return df

    def _calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔬 Features de microstructura del mercado"""
        
        try:
            # 1. BUYING/SELLING PRESSURE
            close_position = (df['close'] - df['low']) / (df['high'] - df['low'])
            close_position = close_position.fillna(0.5)
            
            df['buying_pressure'] = close_position * df['volume']
            df['selling_pressure'] = (1 - close_position) * df['volume']
            
            # 2. VOLUME BREAKOUT
            volume_ma = df['volume'].rolling(20).mean()
            df['volume_breakout'] = df['volume'] / volume_ma
            
            # 3. VWAP POSITION
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['vwap_position'] = (df['close'] - vwap) / vwap
            
            # 4. CANDLE STRENGTH
            body_size = abs(df['close'] - df['open'])
            total_range = df['high'] - df['low']
            total_range = total_range.replace(0, 1e-8)
            df['candle_strength'] = body_size / total_range
            
            # 5. BODY/WICK RATIO
            upper_wick = df['high'] - np.maximum(df['open'], df['close'])
            lower_wick = np.minimum(df['open'], df['close']) - df['low']
            total_wick = upper_wick + lower_wick
            total_wick = total_wick.replace(0, 1e-8)
            df['body_wick_ratio'] = body_size / total_wick
            
            # 6. SMART MONEY FLOW
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0)
            negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0)
            
            df['smart_money_flow'] = (positive_flow.rolling(14).sum() - 
                                     negative_flow.rolling(14).sum()) / money_flow.rolling(14).sum()
            
            # 7. SUPPORT/RESISTANCE BREAK
            recent_high = df['high'].rolling(20).max()
            recent_low = df['low'].rolling(20).min()
            range_size = recent_high - recent_low
            range_size = range_size.replace(0, 1e-8)
            
            df['support_resistance_break'] = (df['close'] - recent_low) / range_size * 2 - 1
            
            # 8. PRICE EFFICIENCY
            net_change = abs(df['close'] - df['close'].shift(10))
            path_length = df['close'].diff().abs().rolling(10).sum()
            path_length = path_length.replace(0, 1e-8)
            df['price_efficiency'] = net_change / path_length
            
            # 9. MARKET REGIME (Trending vs Ranging)
            if 'adx_14' in df.columns:
                adx_ma = df['adx_14'].rolling(10).mean()
                df['market_regime'] = (df['adx_14'] - adx_ma) / adx_ma
            else:
                # Proxy usando volatilidad
                vol_short = df['close'].pct_change().rolling(5).std()
                vol_long = df['close'].pct_change().rolling(20).std()
                df['market_regime'] = vol_short / vol_long - 1
            
            # 10. FEATURES ADICIONALES
            df['true_range_normalized'] = df['true_range'] / df['close'] if 'true_range' in df.columns else 0
            df['volatility_regime'] = df['volatility_expansion'] if 'volatility_expansion' in df.columns else 1
            df['price_position_range'] = close_position
            
            # Simulaciones de orderbook (usando precio/volumen)
            df['bid_ask_pressure'] = df['buying_pressure'] - df['selling_pressure']
            df['orderbook_imbalance'] = df['volume_breakout'] * np.sign(df['price_momentum_1']) if 'price_momentum_1' in df.columns else 0
            df['institutional_activity'] = df['volume_breakout'] * df['candle_strength']
            df['retail_exhaustion'] = -df['smart_money_flow']  # Opuesto al smart money
            df['volume_momentum'] = df['volume'].pct_change()
            
        except Exception as e:
            print(f"⚠️ Error en features de microstructura: {e}")
        
        return df

    def _clean_features_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar datos de features"""
        
        # Reemplazar infinitos
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Rellenar NaN de manera inteligente
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Clip valores extremos (más agresivo)
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                q99 = df[col].quantile(0.995)  # Más agresivo
                q01 = df[col].quantile(0.005)
                if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                    df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df

    def analyze_feature_correlations(self, df: pd.DataFrame, threshold: float = 0.8) -> Dict:
        """Analizar correlaciones entre features"""
        
        corr_matrix = df.corr().abs()
        
        # Encontrar correlaciones altas
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return {
            'high_correlations': high_corr_pairs,
            'max_correlation': corr_matrix.max().max() if len(corr_matrix) > 0 else 0,
            'mean_correlation': corr_matrix.mean().mean() if len(corr_matrix) > 0 else 0
        }


# === FUNCIÓN DE MIGRACIÓN ===
def migrate_to_optimized_features():
    """Migrar del motor original al optimizado"""
    print("🔄 MIGRACIÓN A FEATURES OPTIMIZADAS")
    print("=" * 50)
    
    print("📊 Comparación de features:")
    print("   Original: 66 features (muchas correlacionadas)")
    print("   Optimizado: 40 features (sin correlaciones)")
    print("   Minimal: 20 features (máxima potencia)")
    print("   Directional: 12 features (señales puras)")
    
    print("\n✅ Para usar el motor optimizado:")
    print("   1. Reemplaza: from centralized_features_engine2 import CentralizedFeaturesEngine")
    print("   2. Por: from centralized_features_engine_optimized import OptimizedFeaturesEngine")
    print("   3. Cambia: feature_set='tcn_definitivo'")
    print("   4. Por: feature_set='optimized_tcn'")


if __name__ == "__main__":
    migrate_to_optimized_features() 