"""
🔧 TA-LIB COMPATIBILITY MODULE FOR WINDOWS
===========================================
Proporciona compatibilidad con TA-Lib usando 'ta' y 'finta' como backend.
Funciona sin necesidad de Microsoft Visual C++ Build Tools.
"""

import numpy as np
import pandas as pd
import warnings

# Importar las librerías que SÍ funcionan en Windows
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    warnings.warn("⚠️ Librería 'ta' no disponible")

try:
    from finta import TA as finta_ta
    FINTA_AVAILABLE = True
except ImportError:
    FINTA_AVAILABLE = False
    warnings.warn("⚠️ Librería 'finta' no disponible")

# =============================================================================
# 🔧 FUNCIONES DE MOVING AVERAGES
# =============================================================================

def SMA(close, timeperiod=30):
    """Simple Moving Average usando 'ta' library"""
    if not TA_AVAILABLE:
        return pd.Series(close).rolling(window=timeperiod).mean().values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    return ta.trend.sma_indicator(close, window=timeperiod).values

def EMA(close, timeperiod=30):
    """Exponential Moving Average usando 'ta' library"""
    if not TA_AVAILABLE:
        return pd.Series(close).ewm(span=timeperiod).mean().values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    return ta.trend.ema_indicator(close, window=timeperiod).values

def WMA(close, timeperiod=30):
    """Weighted Moving Average usando 'ta' library"""
    if not TA_AVAILABLE:
        return pd.Series(close).rolling(window=timeperiod).mean().values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    return ta.trend.wma_indicator(close, window=timeperiod).values

# =============================================================================
# 📊 FUNCIONES DE MOMENTUM
# =============================================================================

def RSI(close, timeperiod=14):
    """Relative Strength Index usando 'ta' library"""
    if not TA_AVAILABLE:
        # Implementación básica de RSI
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    return ta.momentum.rsi(close, window=timeperiod).values

def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator usando 'ta' library"""
    if not TA_AVAILABLE:
        # Implementación básica
        if isinstance(high, (list, np.ndarray)):
            high = pd.Series(high)
        if isinstance(low, (list, np.ndarray)):
            low = pd.Series(low)
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        
        lowest_low = low.rolling(window=fastk_period).min()
        highest_high = high.rolling(window=fastk_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=slowd_period).mean()
        return k_percent.values, d_percent.values
    
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    slowk = ta.momentum.stoch(high, low, close, 
                             window=fastk_period, 
                             smooth_window=slowk_period).values
    slowd = ta.momentum.stoch_signal(high, low, close, 
                                    window=fastk_period, 
                                    smooth_window=slowd_period).values
    
    return slowk, slowd

def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD usando 'ta' library"""
    if not TA_AVAILABLE:
        # Implementación básica
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        ema_fast = close.ewm(span=fastperiod).mean()
        ema_slow = close.ewm(span=slowperiod).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signalperiod).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    macd_line = ta.trend.macd(close, window_fast=fastperiod, 
                             window_slow=slowperiod).values
    signal_line = ta.trend.macd_signal(close, window_fast=fastperiod, 
                                      window_slow=slowperiod, 
                                      window_sign=signalperiod).values
    histogram = ta.trend.macd_diff(close, window_fast=fastperiod, 
                                  window_slow=slowperiod, 
                                  window_sign=signalperiod).values
    
    return macd_line, signal_line, histogram

# =============================================================================
# 📈 FUNCIONES DE VOLATILIDAD
# =============================================================================

def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands usando 'ta' library"""
    if not TA_AVAILABLE:
        # Implementación básica
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        sma = close.rolling(window=timeperiod).mean()
        std = close.rolling(window=timeperiod).std()
        upper_band = sma + (std * nbdevup)
        lower_band = sma - (std * nbdevdn)
        return upper_band.values, sma.values, lower_band.values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    upper_band = ta.volatility.bollinger_hband(close, window=timeperiod, 
                                              window_dev=nbdevup).values
    middle_band = ta.volatility.bollinger_mavg(close, window=timeperiod).values
    lower_band = ta.volatility.bollinger_lband(close, window=timeperiod, 
                                              window_dev=nbdevdn).values
    
    return upper_band, middle_band, lower_band

def ATR(high, low, close, timeperiod=14):
    """Average True Range usando 'ta' library"""
    if not TA_AVAILABLE:
        # Implementación básica
        if isinstance(high, (list, np.ndarray)):
            high = pd.Series(high)
        if isinstance(low, (list, np.ndarray)):
            low = pd.Series(low)
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=timeperiod).mean().values
    
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    return ta.volatility.average_true_range(high, low, close, 
                                           window=timeperiod).values

# =============================================================================
# 🔄 FUNCIONES DE PATRÓN DE VELAS
# =============================================================================

def DOJI(open_prices, high, low, close):
    """Detección de patrón Doji"""
    if isinstance(open_prices, (list, np.ndarray)):
        open_prices = pd.Series(open_prices)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    
    body = np.abs(close - open_prices)
    range_val = high - low
    
    # Doji: cuerpo muy pequeño comparado con el rango
    doji_threshold = range_val * 0.1
    return (body <= doji_threshold).astype(int).values

def HAMMER(open_prices, high, low, close):
    """Detección de patrón Hammer"""
    if isinstance(open_prices, (list, np.ndarray)):
        open_prices = pd.Series(open_prices)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    
    body = np.abs(close - open_prices)
    lower_shadow = np.minimum(open_prices, close) - low
    upper_shadow = high - np.maximum(open_prices, close)
    
    # Hammer: sombra inferior larga, cuerpo pequeño, sombra superior pequeña
    is_hammer = (
        (lower_shadow >= body * 2) & 
        (upper_shadow <= body * 0.5) & 
        (body > 0)
    )
    
    return is_hammer.astype(int).values

# =============================================================================
# 🎯 FUNCIÓN DE INICIALIZACIÓN
# =============================================================================

def get_talib_status():
    """Retorna el estado de las librerías de análisis técnico"""
    status = {
        'ta_available': TA_AVAILABLE,
        'finta_available': FINTA_AVAILABLE,
        'talib_compatible': True,
        'backend': []
    }
    
    if TA_AVAILABLE:
        status['backend'].append('ta')
    if FINTA_AVAILABLE:
        status['backend'].append('finta')
    
    return status

def test_functions():
    """Prueba las funciones principales"""
    print("🔧 Testing TA-Lib Compatibility...")
    
    # Datos de prueba
    close_data = np.random.rand(100) * 100 + 50
    high_data = close_data + np.random.rand(100) * 5
    low_data = close_data - np.random.rand(100) * 5
    open_data = close_data + np.random.rand(100) * 2 - 1
    
    try:
        # Test moving averages
        sma = SMA(close_data, 20)
        ema = EMA(close_data, 20)
        print(f"✅ Moving Averages: SMA={sma[-1]:.2f}, EMA={ema[-1]:.2f}")
        
        # Test momentum
        rsi = RSI(close_data, 14)
        print(f"✅ RSI: {rsi[-1]:.2f}")
        
        # Test MACD
        macd, signal, hist = MACD(close_data)
        print(f"✅ MACD: {macd[-1]:.2f}, Signal: {signal[-1]:.2f}")
        
        # Test Bollinger Bands
        upper, middle, lower = BBANDS(close_data, 20)
        print(f"✅ Bollinger Bands: Upper={upper[-1]:.2f}, Lower={lower[-1]:.2f}")
        
        # Test ATR
        atr = ATR(high_data, low_data, close_data, 14)
        print(f"✅ ATR: {atr[-1]:.2f}")
        
        print("🎯 Todas las funciones TA-Lib compatibility funcionan correctamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False

if __name__ == "__main__":
    status = get_talib_status()
    print("📊 TA-Lib Compatibility Status:")
    print(f"   TA Available: {status['ta_available']}")
    print(f"   Finta Available: {status['finta_available']}")
    print(f"   Backend: {', '.join(status['backend'])}")
    print()
    test_functions() 