"""
TA-LIB REPLACEMENT MODULE FOR WINDOWS
====================================
Modulo que reemplaza TA-Lib usando 'ta' library para Windows.
"""

import numpy as np
import pandas as pd
import warnings

# Importar la libreria 'ta' que funciona en Windows
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    warnings.warn("Libreria 'ta' no disponible")

def SMA(close, timeperiod=30):
    """Simple Moving Average"""
    if not TA_AVAILABLE:
        return pd.Series(close).rolling(window=timeperiod).mean().values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    result = ta.trend.sma_indicator(close, window=timeperiod)
    return result.fillna(method='bfill').values

def EMA(close, timeperiod=30):
    """Exponential Moving Average"""
    if not TA_AVAILABLE:
        return pd.Series(close).ewm(span=timeperiod).mean().values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    result = ta.trend.ema_indicator(close, window=timeperiod)
    return result.fillna(method='bfill').values

def RSI(close, timeperiod=14):
    """Relative Strength Index"""
    if not TA_AVAILABLE:
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        result = 100 - (100 / (1 + rs))
        return result.fillna(50).values
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    result = ta.momentum.rsi(close, window=timeperiod)
    return result.fillna(50).values

def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD Indicator"""
    if not TA_AVAILABLE:
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        ema_fast = close.ewm(span=fastperiod).mean()
        ema_slow = close.ewm(span=slowperiod).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signalperiod).mean()
        histogram = macd_line - signal_line
        return (macd_line.fillna(0).values, 
                signal_line.fillna(0).values, 
                histogram.fillna(0).values)
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    macd_line = ta.trend.macd(close, window_fast=fastperiod, 
                             window_slow=slowperiod)
    signal_line = ta.trend.macd_signal(close, window_fast=fastperiod, 
                                      window_slow=slowperiod, 
                                      window_sign=signalperiod)
    histogram = ta.trend.macd_diff(close, window_fast=fastperiod, 
                                  window_slow=slowperiod, 
                                  window_sign=signalperiod)
    
    return (macd_line.fillna(0).values, 
            signal_line.fillna(0).values, 
            histogram.fillna(0).values)

def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    if not TA_AVAILABLE:
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        sma = close.rolling(window=timeperiod).mean()
        std = close.rolling(window=timeperiod).std()
        upper_band = sma + (std * nbdevup)
        lower_band = sma - (std * nbdevdn)
        return (upper_band.fillna(method='bfill').values, 
                sma.fillna(method='bfill').values, 
                lower_band.fillna(method='bfill').values)
    
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    upper_band = ta.volatility.bollinger_hband(close, window=timeperiod, 
                                              window_dev=nbdevup)
    middle_band = ta.volatility.bollinger_mavg(close, window=timeperiod)
    lower_band = ta.volatility.bollinger_lband(close, window=timeperiod, 
                                              window_dev=nbdevdn)
    
    return (upper_band.fillna(method='bfill').values, 
            middle_band.fillna(method='bfill').values, 
            lower_band.fillna(method='bfill').values)

def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    if not TA_AVAILABLE:
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
        result = true_range.rolling(window=timeperiod).mean()
        return result.fillna(method='bfill').values
    
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    result = ta.volatility.average_true_range(high, low, close, 
                                           window=timeperiod)
    return result.fillna(method='bfill').values

def CCI(high, low, close, timeperiod=14):
    """Commodity Channel Index"""
    if not TA_AVAILABLE:
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        result = pd.Series(close).rolling(window=timeperiod).mean()
        return result.fillna(0).values
    
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    result = ta.trend.cci(high, low, close, window=timeperiod)
    return result.fillna(0).values

def ADX(high, low, close, timeperiod=14):
    """Average Directional Movement Index"""
    if not TA_AVAILABLE:
        if isinstance(close, (list, np.ndarray)):
            close = pd.Series(close)
        result = pd.Series(close).rolling(window=timeperiod).mean()
        return result.fillna(25).values
    
    if isinstance(high, (list, np.ndarray)):
        high = pd.Series(high)
    if isinstance(low, (list, np.ndarray)):
        low = pd.Series(low)
    if isinstance(close, (list, np.ndarray)):
        close = pd.Series(close)
    
    result = ta.trend.adx(high, low, close, window=timeperiod)
    return result.fillna(25).values

# Mensaje de inicializacion
if TA_AVAILABLE:
    print("TA-Lib Replacement Module cargado exitosamente")
    print("Backend: ta library")
    print("Functions: SMA, EMA, RSI, MACD, BBANDS, ATR, CCI, ADX disponibles")
else:
    print("TA-Lib Replacement Module cargado con pandas fallback")
    print("Functions disponibles con funcionalidad limitada") 