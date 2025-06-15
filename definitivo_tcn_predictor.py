#!/usr/bin/env python3
"""
🎯 DEFINITIVO TCN PREDICTOR
==========================

Predictor TCN que usa los modelos definitivos con 66 features
del repositorio experimental.

Características:
- ✅ Usa modelos definitivo_*.h5 con 66 features
- ✅ Integra centralized_features_engine.py
- ✅ Compatible con el trading manager existente
- ✅ Datos reales de Binance
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Importar nuestro motor de features
from centralized_features_engine import CentralizedFeaturesEngine


class BinanceDataProvider:
    """
    Proveedor de datos de Binance usando aiohttp para compatibilidad
    """
    
    def __init__(self):
        self.session = None
        self.base_url = "https://api.binance.com"
    
    async def __aenter__(self):
        """Inicializar sesión aiohttp"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cerrar sesión aiohttp"""
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> list:
        """
        Obtener datos de klines desde Binance API
        
        Args:
            symbol: Par de trading (ej: BTCUSDT)
            interval: Intervalo temporal
            limit: Número de velas
            
        Returns:
            Lista de datos OHLCV
        """
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    klines_raw = await response.json()
                    
                    # Convertir a formato estándar
                    klines_data = []
                    for kline in klines_raw:
                        klines_data.append({
                            'timestamp': int(kline[0]),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'close_time': int(kline[6]),
                            'quote_volume': float(kline[7]),
                            'trades': int(kline[8])
                        })
                    
                    print(f"✅ Obtenidos {len(klines_data)} klines para {symbol}")
                    return klines_data
                else:
                    print(f"❌ Error API Binance {symbol}: {response.status}")
                    return []
        
        except Exception as e:
            print(f"❌ Error obteniendo klines {symbol}: {e}")
            return []
    
    async def get_ticker_price(self, symbol: str) -> dict:
        """Obtener precio actual del símbolo"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': symbol}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            print(f"❌ Error obteniendo precio {symbol}: {e}")
            return {}


class DefinitivoTCNPredictor:
    """
    Predictor TCN que usa modelos definitivos con 66 features
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.features_engine = CentralizedFeaturesEngine()
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        print("🎯 Definitivo TCN Predictor inicializando...")
        self.load_models()
    
    def load_models(self):
        """Cargar modelos definitivos entrenados con 66 features"""
        print("="*50)
        print("🤖 Cargando Modelos Definitivos TCN...")
        print("="*50)
        
        loaded_models = []
        
        for pair in self.pairs:
            try:
                # Buscar modelo definitivo
                model_path = f"models/definitivo_{pair.lower()}.h5"
                
                # Verificar si existe
                import os
                if not os.path.exists(model_path):
                    print(f"  ⚠️  Modelo definitivo no encontrado: {model_path}")
                    print(f"     Usando modelo tcn_final como fallback para {pair}")
                    model_path = f"models/tcn_final_{pair.lower()}.h5"
                
                if os.path.exists(model_path):
                    self.models[pair] = tf.keras.models.load_model(model_path)
                    print(f"  ✅ Modelo para {pair} cargado: {model_path}")
                    
                    # Mostrar información del modelo
                    input_shape = self.models[pair].input_shape
                    print(f"     📐 Input shape: {input_shape}")
                    
                    loaded_models.append(pair)
                else:
                    print(f"  ❌ No se encontró modelo para {pair}")
                    # Crear modelo básico como fallback
                    self.models[pair] = self._create_fallback_model()
                    print(f"  🔄 Usando modelo fallback para {pair}")
                
            except Exception as e:
                print(f"  ❌ Error cargando modelo para {pair}: {e}")
                self.models[pair] = self._create_fallback_model()
                print(f"  🔄 Usando modelo fallback para {pair}")
        
        print("-" * 50)
        if loaded_models:
            print(f"👍 Modelos definitivos activos: {', '.join(loaded_models)}")
        else:
            print("🚨 ADVERTENCIA: No se cargó ningún modelo definitivo.")
            print("   Descarga los modelos definitivos del repositorio experimental.")
        print("="*50)
    
    def _create_fallback_model(self):
        """Crear modelo básico como fallback"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(66,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    async def predict_from_real_data(self, pair: str, klines_data: list) -> dict:
        """Hacer predicción usando datos reales con 66 features"""
        if pair not in self.models:
            print(f"❌ Modelo no disponible para {pair}")
            return None
        
        # Crear DataFrame de klines
        df = self._klines_to_dataframe(klines_data)
        
        if df is None or len(df) < 48:  # Solo necesitamos 48 velas para la secuencia del modelo
            print(f"  ⚠️  {pair}: Datos insuficientes para predicción ({len(df) if df is not None else 0} velas)")
            return None
        
        try:
            print(f"🔄 Calculando 66 features para {pair}...")
            
            # Usar el motor centralizado para calcular las 66 features
            features_df = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
            
            if features_df.empty:
                print(f"  ❌ {pair}: Error calculando features")
                return None
            
            print(f"  ✅ {pair}: Features calculadas: {features_df.shape}")
            
            # Normalizar features por columna
            if pair not in self.scalers:
                self.scalers[pair] = {}
                for col in features_df.columns:
                    scaler = RobustScaler()
                    self.scalers[pair][col] = scaler
                    # Fit con todos los datos disponibles
                    features_df[col] = scaler.fit_transform(features_df[col].values.reshape(-1, 1)).flatten()
            else:
                for col in features_df.columns:
                    if col in self.scalers[pair]:
                        features_df[col] = self.scalers[pair][col].transform(features_df[col].values.reshape(-1, 1)).flatten()
            
            # Determinar el shape requerido por el modelo
            model_input_shape = self.models[pair].input_shape
            
            if len(model_input_shape) == 2:  # Dense model (batch_size, features)
                # Usar última fila
                input_data = features_df.iloc[-1:].values  # Shape: (1, 66)
                
                # Verificar que tenemos las 66 features
                if input_data.shape[1] != 66:
                    print(f"  ⚠️  {pair}: Features incompletas: {input_data.shape[1]}/66")
                    if input_data.shape[1] < 66:
                        # Pad con zeros
                        padding = np.zeros((1, 66 - input_data.shape[1]))
                        input_data = np.concatenate([input_data, padding], axis=1)
                    else:
                        # Truncar a 66
                        input_data = input_data[:, :66]
                
            elif len(model_input_shape) == 3:  # LSTM/TCN model (batch_size, timesteps, features)
                timesteps = model_input_shape[1]
                expected_features = model_input_shape[2]
                
                # Usar últimas 'timesteps' filas
                if len(features_df) < timesteps:
                    print(f"  ⚠️  {pair}: Datos insuficientes para secuencia: {len(features_df)} < {timesteps}")
                    return None
                
                sequence_data = features_df.iloc[-timesteps:].values  # Shape: (timesteps, features)
                
                # Ajustar features
                if sequence_data.shape[1] != expected_features:
                    if sequence_data.shape[1] < expected_features:
                        padding = np.zeros((sequence_data.shape[0], expected_features - sequence_data.shape[1]))
                        sequence_data = np.concatenate([sequence_data, padding], axis=1)
                    else:
                        sequence_data = sequence_data[:, :expected_features]
                
                input_data = np.expand_dims(sequence_data, axis=0)  # Shape: (1, timesteps, features)
            
            else:
                print(f"  ❌ {pair}: Shape de modelo no soportado: {model_input_shape}")
                return None
            
            print(f"  📊 {pair}: Input shape final: {input_data.shape}")
            
            # Hacer predicción
            prediction = self.models[pair].predict(input_data, verbose=0)
            probabilities = prediction[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]
            
            result = {
                'pair': pair,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'features_count': input_data.shape[-1],
                'model_type': 'definitivo',
                'timestamp': datetime.now()
            }
            
            print(f"  🎯 {pair}: {signal} (confianza: {confidence:.1%})")
            return result
            
        except Exception as e:
            print(f"  ❌ Error en predicción {pair}: {e}")
            return None
    
    def _klines_to_dataframe(self, klines_data: list) -> pd.DataFrame:
        """Convertir datos de klines a DataFrame"""
        if not klines_data:
            return None
        
        try:
            # Si es una lista de listas (formato raw de Binance)
            if isinstance(klines_data[0], list):
                # Formato: [timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
                df = pd.DataFrame(klines_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convertir timestamp a datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                
                # Convertir columnas numéricas
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
            else:
                # Si ya es un DataFrame o diccionario
                df = pd.DataFrame(klines_data)
                
                # Verificar si tiene columna timestamp
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)
                elif not df.index.name == 'datetime':
                    # Usar índice numérico si no hay timestamp
                    df.reset_index(drop=True, inplace=True)
            
            # Asegurar que tenemos las columnas básicas OHLCV
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            
            if not available_cols:
                print("❌ Error: No se encontraron columnas OHLCV")
                return None
            
            df = df[available_cols].copy()
            df = df.sort_index()
            
            # Verificar y limpiar datos
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            print(f"❌ Error procesando klines: {e}")
            import traceback
            traceback.print_exc()
            return None


# === FUNCIONES DE UTILIDAD ===

async def test_definitivo_predictor():
    """Test del predictor definitivo"""
    print("🧪 TESTING DEFINITIVO TCN PREDICTOR")
    print("=" * 50)
    
    # Crear predictor
    predictor = DefinitivoTCNPredictor()
    
    # Crear proveedor de datos
    async with BinanceDataProvider() as data_provider:
        
        # Test cada par
        for pair in predictor.pairs:
            print(f"\n🔧 Testing predicción para {pair}")
            
            try:
                # Obtener datos reales
                klines = await data_provider.get_klines(pair, "1m", 200)
                
                if not klines:
                    print(f"  ❌ No se pudieron obtener datos para {pair}")
                    continue
                
                # Hacer predicción
                result = await predictor.predict_from_real_data(pair, klines)
                
                if result:
                    print(f"  ✅ Predicción exitosa:")
                    print(f"     🎯 Señal: {result['signal']}")
                    print(f"     📊 Confianza: {result['confidence']:.1%}")
                    print(f"     🔧 Features: {result['features_count']}")
                    print(f"     📈 Probabilidades:")
                    for signal, prob in result['probabilities'].items():
                        print(f"        {signal}: {prob:.1%}")
                else:
                    print(f"  ❌ Predicción falló para {pair}")
                
            except Exception as e:
                print(f"  ❌ Error en test {pair}: {e}")
    
    print(f"\n✅ Test del predictor definitivo completado")


if __name__ == "__main__":
    asyncio.run(test_definitivo_predictor()) 