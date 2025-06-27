#!/usr/bin/env python3
"""
REAL BINANCE PREDICTOR - Predicciones reales con datos reales de Binance
Sistema que usa datos reales del mercado para generar predicciones TCN
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ✅ IMPORTAR CLIENTE BINANCE PARA ÓRDENES REALES
from binance.client import Client
from config import trading_config

class BinanceDataProvider:
    """Proveedor de datos reales de Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = None
        
        # ✅ AGREGAR CLIENTE BINANCE PARA ÓRDENES REALES
        self.client = Client(
            api_key=trading_config.BINANCE_API_KEY,
            api_secret=trading_config.BINANCE_SECRET_KEY
        )
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> list:
        """Obtener datos reales de velas de Binance"""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [{
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5]),
                    'close_time': int(item[6]),
                    'quote_volume': float(item[7]),
                    'count': int(item[8])
                } for item in data]
            else:
                print(f"❌ Error obteniendo datos de {symbol}: {response.status}")
                return []
    
    async def get_ticker_price(self, symbol: str) -> dict:
        """Obtener precio actual"""
        url = f"{self.base_url}/api/v3/ticker/price"
        params = {"symbol": symbol}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            return {}
    
    async def get_24hr_ticker(self, symbol: str) -> dict:
        """Obtener estadísticas 24h"""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {"symbol": symbol}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            return {}

class RealTCNPredictor:
    """Predictor TCN que usa datos reales"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.load_models()
    
    def load_models(self):
        """Cargar modelos entrenados"""
        print("📦 Cargando modelos TCN entrenados...")
        
        for pair in self.pairs:
            try:
                model_path = f"models/tcn_final_{pair.lower()}.h5"
                self.models[pair] = tf.keras.models.load_model(model_path)
                print(f"  ✅ {pair}: Modelo cargado")
            except Exception as e:
                print(f"  ⚠️  {pair}: Error cargando modelo - {e}")
                # Crear modelo básico si no existe
                self.models[pair] = self._create_basic_model()
                print(f"  🔧 {pair}: Usando modelo básico")
    
    def _create_basic_model(self):
        """Crear modelo básico si no existe el entrenado"""
        try:
            from tcn_final_ready import FinalReadyTCN
            tcn = FinalReadyTCN()
            input_shape = (50, 21)  # 50 timesteps, 21 features
            return tcn.build_confidence_model(input_shape)
        except:
            # Modelo muy básico como fallback
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(50, 21)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
    
    def create_features_from_klines(self, klines_data: list) -> pd.DataFrame:
        """Crear features técnicos desde datos de klines reales"""
        if len(klines_data) < 100:
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = pd.DataFrame(klines_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Ordenar por tiempo
        df = df.sort_index()
        
        features = pd.DataFrame(index=df.index)
        
        print(f"📊 Creando features técnicos desde {len(df)} velas reales...")
        
        # 1. Returns en múltiples períodos
        for period in [1, 3, 5, 10, 20]:
            returns = df['close'].pct_change(period)
            features[f'returns_{period}'] = returns
            features[f'returns_{period}_ma'] = returns.rolling(5).mean()
        
        # 2. Volatilidad
        for window in [10, 20, 50]:
            vol = df['close'].pct_change().rolling(window).std()
            features[f'volatility_{window}'] = vol
        
        # 3. Tendencias (SMA)
        for short, long in [(10, 30), (20, 60)]:
            sma_short = df['close'].rolling(short).mean()
            sma_long = df['close'].rolling(long).mean()
            features[f'sma_trend_{short}_{long}'] = (sma_short - sma_long) / df['close']
        
        # 4. RSI
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features['rsi_deviation'] = abs(features['rsi_14'] - 50) / 50
        
        # 5. MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / df['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 6. Bollinger Bands
        bb_period = 20
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        features['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std)
        features['bb_width'] = (bb_std * 4) / bb_middle
        
        # 7. Volume indicators
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # Limpiar NaN
        features = features.fillna(method='ffill').fillna(0)
        
        print(f"  ✅ {len(features.columns)} features creados")
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcular RSI real"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def predict_from_real_data(self, pair: str, klines_data: list) -> dict:
        """Hacer predicción usando datos reales"""
        if pair not in self.models:
            return None
        
        # Crear features
        features = self.create_features_from_klines(klines_data)
        if features.empty or len(features) < 50:
            print(f"  ⚠️  {pair}: Datos insuficientes para predicción")
            return None
        
        try:
            # Normalizar features
            if pair not in self.scalers:
                self.scalers[pair] = {}
                for col in features.columns:
                    scaler = RobustScaler()
                    self.scalers[pair][col] = scaler
                    # Fit con todos los datos disponibles
                    features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
            else:
                for col in features.columns:
                    if col in self.scalers[pair]:
                        features[col] = self.scalers[pair][col].transform(features[col].values.reshape(-1, 1)).flatten()
            
            # Tomar últimas 50 observaciones para secuencia
            sequence = features.iloc[-50:].values
            
            # Ajustar a 21 features si es necesario
            if sequence.shape[1] > 21:
                sequence = sequence[:, :21]
            elif sequence.shape[1] < 21:
                # Pad con zeros si faltan features
                padding = np.zeros((sequence.shape[0], 21 - sequence.shape[1]))
                sequence = np.concatenate([sequence, padding], axis=1)
            
            sequence = np.expand_dims(sequence, axis=0)
            
            # Predicción
            prediction = self.models[pair].predict(sequence, verbose=0)
            probabilities = prediction[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]
            
            return {
                'pair': pair,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'features_count': sequence.shape[1],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"  ❌ Error en predicción {pair}: {e}")
            return None

class RealMarketAnalyzer:
    """Analizador de mercado real"""
    
    def __init__(self):
        self.data_provider = None
        self.predictor = RealTCNPredictor()
    
    async def analyze_real_market(self, pairs: list = None):
        """Analizar mercado real y hacer predicciones"""
        if pairs is None:
            pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        print("🔄 INICIANDO ANÁLISIS DE MERCADO REAL")
        print("="*60)
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Pares: {', '.join(pairs)}")
        print("="*60)
        
        async with BinanceDataProvider() as provider:
            self.data_provider = provider
            
            for pair in pairs:
                await self.analyze_pair(pair)
                print("-" * 40)
    
    async def analyze_pair(self, pair: str):
        """Analizar un par específico"""
        print(f"\\n📈 ANALIZANDO {pair}")
        
        try:
            # Obtener datos reales
            print("  🔄 Obteniendo datos de Binance...")
            klines = await self.data_provider.get_klines(pair, "1m", 500)
            ticker_24h = await self.data_provider.get_24hr_ticker(pair)
            
            if not klines:
                print(f"  ❌ No se pudieron obtener datos para {pair}")
                return
            
            # Información actual del mercado
            current_price = float(klines[-1]['close'])
            price_change_24h = float(ticker_24h.get('priceChangePercent', 0))
            volume_24h = float(ticker_24h.get('volume', 0))
            
            print(f"  💰 Precio actual: ${current_price:,.2f}")
            print(f"  📊 Cambio 24h: {price_change_24h:+.2f}%")
            print(f"  📈 Volumen 24h: {volume_24h:,.0f}")
            
            # Hacer predicción con datos reales
            print("  🧠 Generando predicción TCN...")
            prediction = await self.predictor.predict_from_real_data(pair, klines)
            
            if prediction:
                print(f"  🎯 PREDICCIÓN:")
                print(f"     Señal: {prediction['signal']}")
                print(f"     Confianza: {prediction['confidence']:.3f}")
                print(f"     Probabilidades:")
                for signal, prob in prediction['probabilities'].items():
                    print(f"       {signal}: {prob:.3f}")
                
                # Análisis de confianza
                if prediction['confidence'] >= 0.80:
                    print(f"  🟢 ALTA CONFIANZA - Señal fuerte")
                elif prediction['confidence'] >= 0.65:
                    print(f"  🟡 CONFIANZA MEDIA - Señal moderada")
                else:
                    print(f"  🔴 BAJA CONFIANZA - Señal débil")
                
                # Análisis técnico básico
                recent_prices = [float(k['close']) for k in klines[-20:]]
                sma_20 = np.mean(recent_prices)
                price_vs_sma = (current_price - sma_20) / sma_20 * 100
                
                print(f"  📊 ANÁLISIS TÉCNICO:")
                print(f"     SMA 20: ${sma_20:.2f}")
                print(f"     Precio vs SMA20: {price_vs_sma:+.2f}%")
                
                # Volatilidad reciente
                returns = [klines[i]['close'] / klines[i-1]['close'] - 1 for i in range(1, len(klines))]
                volatility = np.std(returns[-20:]) * 100
                print(f"     Volatilidad 20min: {volatility:.2f}%")
                
            else:
                print(f"  ❌ No se pudo generar predicción")
                
        except Exception as e:
            print(f"  ❌ Error analizando {pair}: {e}")

async def continuous_monitoring(duration_minutes: int = 10):
    """Monitoreo continuo del mercado"""
    print("🔄 INICIANDO MONITOREO CONTINUO")
    print(f"⏱️  Duración: {duration_minutes} minutos")
    print(f"🔄 Actualizaciones cada 60 segundos")
    print("="*60)
    
    analyzer = RealMarketAnalyzer()
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    cycle = 0
    
    try:
        while datetime.now() < end_time:
            cycle += 1
            print(f"\\n\\n🔄 CICLO {cycle} - {datetime.now().strftime('%H:%M:%S')}")
            
            await analyzer.analyze_real_market()
            
            if datetime.now() < end_time:
                print(f"\\n⏳ Esperando 60 segundos hasta próximo análisis...")
                await asyncio.sleep(60)
        
        print(f"\\n✅ Monitoreo completado después de {duration_minutes} minutos")
        
    except KeyboardInterrupt:
        print(f"\\n⏹️  Monitoreo detenido por usuario")

async def single_analysis():
    """Análisis único del mercado"""
    analyzer = RealMarketAnalyzer()
    await analyzer.analyze_real_market()

async def main():
    """Función principal"""
    print("🎯 REAL BINANCE TCN PREDICTOR")
    print("Sistema de predicción con datos reales de Binance")
    print()
    
    print("Selecciona modo de operación:")
    print("1. Análisis único")
    print("2. Monitoreo continuo (10 minutos)")
    print("3. Monitoreo extendido (30 minutos)")
    
    try:
        # Para demo automático, usar análisis único
        print("\\n🚀 Ejecutando análisis único...")
        await single_analysis()
        
        print("\\n" + "="*60)
        print("✅ ANÁLISIS COMPLETADO")
        print("🔄 Para monitoreo continuo, ejecutar con modo 2 o 3")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 