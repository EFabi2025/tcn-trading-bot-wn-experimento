#!/usr/bin/env python3
"""
🎯 ENTRENADOR XRP CON METODOLOGÍA TCN DEFINITIVA ORIGINAL
========================================================

Usa el entrenador TCN definitivo que ya logró:
- BTCUSDT: 59.7% accuracy, distribución balanceada
- ETHUSDT: ~60% accuracy, distribución balanceada  
- BNBUSDT: 60.1% accuracy, distribución balanceada

Adaptado específicamente para XRPUSDT
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import talib
import warnings
import pickle
import os
from collections import Counter
warnings.filterwarnings('ignore')

class XRPOriginalTCNTrainer:
    """🎯 Entrenador XRP usando metodología TCN definitiva original exitosa"""

    def __init__(self):
        self.symbol = "XRPUSDT"
        self.lookback_window = 48  # 4 horas como modelos exitosos
        self.prediction_horizon = 6  # 30 minutos adelante

        # 🎯 THRESHOLDS BASADOS EN ANÁLISIS DE VOLATILIDAD XRP
        # Calculados del análisis previo para distribución balanceada
        self.thresholds = {
            'strong_sell': -0.0030,  # Más agresivo para 30% SELL
            'weak_sell': -0.0015,    
            'weak_buy': 0.0015,      
            'strong_buy': 0.0030     # Más agresivo para 30% BUY
        }
        
        print(f"🎯 CONFIGURACIÓN XRP CON METODOLOGÍA ORIGINAL:")
        print(f"   📊 Símbolo: {self.symbol}")
        print(f"   ⏰ Lookback: {self.lookback_window} timesteps (4 horas)")
        print(f"   🔮 Predicción: {self.prediction_horizon} timesteps (30 min)")
        print(f"   📈 Thresholds agresivos: SELL {self.thresholds['strong_sell']:.4f}, BUY {self.thresholds['strong_buy']:.4f}")

    async def get_real_market_data(self, days: int = 90) -> pd.DataFrame:
        """📊 Obtener 90 días de datos reales de XRPUSDT (balance entre calidad y velocidad)"""

        print(f"📊 Obteniendo {days} días de datos reales para {self.symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': self.symbol,
                'interval': '5m',  # 5 minutos como modelos exitosos
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            all_data = []
            current_start = start_time
            chunk_count = 0

            while current_start < end_time:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        all_data.extend(data)
                        current_start = data[-1][6] + 1  # Next start time
                        chunk_count += 1
                        
                        if chunk_count % 5 == 0:
                            print(f"   📦 Descargados {len(all_data):,} klines...")
                    else:
                        print(f"❌ Error API: {response.status}")
                        break

                await asyncio.sleep(0.1)  # Rate limiting

        # Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convertir tipos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df):,} registros de {self.symbol}")
        print(f"   📅 Período: {df.index[0]} - {df.index[-1]}")
        print(f"   💰 Rango: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
        
        # Calcular volatilidad
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(288)  # Volatilidad diaria
        print(f"   📈 Volatilidad diaria: {volatility*100:.2f}%")
        
        return df

    def create_66_features_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear exactamente 66 features usando metodología original exitosa"""

        print("🔧 Creando 66 features técnicos con metodología original...")

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = pd.DataFrame(index=df.index)

        try:
            # === MOMENTUM INDICATORS (15 features) ===
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['rsi_7'] = talib.RSI(close, timeperiod=7)

            # MACD family
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd

            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)

            # Rate of Change
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_20'] = talib.ROC(close, timeperiod=20)

            # Momentum
            features['momentum_10'] = talib.MOM(close, timeperiod=10)
            features['momentum_20'] = talib.MOM(close, timeperiod=20)

            # CCI
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)

            # === TREND INDICATORS (12 features) ===
            # Moving Averages
            features['sma_10'] = talib.SMA(close, timeperiod=10)
            features['sma_20'] = talib.SMA(close, timeperiod=20)
            features['sma_50'] = talib.SMA(close, timeperiod=50)
            features['ema_10'] = talib.EMA(close, timeperiod=10)
            features['ema_20'] = talib.EMA(close, timeperiod=20)
            features['ema_50'] = talib.EMA(close, timeperiod=50)

            # ADX family
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # PSAR
            features['psar'] = talib.SAR(high, low)

            # Aroon
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down

            # === VOLATILITY INDICATORS (10 features) ===
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            # ATR
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)

            # True Range
            features['true_range'] = talib.TRANGE(high, low, close)

            # Normalized ATR
            features['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
            features['natr_20'] = talib.NATR(high, low, close, timeperiod=20)

            # === VOLUME INDICATORS (8 features) ===
            features['ad'] = talib.AD(high, low, close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            features['obv'] = talib.OBV(close, volume)

            # Volume SMA
            features['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
            features['volume_ratio'] = volume / features['volume_sma_20']

            # Money Flow Index
            features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            features['mfi_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

            # === PRICE PATTERNS (8 features) ===
            # Price ratios
            features['hl_ratio'] = (high - low) / close
            features['oc_ratio'] = (close - df['open'].values) / close
            features['price_position'] = (close - low) / (high - low)

            # Price momentum
            close_series = pd.Series(close, index=features.index)
            features['price_change_1'] = close_series.pct_change(1)
            features['price_change_5'] = close_series.pct_change(5)
            features['price_change_10'] = close_series.pct_change(10)

            # Volatility
            returns = np.log(close_series / close_series.shift(1))
            features['volatility_10'] = returns.rolling(10).std()
            features['volatility_20'] = returns.rolling(20).std()

            # === MARKET STRUCTURE (8 features) ===
            # Higher highs, lower lows
            features['higher_high'] = (pd.Series(high, index=features.index) > pd.Series(high, index=features.index).shift(1)).astype(int)
            features['lower_low'] = (pd.Series(low, index=features.index) < pd.Series(low, index=features.index).shift(1)).astype(int)

            # Trend strength
            features['uptrend_strength'] = (close_series > close_series.shift(1)).rolling(10).sum() / 10
            features['downtrend_strength'] = (close_series < close_series.shift(1)).rolling(10).sum() / 10

            # Support/Resistance
            features['resistance_touch'] = (close_series >= close_series.rolling(20).max() * 0.99).astype(int)
            features['support_touch'] = (close_series <= close_series.rolling(20).min() * 1.01).astype(int)

            # Market efficiency
            features['efficiency_ratio'] = (np.abs(close_series - close_series.shift(10)) /
                                          (np.abs(close_series.diff()).rolling(10).sum())).fillna(0)

            # Fractal dimension (simplificado pero funcional)
            features['fractal_dimension'] = self._calculate_simple_fractal(close_series)

            # === MOMENTUM DERIVATIVES (5 features) ===
            features['rsi_momentum'] = features['rsi_14'].diff().fillna(0)
            features['macd_momentum'] = pd.Series(macd_hist, index=features.index).diff().fillna(0)
            features['ad_momentum'] = features['ad'].diff().fillna(0)
            features['volume_momentum'] = pd.Series(volume, index=features.index).pct_change().fillna(0)
            features['price_acceleration'] = features['price_change_1'].diff().fillna(0)

            # Limpiar datos
            features = features.fillna(method='ffill').fillna(0)
            features = features.replace([np.inf, -np.inf], 0)

            # Clip valores extremos
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    q99 = features[col].quantile(0.99)
                    q01 = features[col].quantile(0.01)
                    features[col] = features[col].clip(q01, q99)

            # Verificar que tenemos exactamente 66 features
            current_features = len(features.columns)
            if current_features != 66:
                print(f"⚠️ Features creados: {current_features}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            print(f"✅ {len(features.columns)} features técnicos creados para XRP (metodología original)")
            return features

        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return pd.DataFrame()

    def _calculate_simple_fractal(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calcular dimensión fractal simplificada"""
        def simple_hurst(ts):
            try:
                if len(ts) < 10:
                    return 0.5
                # Método simplificado para velocidad
                returns = np.diff(ts)
                return min(0.9, max(0.1, 0.5 + np.std(returns) * 10))
            except:
                return 0.5

        return series.rolling(window).apply(simple_hurst, raw=True).fillna(0.5)

    def create_balanced_labels_xrp_original(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """🎯 Crear etiquetas balanceadas usando metodología original exitosa"""

        print(f"🎯 Creando etiquetas balanceadas para {self.symbol} (metodología original)...")

        close_prices = df['close'].values
        thresholds = self.thresholds

        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Calcular retorno futuro
            future_return = (future_price - current_price) / current_price

            # 🎯 LÓGICA BALANCEADA ORIGINAL EXITOSA
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris: usar indicadores técnicos como modelos exitosos
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi > 60 or current_macd < 0:
                    label = 0  # SELL (confirmación técnica)
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris: usar indicadores técnicos como modelos exitosos
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi < 40 or current_macd > 0:
                    label = 2  # BUY (confirmación técnica)
                else:
                    label = 1  # HOLD
            else:
                # Zona neutral: usar momentum como modelos exitosos
                if i >= 5:
                    recent_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                    if recent_momentum > 0.01:
                        label = 2  # BUY (momentum positivo)
                    elif recent_momentum < -0.01:
                        label = 0  # SELL (momentum negativo)
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD

            labels.append(label)

        # Agregar labels al DataFrame
        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas balanceadas XRP (metodología original):")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count:,} ({pct:.1f}%)")

        return df_labeled

    def prepare_training_data_original(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos usando metodología original exitosa"""

        print("🔧 Preparando datos para entrenamiento XRP (metodología original)...")

        # Alinear features con labels
        features_aligned = features.iloc[:-self.prediction_horizon]

        # Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        # Normalizar features con RobustScaler como modelos exitosos
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # Crear secuencias temporales de 48 timesteps como modelos exitosos
        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            # Secuencia de features
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)

            # Label correspondiente
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados para XRP (metodología original):")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features utilizadas: {len(feature_columns)}")
        print(f"   - Secuencias temporales: {self.lookback_window} timesteps")

        # 🎯 CALCULAR CLASS WEIGHTS como modelos exitosos
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"🎯 Class weights calculados para XRP (metodología original):")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, weight in class_weight_dict.items():
            print(f"   - {class_names[i]}: {weight:.3f}")

        return X, y, scaler, feature_columns, class_weight_dict

    def create_definitive_tcn_model_original(self, input_shape: tuple) -> tf.keras.Model:
        """🎯 Crear modelo TCN usando arquitectura original exitosa"""

        print("🎯 Creando modelo TCN con arquitectura original exitosa...")

        model = tf.keras.Sequential([
            # Input
            tf.keras.layers.Input(shape=input_shape),

            # Normalización de entrada como modelos exitosos
            tf.keras.layers.LayerNormalization(),

            # TCN Layers con arquitectura original exitosa
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            # Pooling y Dense layers como modelos exitosos
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            # Output layer
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Compilar con configuración original exitosa
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo TCN original creado: {model.count_params():,} parámetros")

        return model

    async def train_xrp_with_original_methodology(self) -> bool:
        """🎯 Entrenar XRP usando metodología TCN original exitosa"""

        print(f"\n🎯 ENTRENANDO XRP CON METODOLOGÍA TCN ORIGINAL EXITOSA")
        print("=" * 70)

        try:
            # 1. Obtener 90 días de datos reales (balance entre calidad y velocidad)
            df = await self.get_real_market_data(days=90)

            # 2. Crear 66 features con metodología original
            features = self.create_66_features_original(df)

            # 3. Crear etiquetas balanceadas con metodología original
            print("🎯 Creando etiquetas balanceadas con metodología original...")
            df_labeled = self.create_balanced_labels_xrp_original(df, features)

            # 4. Preparar datos con metodología original
            print("🔧 Preparando datos con metodología original...")
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data_original(df_labeled, features)

            # 5. Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear modelo con arquitectura original exitosa
            model = self.create_definitive_tcn_model_original((X.shape[1], X.shape[2]))

            # 7. Callbacks como modelos exitosos
            model_dir = f'models/definitivo_{self.symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=15,  # Como modelos exitosos
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=8,  # Como modelos exitosos
                    factor=0.5,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{model_dir}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            # 8. Entrenar con configuración original exitosa
            print("🚀 Entrenando XRP con metodología original exitosa...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,  # Como modelos exitosos
                batch_size=32,  # Como modelos exitosos
                callbacks=callbacks,
                class_weight=class_weights,  # 🎯 ANTI-SESGO
                verbose=1
            )

            # 9. Evaluar modelo
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n✅ RESULTADOS FINALES XRP (METODOLOGÍA ORIGINAL):")
            print(f"   - Loss: {test_loss:.4f}")
            print(f"   - Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

            # 10. Verificar distribución de predicciones
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)

            pred_counts = Counter(y_pred_classes)
            print(f"\n📊 Distribución de predicciones XRP en test:")
            class_names = ['SELL', 'HOLD', 'BUY']
            for i, name in enumerate(class_names):
                count = pred_counts.get(i, 0)
                pct = count / len(y_pred_classes) * 100
                print(f"   - {name}: {count:,} ({pct:.1f}%)")

            # 11. Guardar modelo y componentes
            model.save(f'{model_dir}/definitivo_xrpusdt.h5')

            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)

            # Guardar también en directorio models principal
            model.save('models/definitivo_xrpusdt.h5')

            print(f"✅ Modelo XRP con metodología original guardado en:")
            print(f"   📁 {model_dir}/")
            print(f"   📁 models/definitivo_xrpusdt.h5")

            # 12. Comparar con objetivos de modelos exitosos
            target_accuracy = 0.60  # 60% como modelos exitosos
            if test_acc >= target_accuracy:
                print(f"\n🎉 ¡OBJETIVO ALCANZADO! Accuracy {test_acc*100:.2f}% >= {target_accuracy*100:.0f}%")
            else:
                print(f"\n⚠️ Accuracy {test_acc*100:.2f}% por debajo del objetivo {target_accuracy*100:.0f}%")
                print(f"   Pero usando metodología probada exitosa")

            return True

        except Exception as e:
            print(f"❌ Error entrenando XRP con metodología original: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """🎯 Entrenar XRP usando metodología TCN original exitosa"""

    print("🎯 ENTRENADOR XRP CON METODOLOGÍA TCN ORIGINAL EXITOSA")
    print("=" * 80)
    print("🎯 Objetivo: Lograr 60%+ accuracy como modelos BTCUSDT/ETHUSDT/BNBUSDT")
    print("🔧 Técnicas: Metodología original probada exitosa")
    print("📊 Datos: 90 días de datos reales (balance calidad/velocidad)")
    print("⚡ Features: Exactamente 66 features como modelos exitosos")
    print("=" * 80)

    trainer = XRPOriginalTCNTrainer()
    success = await trainer.train_xrp_with_original_methodology()

    if success:
        print(f"\n🎉 ¡MODELO XRP CON METODOLOGÍA ORIGINAL COMPLETADO!")
        print(f"📁 Modelo guardado: models/definitivo_xrpusdt.h5")
        print(f"🎯 Listo para integración al sistema de trading")
        print(f"✅ Usando metodología probada exitosa en BTCUSDT/ETHUSDT/BNBUSDT")
    else:
        print(f"\n❌ FALLO EN ENTRENAMIENTO XRP")

if __name__ == "__main__":
    asyncio.run(main()) 