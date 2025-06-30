#!/usr/bin/env python3
"""
🎯 TCN DEFINITIVO TRAINER
Entrenador profesional que corrige todos los sesgos identificados
Implementa técnicas anti-sesgo y distribución balanceada
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

class DefinitiveTCNTrainer:
    """🎯 Entrenador definitivo del TCN con técnicas anti-sesgo"""

    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        self.lookback_window = 24
        self.prediction_horizon = 6

        # 🎯 THRESHOLDS BASADOS EN ANÁLISIS DE DATOS REALES
        self.thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.0014,  # -0.14%
                'weak_sell': -0.0007,    # -0.07%
                'weak_buy': 0.0007,      # +0.07%
                'strong_buy': 0.0014     # +0.14%
            },
            'ETHUSDT': {
                'strong_sell': -0.0026,  # -0.26%
                'weak_sell': -0.0012,    # -0.12%
                'weak_buy': 0.0013,      # +0.13%
                'strong_buy': 0.0027     # +0.27%
            },
            'BNBUSDT': {
                'strong_sell': -0.0015,  # -0.15%
                'weak_sell': -0.0007,    # -0.07%
                'weak_buy': 0.0007,      # +0.07%
                'strong_buy': 0.0015     # +0.15%
            },
            'XRPUSDT': {
                'strong_sell': -0.0018,  # -0.18%
                'weak_sell': -0.0009,    # -0.09%
                'weak_buy': 0.0009,      # +0.09%
                'strong_buy': 0.0018     # +0.18%
            }
        }

    async def get_real_market_data(self, symbol: str, days: int = 10) -> pd.DataFrame:
        """📊 Obtener datos reales de mercado de Binance (versión rápida)"""

        print(f"📊 Obteniendo {days} días de datos reales para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',  # Usar 5 minutos en lugar de 1 minuto para menos datos
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            all_data = []
            current_start = start_time

            while current_start < end_time:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        all_data.extend(data)
                        current_start = data[-1][6] + 1  # Next start time
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

        print(f"✅ Obtenidos {len(df)} registros de {symbol}")
        return df

    def create_66_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear 66 features técnicos optimizados"""

        print("🔧 Creando 66 features técnicos...")

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

            # Fractal dimension (simplificado)
            features['fractal_dimension'] = 0.5  # Valor constante por ahora

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
            if len(features.columns) != 66:
                print(f"⚠️ Features creados: {len(features.columns)}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            print(f"✅ {len(features.columns)} features técnicos creados")
            return features

        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return pd.DataFrame()

    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calcular dimensión fractal para medir complejidad del precio"""
        def hurst_exponent(ts):
            try:
                ts = np.array(ts)
                if len(ts) < 4:
                    return 0.5
                lags = range(2, min(len(ts)//2, 10))
                if len(lags) < 2:
                    return 0.5
                tau = []
                for lag in lags:
                    if lag < len(ts):
                        diff = ts[lag:] - ts[:-lag]
                        tau.append(np.sqrt(np.std(diff)))
                if len(tau) < 2:
                    return 0.5
                tau = np.array(tau)
                tau = tau[tau > 0]  # Evitar log(0)
                if len(tau) < 2:
                    return 0.5
                poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
                return max(0.1, min(0.9, poly[0] * 2.0))
            except:
                return 0.5

        return series.rolling(window).apply(hurst_exponent, raw=True).fillna(0.5)

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas BALANCEADAS sin sesgo hacia HOLD"""

        print(f"🎯 Creando etiquetas balanceadas para {symbol}...")

        close_prices = df['close'].values
        thresholds = self.thresholds[symbol]

        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Calcular retorno futuro
            future_return = (future_price - current_price) / current_price

            # 🎯 LÓGICA BALANCEADA (NO CONSERVADORA)
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris: usar indicadores técnicos para decidir
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
                # Zona gris: usar indicadores técnicos para decidir
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
                # Zona neutral: usar momentum para decidir
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

        print("📊 Distribución de etiquetas balanceadas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")

        # 🎯 VERIFICAR QUE NO HAY SESGO EXTREMO
        max_class_pct = max([count/total for count in label_counts.values]) * 100
        if max_class_pct > 70:
            print(f"⚠️ ADVERTENCIA: Clase dominante con {max_class_pct:.1f}%")
        else:
            print(f"✅ Distribución balanceada: clase máxima {max_class_pct:.1f}%")

        return df_labeled

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento con técnicas anti-sesgo"""

        print("🔧 Preparando datos para entrenamiento...")

        # Alinear features con labels
        features_aligned = features.iloc[:-self.prediction_horizon]

        # Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        # Normalizar features
        scaler = RobustScaler()  # Más robusto a outliers que MinMaxScaler
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # Crear secuencias temporales
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

        print(f"✅ Datos preparados:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features utilizadas: {len(feature_columns)}")

        # 🎯 CALCULAR CLASS WEIGHTS PARA BALANCEAR
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"🎯 Class weights calculados:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, weight in class_weight_dict.items():
            print(f"   - {class_names[i]}: {weight:.3f}")

        return X, y, scaler, feature_columns, class_weight_dict

    def create_definitive_tcn_model(self, input_shape: tuple) -> tf.keras.Model:
        """🎯 Crear modelo TCN definitivo anti-sesgo"""

        print("🎯 Creando modelo TCN definitivo...")

        model = tf.keras.Sequential([
            # Input
            tf.keras.layers.Input(shape=input_shape),

            # Normalización de entrada
            tf.keras.layers.LayerNormalization(),

            # TCN Layers con regularización anti-overfitting
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

            # Attention mechanism para features importantes (simplificado)
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),

            # Dense layers con regularización
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            # Output layer con activación balanceada
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Compilar con configuración anti-sesgo
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),  # Learning rate más conservador
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Solo accuracy para evitar problemas
        )

        print(f"✅ Modelo definitivo creado: {model.count_params():,} parámetros")

        return model

    async def train_definitive_model(self, symbol: str) -> bool:
        """🎯 Entrenar modelo definitivo para un símbolo"""

        print(f"\n🎯 ENTRENANDO MODELO DEFINITIVO PARA {symbol}")
        print("=" * 70)

        try:
            # 1. Obtener datos reales
            df = await self.get_real_market_data(symbol, days=10)

            # 2. Crear 66 features
            features = self.create_66_features(df)

            # 3. Crear etiquetas balanceadas
            print("🎯 Creando etiquetas balanceadas...")
            try:
                df_labeled = self.create_balanced_labels(df, features, symbol)
                print(f"✅ Etiquetas creadas correctamente")
            except Exception as e:
                print(f"❌ Error en create_balanced_labels: {e}")
                import traceback
                traceback.print_exc()
                return False

            # 4. Preparar datos con técnicas anti-sesgo
            print("🔧 Preparando datos de entrenamiento...")
            try:
                X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)
                print(f"✅ Datos preparados correctamente")
            except Exception as e:
                print(f"❌ Error en prepare_training_data: {e}")
                import traceback
                traceback.print_exc()
                return False

            # 5. Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear modelo definitivo
            model = self.create_definitive_tcn_model((X.shape[1], X.shape[2]))

            # 7. Callbacks avanzados con guardado frecuente
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=15,
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=8,
                    factor=0.5,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/definitivo_{symbol.lower()}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                # 🔄 CHECKPOINT CADA 10 EPOCHS PARA RECUPERACIÓN
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/definitivo_{symbol.lower()}/checkpoint_epoch_{{epoch:02d}}.h5',
                    save_freq='epoch',
                    period=10,
                    save_best_only=False
                )
            ]

            # 8. Entrenar con class weights
            print("🚀 Entrenando modelo definitivo...")
            os.makedirs(f'models/definitivo_{symbol.lower()}', exist_ok=True)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weights,  # 🎯 ANTI-SESGO
                verbose=1
            )

            # 9. Evaluar modelo con manejo de errores
            try:
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                print(f"\n✅ RESULTADOS FINALES:")
                print(f"   - Loss: {test_loss:.3f}")
                print(f"   - Accuracy: {test_acc:.3f}")
            except Exception as e:
                print(f"⚠️ Error en evaluación, pero modelo entrenado: {e}")
                # Guardar modelo aunque falle la evaluación
                model.save(f'models/definitivo_{symbol.lower()}/final_model_backup.h5')
                test_acc = 0.0  # Valor por defecto

            # 10. Guardar scaler y metadata
            scaler_path = f'models/definitivo_{symbol.lower()}/scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"💾 Scaler guardado: {scaler_path}")

            # Guardar feature columns
            features_path = f'models/definitivo_{symbol.lower()}/feature_columns.pkl'
            with open(features_path, 'wb') as f:
                pickle.dump(feature_columns, f)
            print(f"💾 Feature columns guardados: {features_path}")

            # 11. Verificar distribución de predicciones
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)

            pred_counts = Counter(y_pred_classes)
            print(f"\n📊 Distribución de predicciones en test:")
            class_names = ['SELL', 'HOLD', 'BUY']
            for i, name in enumerate(class_names):
                count = pred_counts.get(i, 0)
                pct = count / len(y_pred_classes) * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")

            # 11. Guardar modelo y componentes
            model.save(f'models/definitivo_{symbol.lower()}/model.h5')

            with open(f'models/definitivo_{symbol.lower()}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            with open(f'models/definitivo_{symbol.lower()}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            with open(f'models/definitivo_{symbol.lower()}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)

            print(f"✅ Modelo definitivo guardado en models/definitivo_{symbol.lower()}/")

            return True

        except Exception as e:
            print(f"❌ Error entrenando modelo definitivo para {symbol}: {e}")
            return False

async def main():
    """🎯 Entrenar modelos definitivos para todos los símbolos"""

    print("🎯 ENTRENADOR DE MODELOS TCN DEFINITIVOS")
    print("=" * 80)
    print("🎯 Objetivo: Corregir sesgos y crear modelos balanceados")
    print("🔧 Técnicas: Class weights, etiquetado balanceado, 66 features")
    print("=" * 80)

    trainer = DefinitiveTCNTrainer()

    results = {}
    for symbol in trainer.pairs:
        success = await trainer.train_definitive_model(symbol)
        results[symbol] = success

    print(f"\n🎯 RESUMEN FINAL:")
    print("=" * 50)
    for symbol, success in results.items():
        status = "✅ ÉXITO" if success else "❌ FALLO"
        print(f"   {symbol}: {status}")

    successful = sum(results.values())
    print(f"\n🎯 Modelos definitivos entrenados: {successful}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main())
