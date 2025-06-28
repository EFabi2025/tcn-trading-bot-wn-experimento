#!/usr/bin/env python3
"""
🎯 TCN DEFINITIVO TRAINER PARA XRPUSDT
Entrenador profesional que corrige todos los sesgos identificados
Implementa técnicas anti-sesgo y distribución balanceada específicamente para XRP
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

class XRPDefinitiveTCNTrainer:
    """🎯 Entrenador definitivo del TCN para XRPUSDT con técnicas anti-sesgo"""

    def __init__(self):
        self.symbol = "XRPUSDT"
        self.lookback_window = 48  # 4 horas como documentado (48 * 5min = 4h)
        self.prediction_horizon = 6  # 30 minutos adelante (6 * 5min = 30min)

        # 🎯 THRESHOLDS OPTIMIZADOS PARA XRP (basados en análisis de volatilidad)
        self.thresholds = {
            'strong_sell': -0.0022,  # -0.22% (más agresivo que análisis inicial)
            'weak_sell': -0.0011,    # -0.11%
            'weak_buy': 0.0011,      # +0.11%
            'strong_buy': 0.0022     # +0.22%
        }
        
        print(f"🎯 CONFIGURACIÓN XRP DEFINITIVA:")
        print(f"   📊 Símbolo: {self.symbol}")
        print(f"   ⏰ Lookback: {self.lookback_window} timesteps (4 horas)")
        print(f"   🔮 Predicción: {self.prediction_horizon} timesteps (30 min)")
        print(f"   📈 Thresholds: SELL {self.thresholds['strong_sell']:.4f}, BUY {self.thresholds['strong_buy']:.4f}")

    async def get_real_market_data(self, days: int = 30) -> pd.DataFrame:
        """📊 Obtener 30 días de datos reales de XRPUSDT de Binance (optimizado)"""

        print(f"📊 Obteniendo {days} días de datos reales para {self.symbol}... (modo rápido)")

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
                        
                        if chunk_count % 10 == 0:
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

    def create_essential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear exactamente 66 features técnicos como modelos exitosos"""

        print("🔧 Creando 66 features técnicos definitivos...")

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
            features['fractal_dimension'] = self._calculate_fractal_dimension(close_series)

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

            print(f"✅ {len(features.columns)} features técnicos creados para XRP")
            return features

        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return pd.DataFrame()

    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calcular dimensión fractal para medir complejidad del precio XRP"""
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

    def create_balanced_labels_xrp(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """🎯 Crear etiquetas BALANCEADAS específicas para XRP sin sesgo hacia HOLD"""

        print(f"🎯 Creando etiquetas balanceadas para {self.symbol}...")

        close_prices = df['close'].values
        thresholds = self.thresholds

        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Calcular retorno futuro
            future_return = (future_price - current_price) / current_price

            # 🎯 LÓGICA BALANCEADA ESPECÍFICA PARA XRP
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris: usar indicadores técnicos específicos de XRP
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                    current_bb_pos = features['bb_position'].iloc[i] if i < len(features) else 0.5
                except:
                    current_rsi = 50
                    current_macd = 0
                    current_bb_pos = 0.5

                # XRP tiende a ser más volátil, usar criterios más agresivos
                if current_rsi > 65 or current_macd < 0 or current_bb_pos > 0.8:
                    label = 0  # SELL (confirmación técnica)
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris: usar indicadores técnicos específicos de XRP
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                    current_bb_pos = features['bb_position'].iloc[i] if i < len(features) else 0.5
                except:
                    current_rsi = 50
                    current_macd = 0
                    current_bb_pos = 0.5

                # XRP: criterios más agresivos para BUY
                if current_rsi < 35 or current_macd > 0 or current_bb_pos < 0.2:
                    label = 2  # BUY (confirmación técnica)
                else:
                    label = 1  # HOLD
            else:
                # Zona neutral: usar momentum específico de XRP
                if i >= 5:
                    recent_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                    # XRP es más volátil, usar thresholds más bajos
                    if recent_momentum > 0.008:  # 0.8% en lugar de 1%
                        label = 2  # BUY (momentum positivo)
                    elif recent_momentum < -0.008:  # -0.8% en lugar de -1%
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

        print("📊 Distribución de etiquetas balanceadas XRP:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count:,} ({pct:.1f}%)")

        # 🎯 VERIFICAR QUE NO HAY SESGO EXTREMO
        max_class_pct = max([count/total for count in label_counts.values]) * 100
        if max_class_pct > 70:
            print(f"⚠️ ADVERTENCIA: Clase dominante con {max_class_pct:.1f}%")
        else:
            print(f"✅ Distribución balanceada: clase máxima {max_class_pct:.1f}%")

        return df_labeled

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento con técnicas anti-sesgo"""

        print("🔧 Preparando datos para entrenamiento XRP...")

        # Alinear features con labels
        features_aligned = features.iloc[:-self.prediction_horizon]

        # Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        # Normalizar features con RobustScaler (mejor para XRP volátil)
        scaler = RobustScaler()  # Más robusto a outliers que MinMaxScaler
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # Crear secuencias temporales de 48 timesteps
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

        print(f"✅ Datos preparados para XRP:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features utilizadas: {len(feature_columns)}")
        print(f"   - Secuencias temporales: {self.lookback_window} timesteps")

        # 🎯 CALCULAR CLASS WEIGHTS PARA BALANCEAR XRP
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"🎯 Class weights calculados para XRP:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, weight in class_weight_dict.items():
            print(f"   - {class_names[i]}: {weight:.3f}")

        return X, y, scaler, feature_columns, class_weight_dict

    def create_definitive_tcn_model_xrp(self, input_shape: tuple) -> tf.keras.Model:
        """🎯 Crear modelo TCN definitivo optimizado para XRP"""

        print("🎯 Creando modelo TCN definitivo para XRP...")

        model = tf.keras.Sequential([
            # Input
            tf.keras.layers.Input(shape=input_shape),

            # Normalización de entrada
            tf.keras.layers.LayerNormalization(),

            # TCN Layers optimizadas para XRP (más volátil)
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

            # Attention mechanism para XRP
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),

            # Dense layers optimizadas para XRP
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

        # Compilar con configuración optimizada para XRP
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0003),  # LR más bajo para XRP volátil
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo definitivo XRP creado: {model.count_params():,} parámetros")

        return model

    async def train_definitivo_model_xrp(self) -> bool:
        """🎯 Entrenar modelo definitivo para XRPUSDT"""

        print(f"\n🎯 ENTRENANDO MODELO DEFINITIVO PARA {self.symbol}")
        print("=" * 70)

        try:
            # 1. Obtener 30 días de datos reales (optimizado)
            df = await self.get_real_market_data(days=30)

            # 2. Crear 66 features
            features = self.create_essential_features(df)

            # 3. Crear etiquetas balanceadas específicas para XRP
            print("🎯 Creando etiquetas balanceadas para XRP...")
            try:
                df_labeled = self.create_balanced_labels_xrp(df, features)
                print(f"✅ Etiquetas XRP creadas correctamente")
            except Exception as e:
                print(f"❌ Error en create_balanced_labels_xrp: {e}")
                import traceback
                traceback.print_exc()
                return False

            # 4. Preparar datos con técnicas anti-sesgo
            print("🔧 Preparando datos de entrenamiento XRP...")
            try:
                X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)
                print(f"✅ Datos XRP preparados correctamente")
            except Exception as e:
                print(f"❌ Error en prepare_training_data: {e}")
                import traceback
                traceback.print_exc()
                return False

            # 5. Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear modelo definitivo para XRP
            model = self.create_definitive_tcn_model_xrp((X.shape[1], X.shape[2]))

            # 7. Callbacks avanzados con guardado frecuente
            model_dir = f'models/definitivo_{self.symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=10,  # Optimizado: menos paciencia
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=5,  # Optimizado: menos paciencia
                    factor=0.5,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{model_dir}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            # 8. Entrenar con class weights
            print("🚀 Entrenando modelo definitivo XRP...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,  # Optimizado: menos épocas
                batch_size=64,  # Optimizado: batch más grande
                callbacks=callbacks,
                class_weight=class_weights,  # 🎯 ANTI-SESGO
                verbose=1
            )

            # 9. Evaluar modelo
            try:
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                print(f"\n✅ RESULTADOS FINALES XRP:")
                print(f"   - Loss: {test_loss:.4f}")
                print(f"   - Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
            except Exception as e:
                print(f"⚠️ Error en evaluación, pero modelo entrenado: {e}")
                model.save(f'{model_dir}/final_model_backup.h5')
                test_acc = 0.0

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

            print(f"✅ Modelo definitivo XRP guardado en:")
            print(f"   📁 {model_dir}/")
            print(f"   📁 models/definitivo_xrpusdt.h5")

            return True

        except Exception as e:
            print(f"❌ Error entrenando modelo definitivo XRP: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """🎯 Entrenar modelo definitivo para XRPUSDT"""

    print("🎯 ENTRENADOR DE MODELO TCN DEFINITIVO PARA XRP")
    print("=" * 80)
    print("🎯 Objetivo: Lograr 60%+ accuracy con distribución balanceada")
    print("🔧 Técnicas: Class weights, 66 features, 48 timesteps, thresholds optimizados")
    print("📊 Datos: 30 días de datos reales de Binance (optimizado)")
    print("=" * 80)

    trainer = XRPDefinitiveTCNTrainer()
    success = await trainer.train_definitivo_model_xrp()

    if success:
        print(f"\n🎉 ¡MODELO XRP DEFINITIVO ENTRENADO EXITOSAMENTE!")
        print(f"📁 Modelo guardado: models/definitivo_xrpusdt.h5")
        print(f"🎯 Listo para integración al sistema de trading")
    else:
        print(f"\n❌ FALLO EN ENTRENAMIENTO DEL MODELO XRP")

if __name__ == "__main__":
    asyncio.run(main()) 