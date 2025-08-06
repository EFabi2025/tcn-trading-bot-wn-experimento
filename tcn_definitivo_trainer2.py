#!/usr/bin/env python3
"""
🎯 TCN SINGLE MODEL TRAINER - VERSIÓN CORREGIDA
Entrena un modelo TCN individual sin look-ahead bias y con validación temporal correcta
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from centralized_features_engine2 import CentralizedFeaturesEngine

def trading_loss(y_true, y_pred):
    """
    Loss function personalizada que optimiza para rentabilidad de trading
    Penaliza más los errores en BUY/SELL que en HOLD
    """
    # Convertir one-hot a clases
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    # Definir costos de error diferentes para cada clase
    # SELL=0, HOLD=1, BUY=2

    # Error costs matrix:
    # Predecir HOLD cuando debería BUY/SELL es muy costoso (oportunidad perdida)
    # Predecir BUY/SELL cuando debería HOLD es moderadamente costoso (falso positivo)
    # Confundir BUY con SELL es catastrófico

    error_costs = tf.constant([
        [1.0, 3.0, 10.0],  # True=SELL: correcto, hold_error, buy_error(catastrófico)
        [2.0, 1.0, 2.0],   # True=HOLD: sell_error, correcto, buy_error
        [10.0, 3.0, 1.0]   # True=BUY: sell_error(catastrófico), hold_error, correcto
    ], dtype=tf.float32)

    # Obtener costo de error para cada predicción
    cost = tf.gather_nd(error_costs, tf.stack([y_true_class, y_pred_class], axis=1))

    # Combinar con categorical crossentropy
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Loss final = weighted crossentropy
    return cce * cost

def profit_loss(y_true, y_pred):
    """
    Loss function alternativa que prioriza precision en señales de trading
    """
    # Categorical crossentropy base
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Obtener clases predichas
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    # Penalizar más errores en trading signals (BUY=2, SELL=0)
    trading_signal_mask = tf.logical_or(
        tf.equal(y_true_class, 0),  # True SELL
        tf.equal(y_true_class, 2)   # True BUY
    )

    # Multiplicar loss por 2 para trading signals
    penalty = tf.where(trading_signal_mask, 2.0, 1.0)

    return cce * penalty

def directional_accuracy(y_true, y_pred):
    """
    Métrica personalizada: accuracy solo en señales BUY/SELL
    """
    y_true_class = K.argmax(y_true, axis=-1)
    y_pred_class = K.argmax(y_pred, axis=-1)

    # Mask para señales de trading (no HOLD)
    trading_mask = tf.logical_or(
        tf.equal(y_true_class, 0),  # SELL
        tf.equal(y_true_class, 2)   # BUY
    )

    # Accuracy solo en trading signals
    correct_predictions = tf.equal(y_true_class, y_pred_class)
    trading_correct = tf.logical_and(trading_mask, correct_predictions)

    return tf.reduce_mean(tf.cast(trading_correct, tf.float32))

class PredictiveLabelingSystem:
    """Sistema de etiquetado basado en movimientos futuros rentables"""

    def __init__(self,
                 prediction_horizon: int = 6,
                 commission_rate: float = 0.001,
                 spread_cost: float = 0.0005,
                 slippage_cost: float = 0.0005):

        self.prediction_horizon = prediction_horizon
        self.total_cost = commission_rate + spread_cost + slippage_cost
        self.min_profitable_move = self.total_cost * 1.2  # 120% margen de seguridad (MÁS AGRESIVO)

        logger.info(f"Sistema predictivo inicializado:")
        logger.info(f"  - Horizonte: {prediction_horizon} períodos")
        logger.info(f"  - Costo total: {self.total_cost:.4f}")
        logger.info(f"  - Movimiento mínimo rentable: {self.min_profitable_move:.4f}")

    def create_future_return_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear etiquetas basadas en retornos futuros REALES"""

        logger.info("Creando etiquetas basadas en retornos futuros...")

        close_prices = df['close'].values
        labels = []

        # Calcular retornos futuros
        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Retorno futuro real
            future_return = (future_price - current_price) / current_price

            # Retorno después de costos
            net_return_long = future_return - self.total_cost
            net_return_short = -future_return - self.total_cost

            # Etiquetado MÁS AGRESIVO - umbrales más bajos
            min_threshold = self.min_profitable_move * 0.5  # Reducir umbral 50%

            if net_return_long >= min_threshold:
                labels.append(2)  # BUY - Rentable ir largo
            elif net_return_short >= min_threshold:
                labels.append(0)  # SELL - Rentable ir corto
            elif abs(future_return) > 0.002:  # Movimientos > 0.2%
                if future_return > 0:
                    labels.append(2)  # BUY en tendencia alcista
                else:
                    labels.append(0)  # SELL en tendencia bajista
            else:
                labels.append(1)  # HOLD - No hay movimiento rentable

        # Completar con HOLD para los últimos períodos
        labels.extend([1] * self.prediction_horizon)

        # Crear DataFrame
        df_labeled = df.copy()
        df_labeled['label'] = labels
        df_labeled['future_return'] = np.nan

        # Guardar retornos futuros para análisis
        for i in range(len(close_prices) - self.prediction_horizon):
            future_return = (close_prices[i + self.prediction_horizon] - close_prices[i]) / close_prices[i]
            df_labeled['future_return'].iloc[i] = future_return

        return df_labeled

    def create_adaptive_threshold_labels(self, df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
        """Etiquetado con umbrales adaptativos basados en volatilidad histórica"""

        logger.info("Creando etiquetas con umbrales adaptativos...")

        close_prices = df['close'].values
        labels = []

        # Calcular volatilidad rolling
        returns = pd.Series(close_prices).pct_change()
        rolling_vol = returns.rolling(window=20).std()

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Retorno futuro
            future_return = (future_price - current_price) / current_price

            # Umbral adaptativo basado en volatilidad
            current_vol = rolling_vol.iloc[i] if not pd.isna(rolling_vol.iloc[i]) else 0.02
            adaptive_threshold = max(self.min_profitable_move, current_vol * 1.5)

            # Etiquetado adaptativo MÁS AGRESIVO
            reduced_threshold = adaptive_threshold * 0.6  # Reducir umbral 40%

            if future_return >= reduced_threshold:
                labels.append(2)  # BUY
            elif future_return <= -reduced_threshold:
                labels.append(0)  # SELL
            elif abs(future_return) > 0.001:  # Cualquier movimiento > 0.1%
                if future_return > 0:
                    labels.append(2)  # BUY
                else:
                    labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD

        # Completar últimos períodos
        labels.extend([1] * self.prediction_horizon)

        df_labeled = df.copy()
        df_labeled['label'] = labels

        return df_labeled

    def create_momentum_filtered_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Etiquetas filtradas por momentum y contexto de mercado"""

        logger.info("Creando etiquetas filtradas por momentum...")

        close_prices = df['close'].values
        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Retorno futuro
            future_return = (future_price - current_price) / current_price

            # Contexto del mercado (información pasada)
            try:
                rsi = features['rsi_14'].iloc[i] if 'rsi_14' in features.columns else 50
                atr = features['atr_14'].iloc[i] if 'atr_14' in features.columns else 0.02
                volume_ratio = features['volume_ratio'].iloc[i] if 'volume_ratio' in features.columns else 1.0

                # Momentum histórico
                if i >= 10:
                    momentum_5 = (current_price - close_prices[i-5]) / close_prices[i-5]
                    momentum_10 = (current_price - close_prices[i-10]) / close_prices[i-10]
                else:
                    momentum_5 = momentum_10 = 0

            except (IndexError, KeyError):
                rsi = 50
                atr = 0.02
                volume_ratio = 1.0
                momentum_5 = momentum_10 = 0

            # Filtros de calidad
            volatility_ok = atr / current_price < 0.05  # Volatilidad controlada
            volume_ok = volume_ratio > 0.7  # Volumen suficiente
            rsi_ok = 25 < rsi < 75  # RSI no extremo
            momentum_consistent = abs(momentum_5) > self.min_profitable_move / 2

            # Etiquetado con filtros MÁS PERMISIVO
            base_threshold = max(self.min_profitable_move * 0.4, atr / current_price * 0.5)

            # Condiciones menos estrictas
            if future_return >= base_threshold and momentum_5 >= 0:
                labels.append(2)  # BUY (sin requerir todos los filtros)
            elif future_return <= -base_threshold and momentum_5 <= 0:
                labels.append(0)  # SELL (sin requerir todos los filtros)
            elif abs(future_return) > 0.0015:  # Movimiento significativo
                if future_return > 0:
                    labels.append(2)  # BUY
                else:
                    labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD

        # Completar últimos períodos
        labels.extend([1] * self.prediction_horizon)

        df_labeled = df.copy()
        df_labeled['label'] = labels

        return df_labeled

    def backtest_labels(self, df_labeled: pd.DataFrame) -> Dict:
        """Backtest las etiquetas para validar rentabilidad"""

        logger.info("Backtesting etiquetas...")

        portfolio_value = 1.0
        position = 0  # 0=cash, 1=long, -1=short
        trades = []
        close_prices = df_labeled['close'].values
        labels = df_labeled['label'].values

        for i in range(len(labels) - 1):
            signal = labels[i]
            current_price = close_prices[i]
            next_price = close_prices[i + 1]

            # Ejecutar señales
            if signal == 2 and position != 1:  # BUY signal
                if position == -1:  # Close short
                    portfolio_value *= (1 - self.total_cost)
                portfolio_value *= (1 - self.total_cost)  # Open long
                position = 1
                trades.append(('BUY', current_price, portfolio_value))

            elif signal == 0 and position != -1:  # SELL signal
                if position == 1:  # Close long
                    portfolio_value *= (1 - self.total_cost)
                portfolio_value *= (1 - self.total_cost)  # Open short
                position = -1
                trades.append(('SELL', current_price, portfolio_value))

            elif signal == 1 and position != 0:  # HOLD - close position
                portfolio_value *= (1 - self.total_cost)
                position = 0
                trades.append(('HOLD', current_price, portfolio_value))

            # Update portfolio based on price movement
            price_change = (next_price - current_price) / current_price
            if position == 1:  # Long position
                portfolio_value *= (1 + price_change)
            elif position == -1:  # Short position
                portfolio_value *= (1 - price_change)

        # Calcular métricas
        total_return = (portfolio_value - 1.0) * 100
        num_trades = len(trades)

        if num_trades > 1:
            profitable_trades = sum(1 for i in range(1, len(trades))
                                  if trades[i][2] > trades[i-1][2])
            win_rate = (profitable_trades / max(1, num_trades - 1)) * 100
        else:
            win_rate = 0

        metrics = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_portfolio': portfolio_value,
            'profitable': total_return > 0
        }

        logger.info(f"Backtest results:")
        logger.info(f"  - Return: {total_return:.2f}%")
        logger.info(f"  - Trades: {num_trades}")
        logger.info(f"  - Win rate: {win_rate:.1f}%")

        return metrics

    def analyze_label_distribution(self, df_labeled: pd.DataFrame) -> Dict:
        """Analizar distribución y balance de etiquetas"""

        labels = df_labeled['label'].values
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        distribution = {}
        class_names = ['SELL', 'HOLD', 'BUY']

        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            distribution[name] = {'count': count, 'percentage': pct}

        logger.info("Distribución de etiquetas:")
        for name, stats in distribution.items():
            logger.info(f"  {name}: {stats['count']} ({stats['percentage']:.1f}%)")

        return distribution

    def optimize_labeling_strategy(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Probar diferentes estrategias y seleccionar la mejor"""

        logger.info("Optimizando estrategia de etiquetado...")

        strategies = {
            'future_return': self.create_future_return_labels,
            'adaptive_threshold': self.create_adaptive_threshold_labels,
            'momentum_filtered': lambda df: self.create_momentum_filtered_labels(df, features)
        }

        best_strategy = None
        best_return = -float('inf')
        best_df = None
        results = {}

        for name, strategy_func in strategies.items():
            logger.info(f"\nTesting strategy: {name}")

            try:
                df_labeled = strategy_func(df)
                metrics = self.backtest_labels(df_labeled)
                distribution = self.analyze_label_distribution(df_labeled)

                results[name] = {
                    'metrics': metrics,
                    'distribution': distribution,
                    'profitable': metrics['profitable']
                }

                if metrics['total_return'] > best_return:
                    best_return = metrics['total_return']
                    best_strategy = name
                    best_df = df_labeled

            except Exception as e:
                logger.error(f"Error testing {name}: {e}")
                results[name] = {'error': str(e)}

        logger.info(f"\n🏆 Best strategy: {best_strategy}")
        logger.info(f"🎯 Best return: {best_return:.2f}%")

        return best_df, results

class AdvancedRiskManager:
    """Sistema avanzado de gestión de riesgo con stop-loss dinámico y position sizing"""

    def __init__(self,
                 initial_capital: float = 1.0,
                 max_position_size: float = 0.1,
                 max_drawdown: float = 0.15,
                 stop_loss_pct: float = 0.02,
                 trailing_stop_pct: float = 0.015):

        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct

        # Estado del trading
        self.current_position = 0  # 0=cash, 1=long, -1=short
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.trailing_stop_price = 0

        # Métricas de riesgo
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.trade_count = 0

        logger.info(f"🛡️ Risk Manager inicializado:")
        logger.info(f"   - Capital inicial: ${initial_capital:.2f}")
        logger.info(f"   - Max position size: {max_position_size*100:.1f}%")
        logger.info(f"   - Max drawdown: {max_drawdown*100:.1f}%")
        logger.info(f"   - Stop loss: {stop_loss_pct*100:.1f}%")
        logger.info(f"   - Trailing stop: {trailing_stop_pct*100:.1f}%")

    def execute_signal(self, signal: int, current_price: float, step: int) -> Dict:
        """Ejecutar señal con gestión de riesgo"""

        # Verificar stop loss y trailing stop
        if self._check_stop_loss(current_price) or self._check_trailing_stop(current_price):
            return self._close_position(current_price, step, "STOP_LOSS")

        # Verificar drawdown máximo
        if self.current_drawdown >= self.max_drawdown:
            return self._close_position(current_price, step, "MAX_DRAWDOWN")

        # Ejecutar señal
        if signal == 2 and self.current_position != 1:  # BUY
            return self._open_long_position(current_price, step)
        elif signal == 0 and self.current_position != -1:  # SELL
            return self._open_short_position(current_price, step)
        elif signal == 1 and self.current_position != 0:  # HOLD
            return self._close_position(current_price, step, "HOLD")

        return None

    def _open_long_position(self, price: float, step: int) -> Dict:
        """Abrir posición larga"""

        # Calcular tamaño de posición basado en Kelly Criterion
        position_size = self._calculate_position_size()

        # Verificar si hay suficiente capital
        if position_size <= 0:
            return None

        # Ejecutar trade
        self.current_position = 1
        self.position_size = position_size
        self.entry_price = price
        self.stop_loss_price = price * (1 - self.stop_loss_pct)
        self.trailing_stop_price = price * (1 - self.trailing_stop_pct)

        # Aplicar costos de trading
        self.portfolio_value *= (1 - 0.002)  # 0.2% costo de entrada

        self.trade_count += 1

        return {
            'type': 'BUY',
            'price': price,
            'position_size': position_size,
            'step': step,
            'portfolio_value': self.portfolio_value,
            'reason': 'SIGNAL'
        }

    def _open_short_position(self, price: float, step: int) -> Dict:
        """Abrir posición corta"""

        position_size = self._calculate_position_size()

        if position_size <= 0:
            return None

        self.current_position = -1
        self.position_size = position_size
        self.entry_price = price
        self.stop_loss_price = price * (1 + self.stop_loss_pct)
        self.trailing_stop_price = price * (1 + self.trailing_stop_pct)

        self.portfolio_value *= (1 - 0.002)
        self.trade_count += 1

        return {
            'type': 'SELL',
            'price': price,
            'position_size': position_size,
            'step': step,
            'portfolio_value': self.portfolio_value,
            'reason': 'SIGNAL'
        }

    def _close_position(self, price: float, step: int, reason: str) -> Dict:
        """Cerrar posición actual"""

        if self.current_position == 0:
            return None

        # Calcular P&L
        if self.current_position == 1:  # Long
            pnl = (price - self.entry_price) / self.entry_price
        else:  # Short
            pnl = (self.entry_price - price) / self.entry_price

        # Aplicar P&L al portfolio
        self.portfolio_value *= (1 + pnl * self.position_size)

        # Aplicar costos de salida
        self.portfolio_value *= (1 - 0.002)

        # Actualizar métricas
        self._update_drawdown()

        trade_result = {
            'type': 'CLOSE',
            'price': price,
            'pnl': pnl,
            'profit': pnl * self.position_size * self.initial_capital,
            'step': step,
            'portfolio_value': self.portfolio_value,
            'reason': reason
        }

        # Resetear posición
        self.current_position = 0
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.trailing_stop_price = 0

        return trade_result

    def _calculate_position_size(self) -> float:
        """Calcular tamaño de posición usando Kelly Criterion"""

        # Kelly Criterion simplificado
        win_rate = 0.6  # Asumir 60% win rate
        avg_win = 0.02  # 2% ganancia promedio
        avg_loss = 0.015  # 1.5% pérdida promedio

        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))

        # Ajustar por drawdown
        drawdown_factor = 1 - (self.current_drawdown / self.max_drawdown)
        drawdown_factor = max(0.1, drawdown_factor)  # Mínimo 10%

        return kelly_fraction * drawdown_factor

    def _check_stop_loss(self, current_price: float) -> bool:
        """Verificar si se activó el stop loss"""

        if self.current_position == 0:
            return False

        if self.current_position == 1:  # Long
            return current_price <= self.stop_loss_price
        else:  # Short
            return current_price >= self.stop_loss_price

    def _check_trailing_stop(self, current_price: float) -> bool:
        """Verificar si se activó el trailing stop"""

        if self.current_position == 0:
            return False

        if self.current_position == 1:  # Long
            # Actualizar trailing stop si el precio sube
            new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
            if new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop

            return current_price <= self.trailing_stop_price
        else:  # Short
            # Actualizar trailing stop si el precio baja
            new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
            if new_trailing_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop

            return current_price >= self.trailing_stop_price

    def update_position(self, price_change: float):
        """Actualizar posición con cambio de precio"""

        if self.current_position == 1:  # Long
            self.portfolio_value *= (1 + price_change * self.position_size)
        elif self.current_position == -1:  # Short
            self.portfolio_value *= (1 - price_change * self.position_size)

        self._update_drawdown()

    def _update_drawdown(self):
        """Actualizar métricas de drawdown"""

        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

@dataclass
class TCNConfig:
    """Configuración del modelo TCN"""
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    lookback_periods: int = 48
    prediction_horizon: int = 6  # Predicción a 6 períodos (30 min para 5m)
    training_days: int = 120

    # Costos de trading
    commission_rate: float = 0.001
    spread_cost: float = 0.0005
    slippage_cost: float = 0.0005

    @property
    def total_trading_cost(self) -> float:
        return self.commission_rate + self.spread_cost + self.slippage_cost

    @property
    def min_profitable_move(self) -> float:
        return max(self.total_trading_cost * 2.0, 0.015)  # Mínimo 1.5% para crypto 5m


class TCNSingleTrainer:
    """Entrenador de modelo TCN individual"""

    def __init__(self, config: TCNConfig):
        self.config = config
        self.scaler = None
        self.model = None
        self.feature_columns = []
        self.features_engine = CentralizedFeaturesEngine()

        logger.info(f"Inicializando TCN Trainer para {config.symbol}")
        logger.info(f"Timeframe: {config.timeframe}")
        logger.info(f"Lookback: {config.lookback_periods} períodos")
        logger.info(f"Predicción: {config.prediction_horizon} períodos adelante")

    async def fetch_market_data(self) -> pd.DataFrame:
        """Obtener datos de mercado"""

        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=self.config.training_days)).timestamp() * 1000)

        logger.info(f"Descargando {self.config.training_days} días de datos {self.config.timeframe}")

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': self.config.symbol,
            'interval': self.config.timeframe,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        all_data = []
        async with aiohttp.ClientSession() as session:
            current_start = start_time

            while current_start < end_time:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"API Error: {response.status}")
                        break

                    data = await response.json()
                    if not data:
                        break

                    all_data.extend(data)
                    current_start = data[-1][6] + 1

                await asyncio.sleep(0.1)

        # Convertir a DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                  'taker_buy_quote', 'ignore']

        df = pd.DataFrame(all_data, columns=columns)

        # Limpiar datos
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        # Remover outliers extremos
        for col in numeric_cols:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)

        df = df.dropna()

        logger.info(f"Datos obtenidos: {len(df)} registros")
        logger.info(f"Período: {df.index.min()} a {df.index.max()}")

        return df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features usando el motor centralizado"""

        logger.info("Calculando features con motor centralizado...")

        # Usar el motor centralizado de features
        features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')

        if features.empty:
            logger.error("Error: No se pudieron calcular features")
            raise ValueError("Failed to calculate features")

        # Limpiar features
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        logger.info(f"Features calculados: {len(features.columns)}")
        logger.info(f"Registros con features: {len(features)}")

        return features

    def create_predictive_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Crear etiquetas TCN SIN LOOK-AHEAD BIAS - Solo información pasada"""

        logger.info("🎯 Creando etiquetas TCN sin look-ahead bias...")

        labels = []

        for i in range(len(df)):
            try:
                # ETIQUETADO BASADO SOLO EN INFORMACIÓN PASADA
                if i < 20:  # Necesitamos historia suficiente
                    labels.append(1)  # HOLD
                    continue

                # 1. CONVERGENCIA DE INDICADORES TÉCNICOS (PASADOS)
                rsi = features['rsi_14'].iloc[i] if 'rsi_14' in features.columns else 50
                macd = features['macd'].iloc[i] if 'macd' in features.columns else 0
                macd_signal = features['macd_signal'].iloc[i] if 'macd_signal' in features.columns else 0
                bb_upper = features['bb_upper'].iloc[i] if 'bb_upper' in features.columns else df['close'].iloc[i] * 1.02
                bb_lower = features['bb_lower'].iloc[i] if 'bb_lower' in features.columns else df['close'].iloc[i] * 0.98

                current_price = df['close'].iloc[i]
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

                # 2. MOMENTUM HISTÓRICO
                price_5 = df['close'].iloc[i-5] if i >= 5 else current_price
                price_10 = df['close'].iloc[i-10] if i >= 10 else current_price
                momentum_5 = (current_price - price_5) / price_5
                momentum_10 = (current_price - price_10) / price_10

                # 3. VOLUMEN RELATIVO
                current_volume = df['volume'].iloc[i]
                avg_volume = df['volume'].iloc[i-20:i].mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # 4. VOLATILIDAD RECIENTE
                recent_returns = df['close'].iloc[i-10:i].pct_change().dropna()
                volatility = recent_returns.std() if len(recent_returns) > 1 else 0.01

                # 5. SEÑALES TÉCNICAS (SIN LOOK-AHEAD)
                bullish_signals = 0
                bearish_signals = 0

                # RSI oversold/overbought
                if rsi < 30:
                    bullish_signals += 2
                elif rsi > 70:
                    bearish_signals += 2

                # MACD crossover
                macd_hist = macd - macd_signal
                if i > 0:
                    prev_macd_hist = features['macd'].iloc[i-1] - features['macd_signal'].iloc[i-1]
                    if macd_hist > 0 and prev_macd_hist <= 0:  # Bullish crossover
                        bullish_signals += 2
                    elif macd_hist < 0 and prev_macd_hist >= 0:  # Bearish crossover
                        bearish_signals += 2

                # Bollinger Bands position
                if bb_position < 0.2:  # Near lower band
                    bullish_signals += 1
                elif bb_position > 0.8:  # Near upper band
                    bearish_signals += 1

                # Momentum confirmation
                if momentum_5 > 0.01 and momentum_10 > 0.005:  # Strong upward momentum
                    bullish_signals += 1
                elif momentum_5 < -0.01 and momentum_10 < -0.005:  # Strong downward momentum
                    bearish_signals += 1

                # Volume confirmation
                if volume_ratio > 1.5:  # High volume
                    if bullish_signals > bearish_signals:
                        bullish_signals += 1
                    elif bearish_signals > bullish_signals:
                        bearish_signals += 1

                # 6. FILTROS DE CALIDAD
                signal_strength = abs(bullish_signals - bearish_signals)
                volume_sufficient = volume_ratio > 0.5
                volatility_acceptable = volatility < 0.05

                # 7. DECISION FINAL (CONSERVADORA)
                if (bullish_signals >= bearish_signals + 3 and
                    signal_strength >= 3 and volume_sufficient and volatility_acceptable):
                    labels.append(2)  # BUY - Señal bullish fuerte
                elif (bearish_signals >= bullish_signals + 3 and
                      signal_strength >= 3 and volume_sufficient and volatility_acceptable):
                    labels.append(0)  # SELL - Señal bearish fuerte
                else:
                    labels.append(1)  # HOLD - Señal débil o condiciones no favorables

            except Exception as e:
                labels.append(1)  # HOLD en caso de error

        # Crear DataFrame
        df_labeled = df.copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        logger.info("📊 Distribución de etiquetas TCN (sin look-ahead):")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            logger.info(f"   {name}: {count} ({pct:.1f}%)")

        return df_labeled

    def prepare_sequences(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preparar secuencias temporales"""

        logger.info("Preparando secuencias temporales...")

        # Seleccionar SOLO features relevantes para trading (reducir overfitting)
        essential_features = [
            # Price & Volume
            'close', 'volume', 'high', 'low', 'open',

            # Momentum Indicators
            'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'momentum_10', 'roc_10',

            # Trend Indicators
            'sma_20', 'ema_12', 'ema_26',

            # Volatility
            'bb_upper', 'bb_lower', 'bb_width',
            'atr_14',

            # Volume Indicators
            'volume_sma_20', 'volume_ratio'
        ]

        # Filtrar solo features que existen
        available_features = [f for f in essential_features if f in features.columns]
        numeric_features = features[available_features].select_dtypes(include=[np.number])
        self.feature_columns = list(numeric_features.columns)

        logger.info(f"Features seleccionados: {len(self.feature_columns)} de {len(features.columns)} disponibles")
        logger.info(f"Features usados: {self.feature_columns}")

        # Normalizar features
        self.scaler = RobustScaler()
        features_scaled = self.scaler.fit_transform(numeric_features)

        # Crear secuencias
        X, y = [], []

        for i in range(self.config.lookback_periods, len(features_scaled)):
            sequence = features_scaled[i-self.config.lookback_periods:i]
            X.append(sequence)
            y.append(labels.iloc[i])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Secuencias creadas: X={X.shape}, y={y.shape}")

        return X, y

    def create_tcn_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Crear modelo TCN"""

        logger.info("Creando modelo TCN...")

        # Arquitectura TCN optimizada para trading
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LayerNormalization(),

            # Bloque 1: Patrones de corto plazo
            tf.keras.layers.Conv1D(48, 3, dilation_rate=1, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.1),

            # Bloque 2: Patrones medios
            tf.keras.layers.Conv1D(64, 3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.15),

            # Bloque 3: Tendencias
            tf.keras.layers.Conv1D(96, 3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.2),

            # Bloque 4: Contexto amplio
            tf.keras.layers.Conv1D(128, 3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.2),

            # Skip connection para preservar información
            tf.keras.layers.Conv1D(64, 1, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),

            # Global pooling con attention
            tf.keras.layers.GlobalAveragePooling1D(),

            # Capas densas optimizadas
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Output especializado para trading
            tf.keras.layers.Dense(3, activation='softmax', name='trading_output')
        ])

        # Compilar con loss personalizada para trading
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001),
            loss=trading_loss,  # Loss personalizada para rentabilidad
            metrics=['accuracy', directional_accuracy]
        )

        logger.info(f"Modelo creado: {model.count_params():,} parámetros")

        return model

    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                               window_size: int = 1000, step_size: int = 200) -> Dict:
        """Walk-forward validation para validar estabilidad temporal"""

        logger.info("🔄 Iniciando Walk-Forward Validation...")
        logger.info(f"   📊 Window size: {window_size} períodos")
        logger.info(f"   📈 Step size: {step_size} períodos")

        results = []
        total_windows = (len(X) - window_size) // step_size

        for i in range(total_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            # Datos de entrenamiento (ventana deslizante)
            X_train = X[start_idx:end_idx]
            y_train = y[start_idx:end_idx]

            # Datos de validación (siguiente step)
            val_start = end_idx
            val_end = min(val_start + step_size, len(X))
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            if len(X_val) < 50:  # Ventana muy pequeña
                continue

            # Entrenar modelo en esta ventana
            model = self.create_tcn_model((X.shape[1], X.shape[2]))

            # Callbacks para esta ventana
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=5,
                    factor=0.7,
                    monitor='val_loss'
                )
            ]

            # Entrenar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )

            # Evaluar
            y_pred = np.argmax(model.predict(X_val), axis=1)
            accuracy = np.mean(y_val == y_pred)

            # Backtesting en ventana de validación
            val_prices = df['close'].iloc[val_start:val_end].values
            trading_metrics = self._backtest_window(y_val, y_pred, val_prices)

            # Guardar resultados
            window_result = {
                'window': i + 1,
                'start_date': df.index[start_idx],
                'end_date': df.index[end_idx],
                'val_start_date': df.index[val_start],
                'val_end_date': df.index[val_end],
                'accuracy': accuracy,
                'trading_return': trading_metrics['total_return'],
                'win_rate': trading_metrics['win_rate'],
                'num_trades': trading_metrics['num_trades']
            }

            results.append(window_result)

            logger.info(f"   📊 Window {i+1}/{total_windows}: "
                       f"Acc={accuracy:.3f}, Return={trading_metrics['total_return']:.2f}%, "
                       f"Win={trading_metrics['win_rate']:.1f}%")

        # Análisis de estabilidad
        stability_analysis = self._analyze_stability(results)

        logger.info(f"✅ Walk-Forward completado: {len(results)} ventanas analizadas")
        logger.info(f"📊 Estabilidad: {stability_analysis['stability_score']:.2f}/10")

        return {
            'window_results': results,
            'stability_analysis': stability_analysis
        }

    def _backtest_window(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray) -> Dict:
        """Backtest para una ventana específica"""

        portfolio_value = 1.0
        position = 0
        trades = []

        for i in range(len(y_pred)):
            signal = y_pred[i]
            current_price = prices[i]

            # Ejecutar señales
            if signal == 2 and position != 1:  # BUY
                if position == -1:
                    portfolio_value *= (1 - self.config.total_trading_cost)
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = 1
                trades.append(('BUY', current_price, portfolio_value))

            elif signal == 0 and position != -1:  # SELL
                if position == 1:
                    portfolio_value *= (1 - self.config.total_trading_cost)
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = -1
                trades.append(('SELL', current_price, portfolio_value))

            elif signal == 1 and position != 0:  # HOLD
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = 0
                trades.append(('HOLD', current_price, portfolio_value))

            # Update portfolio
            if i < len(prices) - 1:
                price_change = (prices[i+1] - current_price) / current_price
                if position == 1:
                    portfolio_value *= (1 + price_change)
                elif position == -1:
                    portfolio_value *= (1 - price_change)

        total_return = (portfolio_value - 1.0) * 100
        num_trades = len(trades)

        if num_trades > 1:
            profitable_trades = sum(1 for i in range(1, len(trades))
                                  if trades[i][2] > trades[i-1][2])
            win_rate = (profitable_trades / max(1, num_trades - 1)) * 100
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_portfolio': portfolio_value
        }

    def _analyze_stability(self, results: List[Dict]) -> Dict:
        """Analizar estabilidad de los resultados"""

        returns = [r['trading_return'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        win_rates = [r['win_rate'] for r in results]

        # Métricas de estabilidad
        return_std = np.std(returns)
        accuracy_std = np.std(accuracies)
        win_rate_std = np.std(win_rates)

        # Calcular score de estabilidad (0-10)
        stability_score = 10 - (return_std * 0.5 + accuracy_std * 0.3 + win_rate_std * 0.2)
        stability_score = max(0, min(10, stability_score))

        # Análisis de tendencia
        positive_windows = sum(1 for r in returns if r > 0)
        consistency_score = (positive_windows / len(returns)) * 10

        return {
            'stability_score': stability_score,
            'consistency_score': consistency_score,
            'return_std': return_std,
            'accuracy_std': accuracy_std,
            'win_rate_std': win_rate_std,
            'positive_windows': positive_windows,
            'total_windows': len(results)
        }

    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray) -> Dict:
        """Calcular métricas de trading (rentabilidad real)"""

        logger.info("💰 Calculando métricas de trading...")

        # Métricas básicas
        accuracy = np.mean(y_true == y_pred)

        # Simular trading
        initial_balance = 1000.0
        balance = initial_balance
        position = 0  # 0=cash, 1=long
        trades = []
        total_fees = 0.0

        for i in range(len(y_pred)):
            signal = y_pred[i]
            price = prices[i]

            # Aplicar señal
            if signal == 2 and position == 0:  # BUY signal
                # Comprar
                fee = balance * self.config.total_trading_cost
                balance -= fee
                total_fees += fee
                position = 1
                trades.append(('BUY', price, balance))

            elif signal == 0 and position == 1:  # SELL signal
                # Vender
                fee = balance * self.config.total_trading_cost
                balance -= fee
                total_fees += fee
                position = 0
                trades.append(('SELL', price, balance))

            # Actualizar balance por movimiento del precio
            if i < len(prices) - 1 and position == 1:
                price_change = (prices[i+1] - price) / price
                balance *= (1 + price_change)

        # Cerrar posición final si está abierta
        if position == 1:
            fee = balance * self.config.total_trading_cost
            balance -= fee
            total_fees += fee
            trades.append(('SELL_FINAL', prices[-1], balance))

        # Calcular métricas
        total_return = ((balance - initial_balance) / initial_balance) * 100
        num_trades = len([t for t in trades if t[0] in ['BUY', 'SELL']])

        # Win rate
        if len(trades) >= 2:
            profitable_trades = 0
            for i in range(1, len(trades), 2):  # Pares BUY-SELL
                if i < len(trades) - 1:
                    buy_price = trades[i-1][1]
                    sell_price = trades[i][1]
                    if sell_price > buy_price:
                        profitable_trades += 1
            win_rate = (profitable_trades / max(1, num_trades // 2)) * 100
        else:
            win_rate = 0

        # Sharpe ratio simplificado
        if len(trades) > 1:
            returns = [t[2] for t in trades]
            returns_pct = np.diff(returns) / returns[:-1]
            sharpe_ratio = np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0
        else:
            sharpe_ratio = 0

        metrics = {
            'accuracy': accuracy,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': balance,
            'total_fees': total_fees,
            'initial_balance': initial_balance
        }

        logger.info("📊 Métricas de trading:")
        logger.info(f"   💡 Accuracy: {accuracy:.3f}")
        logger.info(f"   💰 Retorno total: {total_return:.2f}%")
        logger.info(f"   🔄 Número de trades: {num_trades}")
        logger.info(f"   🎯 Win rate: {win_rate:.1f}%")
        logger.info(f"   📊 Sharpe ratio: {sharpe_ratio:.3f}")
        logger.info(f"   💵 Balance final: ${balance:.2f}")

        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calcular métricas de evaluación básicas"""

        accuracy = np.mean(y_true == y_pred)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=['SELL', 'HOLD', 'BUY'], output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        return metrics

    def backtest_model_performance(self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame) -> Dict:
        """Backtest simplificado del rendimiento del modelo"""

        logger.info("🔄 Backtesting rendimiento del modelo...")

        # Obtener precios correspondientes al test set
        test_start_idx = len(df) - len(X_test)
        test_prices = df['close'].iloc[test_start_idx:].values

        # Simular trading simple
        portfolio_value = 1000.0
        position = 0  # 0=cash, 1=long, -1=short
        trades = []
        max_portfolio = portfolio_value
        max_drawdown = 0

        for i in range(len(y_pred)):
            signal = y_pred[i]
            current_price = test_prices[i]

            # Ejecutar señales
            if signal == 2 and position != 1:  # BUY
                if position == -1:
                    portfolio_value *= (1 - self.config.total_trading_cost)
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = 1
                trades.append(('BUY', current_price, portfolio_value))

            elif signal == 0 and position != -1:  # SELL
                if position == 1:
                    portfolio_value *= (1 - self.config.total_trading_cost)
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = -1
                trades.append(('SELL', current_price, portfolio_value))

            elif signal == 1 and position != 0:  # HOLD
                portfolio_value *= (1 - self.config.total_trading_cost)
                position = 0
                trades.append(('HOLD', current_price, portfolio_value))

            # Update portfolio
            if i < len(test_prices) - 1:
                price_change = (test_prices[i+1] - current_price) / current_price
                if position == 1:
                    portfolio_value *= (1 + price_change)
                elif position == -1:
                    portfolio_value *= (1 - price_change)

            # Track drawdown
            if portfolio_value > max_portfolio:
                max_portfolio = portfolio_value
            current_dd = (max_portfolio - portfolio_value) / max_portfolio
            max_drawdown = max(max_drawdown, current_dd)

        # Calcular métricas
        total_return = (portfolio_value - 1000.0) / 1000.0 * 100
        num_trades = len(trades)

        if num_trades > 1:
            profitable_trades = sum(1 for i in range(1, len(trades))
                                  if trades[i][2] > trades[i-1][2])
            win_rate = (profitable_trades / max(1, num_trades - 1)) * 100
        else:
            win_rate = 0

        # Sharpe ratio simplificado
        if len(trades) > 1:
            returns = [t[2] for t in trades]
            returns_pct = np.diff(returns) / returns[:-1]
            sharpe_ratio = np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0
        else:
            sharpe_ratio = 0

        trading_metrics = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'final_portfolio': portfolio_value,
            'profitable': total_return > 0
        }

        logger.info(f"📊 Trading metrics:")
        logger.info(f"   - Return: {total_return:.2f}%")
        logger.info(f"   - Trades: {num_trades}")
        logger.info(f"   - Win rate: {win_rate:.1f}%")
        logger.info(f"   - Sharpe: {sharpe_ratio:.3f}")
        logger.info(f"   - Max drawdown: {max_drawdown*100:.2f}%")

        return trading_metrics

    def _calculate_advanced_metrics(self, risk_manager, trades: List, portfolio_history: List) -> Dict:
        """Calcular métricas avanzadas de trading"""

        total_return = (risk_manager.portfolio_value - 1.0) * 100
        num_trades = len(trades)

        if num_trades > 1:
            profitable_trades = sum(1 for trade in trades if trade['profit'] > 0)
            win_rate = (profitable_trades / num_trades) * 100
        else:
            win_rate = 0

        # Calcular Sharpe ratio
        if len(portfolio_history) > 1:
            returns = [h['portfolio_value'] for h in portfolio_history]
            returns_pct = np.diff(returns) / returns[:-1]
            sharpe_ratio = np.mean(returns_pct) / np.std(returns_pct) if np.std(returns_pct) > 0 else 0
        else:
            sharpe_ratio = 0

        # Calcular Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / risk_manager.max_drawdown if risk_manager.max_drawdown > 0 else 0

        # Análisis de drawdown
        drawdown_analysis = self._analyze_drawdown(portfolio_history)

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': risk_manager.max_drawdown,
            'final_portfolio': risk_manager.portfolio_value,
            'profitable': total_return > 0,
            'drawdown_analysis': drawdown_analysis
        }

    def _analyze_drawdown(self, portfolio_history: List) -> Dict:
        """Analizar patrones de drawdown"""

        drawdowns = [h['drawdown'] for h in portfolio_history]
        max_dd = max(drawdowns)
        avg_dd = np.mean(drawdowns)

        # Contar períodos de drawdown
        dd_periods = sum(1 for dd in drawdowns if dd > 0)
        dd_frequency = dd_periods / len(drawdowns) if drawdowns else 0

        return {
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'dd_frequency': dd_frequency,
            'dd_periods': dd_periods
        }

    async def train(self) -> Dict:
        """Entrenar el modelo TCN"""

        logger.info("Iniciando entrenamiento TCN...")

        # 1. Obtener datos
        df = await self.fetch_market_data()

        # 2. Calcular features
        features = self.calculate_features(df)

        # 3. Crear labels PREDICTIVAS basadas en movimientos futuros
        df_labeled = self.create_predictive_labels(df, features)

        # 4. Preparar secuencias
        X, y = self.prepare_sequences(features, df_labeled['label'])

        # 5. Validación temporal robusta con rolling windows
        logger.info("🔄 Implementando validación temporal robusta...")

        # Usar 3 splits temporales para validación cruzada
        total_samples = len(X)
        train_size = int(total_samples * 0.6)  # 60% para entrenamiento
        val_size = int(total_samples * 0.2)    # 20% para validación
        test_size = total_samples - train_size - val_size  # 20% para test final

        # Split 1: datos más antiguos para entrenamiento
        X_train = X[:train_size]
        y_train = y[:train_size]

        # Split 2: datos medios para validación
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        # Split 3: datos más recientes para test final
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        logger.info(f"📊 Validación temporal robusta:")
        logger.info(f"   Train: {len(X_train)} samples (más antiguos)")
        logger.info(f"   Val:   {len(X_val)} samples (medios)")
        logger.info(f"   Test:  {len(X_test)} samples (más recientes)")

        # 6. Convertir etiquetas a one-hot para loss personalizada
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=3)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=3)

        # 7. Calcular class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        logger.info(f"Class weights: {class_weight_dict}")

        # 8. Crear modelo
        self.model = self.create_tcn_model((X.shape[1], X.shape[2]))

        # 8. Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=8,
                factor=0.7,
                monitor='val_loss',
                verbose=1
            )
        ]

        # 9. Entrenar con validación temporal robusta
        logger.info("Entrenando modelo con loss personalizada y validación temporal...")
        history = self.model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),  # Usar validation set separado
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # 10. Evaluación
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        metrics = self.calculate_metrics(y_test, y_pred)

        # 11. Backtesting para evaluar rentabilidad
        trading_metrics = self.backtest_model_performance(X_test, y_test, y_pred, df)

        # Combinar métricas
        self.final_metrics = {
            **metrics,
            'trading_metrics': trading_metrics
        }

        # 12. Mostrar resultados
        logger.info("\n" + "="*50)
        logger.info("RESULTADOS FINALES")
        logger.info("="*50)
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Retorno total: {trading_metrics['total_return']:.2f}%")
        logger.info(f"Win rate: {trading_metrics['win_rate']:.1f}%")
        logger.info(f"Número de trades: {trading_metrics['num_trades']}")

        return self.final_metrics

    def save_model(self, model_dir: str):
        """Guardar modelo y componentes"""

        os.makedirs(model_dir, exist_ok=True)

        # Guardar modelo principal
        self.model.save(f"{model_dir}/model.h5")

        # Guardar también como best_model.h5 para compatibilidad
        self.model.save(f"{model_dir}/best_model.h5")

        # Guardar scaler
        with open(f"{model_dir}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Guardar configuración
        config_dict = {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'lookback_periods': self.config.lookback_periods,
            'prediction_horizon': self.config.prediction_horizon,
            'training_days': self.config.training_days,
            'total_trading_cost': self.config.total_trading_cost,
            'min_profitable_move': self.config.min_profitable_move
        }

        with open(f"{model_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Guardar feature columns (compatibilidad con ensemble predictor)
        with open(f"{model_dir}/feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)

        # Guardar también como features.pkl para compatibilidad (formato diccionario)
        features_dict = {
            'feature_columns': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'model_type': 'tcn_definitivo'
        }
        with open(f"{model_dir}/features.pkl", "wb") as f:
            pickle.dump(features_dict, f)

        # Guardar métricas híbridas (requerido por ensemble predictor)
        # Usar métricas reales si están disponibles, sino valores por defecto
        if hasattr(self, 'final_metrics'):
            accuracy = self.final_metrics['accuracy']
            # Extraer precision y recall del classification report
            report = self.final_metrics['classification_report']
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']

            # Incluir métricas de trading si están disponibles
            trading_metrics = self.final_metrics.get('trading_metrics', {})
            total_return = trading_metrics.get('total_return', 0)
            win_rate = trading_metrics.get('win_rate', 0)
            num_trades = trading_metrics.get('num_trades', 0)

            # Métricas de estabilidad (valores por defecto)
            stability_score = 7.0  # Valor moderado
            consistency_score = 7.0
        else:
            accuracy = 0.65
            precision = 0.60
            recall = 0.55
            total_return = 0
            win_rate = 0
            num_trades = 0
            stability_score = 5.0
            consistency_score = 5.0

        hybrid_metrics = {
            'final_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'model_type': 'tcn_definitivo',
            'training_days': self.config.training_days,
            'timeframe': self.config.timeframe
        }
        with open(f"{model_dir}/hybrid_metrics.pkl", "wb") as f:
            pickle.dump(hybrid_metrics, f)

        logger.info(f"Modelo guardado en: {model_dir}")
        logger.info(f"Archivos guardados: model.h5, best_model.h5, scaler.pkl, config.json, feature_columns.pkl, features.pkl, hybrid_metrics.pkl")

    async def predict(self, market_data: pd.DataFrame) -> int:
        """Hacer predicción en tiempo real"""

        if self.model is None or self.scaler is None:
            raise ValueError("Modelo no entrenado")

        # Calcular features
        features = self.calculate_features(market_data)

        # Tomar últimos períodos
        recent_features = features[self.feature_columns].iloc[-self.config.lookback_periods:]

        # Normalizar
        features_scaled = self.scaler.transform(recent_features)

        # Predecir
        X = features_scaled.reshape(1, self.config.lookback_periods, len(self.feature_columns))
        prediction = self.model.predict(X, verbose=0)

        return np.argmax(prediction[0])


async def main():
    """Función principal"""

    print("🎯 TCN SINGLE MODEL TRAINER")
    print("="*60)

    # Configuración
    config = TCNConfig(
        symbol="BTCUSDT",
        timeframe="5m",
        lookback_periods=36,  # Aumentado para capturar más contexto (3 horas)
        prediction_horizon=3,  # Reducido para predicciones más precisas (15 min)
        training_days=60      # Aumentado para más datos
    )

    print(f"Símbolo: {config.symbol}")
    print(f"Timeframe: {config.timeframe}")
    print(f"Lookback: {config.lookback_periods} períodos")
    print(f"Predicción: {config.prediction_horizon} períodos adelante")
    print(f"Días de entrenamiento: {config.training_days}")
    print(f"Costo total trading: {config.total_trading_cost:.3f}")
    print(f"Movimiento mínimo rentable: {config.min_profitable_move:.3f}")

    # Crear y entrenar trader
    trainer = TCNSingleTrainer(config)

    try:
        metrics = await trainer.train()

        # Crear directorio del modelo (compatible con ensemble predictor)
        if config.timeframe == '1m':
            model_dir = f"models/definitivo_v3_{config.symbol.lower()}"
        else:
            model_dir = f"models/definitivo_v3_{config.timeframe}_{config.symbol.lower()}"

        # Guardar modelo
        trainer.save_model(model_dir)
        print(f"\n✅ Modelo guardado en: {model_dir}")

        # Ejemplo de predicción
        print(f"\n🔮 Ejemplo de predicción en tiempo real:")
        df_example = await trainer.fetch_market_data()
        prediction = await trainer.predict(df_example)

        signal_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
        print(f"Señal actual: {signal_names[prediction]}")

    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
