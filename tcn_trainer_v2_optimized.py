#!/usr/bin/env python3
"""
🎯 TCN TRAINER V2 OPTIMIZADO - ETIQUETADO BALANCEADO
Versión mejorada con etiquetado balanceado dinámico usando percentiles para distribución óptima
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
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

from centralized_features_engine_optimized import CentralizedFeaturesEngineOptimized as CentralizedFeaturesEngine


class OptimizedTCNTrainer:
    """🎯 Entrenador TCN V2 OPTIMIZADO con etiquetado balanceado"""

    def __init__(self, config=None):
        # ✅ CONFIGURACIÓN FLEXIBLE
        self.config = config or {}
        
        # Configuración profesional por defecto
        self.pairs = self.config.get('pairs', ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOTUSDT"])
        self.timeframe = self.config.get('timeframe', '5m')  # Timeframe configurable
        self.lookback_window = self.config.get('lookback_window', 48)  # 4 horas de historia
        self.prediction_horizon = self.config.get('prediction_horizon', 12)  # 1 hora de predicción
        self.days = self.config.get('days', 45)  # Más días para entrenamiento profesional
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ CONFIGURACIÓN DE ETIQUETADO BALANCEADO
        self.aggressiveness = self.config.get('aggressiveness', 'balanced')  # 'conservative', 'balanced', 'aggressive'
        self.use_adaptive_thresholds = self.config.get('use_adaptive_thresholds', True)
        self.force_signals = self.config.get('force_signals', False)  # No forzar señales en sistema balanceado

        # 🎯 CONFIGURACIONES PROFESIONALES ESPECÍFICAS POR PAR - MÁS PERMISIVAS
        self.professional_configs = {
            'BTCUSDT': {
                'timeframe': '5m',
                'lookback_window': 48,  # 4 horas
                'prediction_horizon': 12,  # 1 hora
                'days': 60,  # 2 meses
                'min_profit_threshold': 0.003,  # 0.3% mínimo (más permisivo)
                'percentile_sell': 35,  # Más permisivo
                'percentile_buy': 65,   # Más permisivo
                'rsi_sell_threshold': 65,
                'rsi_buy_threshold': 35,
                'momentum_threshold': 0.005  # 0.5% momentum (más permisivo)
            },
            'ETHUSDT': {
                'timeframe': '5m',
                'lookback_window': 60,  # 5 horas (más volátil)
                'prediction_horizon': 15,  # 1.25 horas
                'days': 75,  # 2.5 meses
                'min_profit_threshold': 0.004,  # 0.4% mínimo (más permisivo)
                'percentile_sell': 30,  # Más permisivo
                'percentile_buy': 70,   # Más permisivo
                'rsi_sell_threshold': 70,
                'rsi_buy_threshold': 30,
                'momentum_threshold': 0.006  # 0.6% momentum (más permisivo)
            },
            'BNBUSDT': {
                'timeframe': '5m',
                'lookback_window': 36,  # 3 horas
                'prediction_horizon': 10,  # 50 minutos
                'days': 45,  # 1.5 meses
                'min_profit_threshold': 0.003,  # 0.3% mínimo (más permisivo)
                'percentile_sell': 35,  # Más permisivo
                'percentile_buy': 65,   # Más permisivo
                'rsi_sell_threshold': 65,
                'rsi_buy_threshold': 35,
                'momentum_threshold': 0.005  # 0.5% momentum (más permisivo)
            },
            'XRPUSDT': {
                'timeframe': '5m',
                'lookback_window': 72,  # 6 horas (muy volátil)
                'prediction_horizon': 18,  # 1.5 horas
                'days': 90,  # 3 meses
                'min_profit_threshold': 0.005,  # 0.5% mínimo (más permisivo)
                'percentile_sell': 25,  # Más permisivo
                'percentile_buy': 75,   # Más permisivo
                'rsi_sell_threshold': 75,
                'rsi_buy_threshold': 25,
                'momentum_threshold': 0.008  # 0.8% momentum (más permisivo)
            },
            'DOTUSDT': {
                'timeframe': '5m',
                'lookback_window': 54,  # 4.5 horas
                'prediction_horizon': 14,  # 1.17 horas
                'days': 60,  # 2 meses
                'min_profit_threshold': 0.004,  # 0.4% mínimo (más permisivo)
                'percentile_sell': 30,  # Más permisivo
                'percentile_buy': 70,   # Más permisivo
                'rsi_sell_threshold': 70,
                'rsi_buy_threshold': 30,
                'momentum_threshold': 0.006  # 0.6% momentum (más permisivo)
            }
        }

        # 🎯 CONFIGURACIONES FLEXIBLES ADICIONALES
        self.flexible_configs = {
            'timeframes': ['1m', '3m', '5m'],
            'days_options': [24, 32, 48, 60, 75, 90],
            'prediction_horizons': [6, 8, 12, 15, 18],
            'lookback_windows': {
                '1m': [60, 120, 180, 240, 300],  # 1-5 horas
                '3m': [20, 40, 60, 80, 100],     # 1-5 horas
                '5m': [12, 24, 36, 48, 60, 72]   # 1-6 horas
            }
        }

        # 🎯 THRESHOLDS BALANCEADOS - PARA DISTRIBUCIÓN ÓPTIMA
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.006, 'weak_sell': -0.003,
                'weak_buy': 0.003, 'strong_buy': 0.006
            },
            'ETHUSDT': {
                'strong_sell': -0.008, 'weak_sell': -0.004,
                'weak_buy': 0.004, 'strong_buy': 0.008
            },
            'BNBUSDT': {
                'strong_sell': -0.005, 'weak_sell': -0.0025,
                'weak_buy': 0.0025, 'strong_buy': 0.005
            },
            'XRPUSDT': {
                'strong_sell': -0.010, 'weak_sell': -0.005,
                'weak_buy': 0.005, 'strong_buy': 0.010
            },
            'DOTUSDT': {
                'strong_sell': -0.012, 'weak_sell': -0.006,
                'weak_buy': 0.006, 'strong_buy': 0.012
            }
        }

        # ✅ FACTORES DE AGRESIVIDAD
        self.aggressiveness_factors = {
            'conservative': {'factor': 0.7, 'momentum_threshold': 0.008, 'rsi_buffer': 10},
            'balanced': {'factor': 1.0, 'momentum_threshold': 0.005, 'rsi_buffer': 5},
            'aggressive': {'factor': 1.3, 'momentum_threshold': 0.003, 'rsi_buffer': 2}
        }

        print(f"🎯 CONFIGURACIÓN BALANCEADA:")
        print(f"   ⏰ Timeframe: {self.timeframe}")
        print(f"   🎯 Agresividad: {self.aggressiveness}")
        print(f"   📊 Lookback: {self.lookback_window}")
        print(f"   ⏰ Horizonte: {self.prediction_horizon}")
        print(f"   📅 Días: {self.days}")
        print(f"   🔥 Forzar señales: {self.force_signals} (deshabilitado en sistema balanceado)")

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """🎯 Thresholds adaptativos BALANCEADOS usando motor centralizado"""
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]

        try:
            # ✅ USAR MOTOR CENTRALIZADO EN LUGAR DE CÁLCULO MANUAL
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')
            
            if features.empty or 'atr_14' not in features.columns:
                print(f"⚠️ No se pudo obtener ATR del motor centralizado, usando configuración fija")
                return self.fixed_thresholds[symbol]

            # ✅ USAR ATR DEL MOTOR CENTRALIZADO
            atr_14 = features['atr_14'].values
            close_prices = df['close'].values

            # Promedio de ATR reciente
            avg_atr = np.nanmean(atr_14[-50:]) if len(atr_14) > 50 else np.nanmean(atr_14)
            avg_price = np.mean(close_prices[-50:]) if len(close_prices) > 50 else np.mean(close_prices)

            # ATR como porcentaje del precio
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02

            # ✅ FACTOR RESPONSIVO según agresividad
            factor = self.aggressiveness_factors[self.aggressiveness]['factor']
            base_threshold = atr_percent * factor

            adaptive_thresholds = {
                'strong_sell': -base_threshold * 1.0,  # Reducido de 1.2
                'weak_sell': -base_threshold * 0.5,    # Reducido de 0.6
                'weak_buy': base_threshold * 0.5,      # Reducido de 0.6
                'strong_buy': base_threshold * 1.0     # Reducido de 1.2
            }

            print(f"🎯 {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%)")
            print(f"   📊 Factor agresividad: {factor}")
            print(f"   📊 Thresholds BALANCEADOS: Buy {adaptive_thresholds['strong_buy']:.4f}, Sell {adaptive_thresholds['strong_sell']:.4f}")

            return adaptive_thresholds

        except Exception as e:
            print(f"⚠️ Error calculando thresholds: {e}")
            return self.fixed_thresholds[symbol]

    def create_responsive_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Etiquetado BALANCEADO DINÁMICO - Usando percentiles (más eficiente que thresholds fijos)"""

        print(f"🎯 Creando etiquetas BALANCEADAS DINÁMICAS para {symbol} ({self.timeframe})...")

        close_prices = df['close'].values

        # ✅ NUEVA LÓGICA: PERCENTILES DINÁMICOS EN LUGAR DE THRESHOLDS FIJOS
        print("🔧 Calculando retornos futuros...")
        future_returns = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]
            gross_return = (future_price - current_price) / current_price
            future_returns.append(gross_return)

        future_returns = np.array(future_returns)

        # 🎯 CONFIGURACIÓN ESPECÍFICA PARA EL PAR
        if symbol in self.professional_configs and self.config.get('mode') == 'professional':
            # Configuración profesional específica
            config = self.professional_configs[symbol]
            sell_percentile = config['percentile_sell']
            buy_percentile = config['percentile_buy']
            min_profitable_move = config['min_profit_threshold']
            rsi_sell_threshold = config['rsi_sell_threshold']
            rsi_buy_threshold = config['rsi_buy_threshold']
            momentum_threshold = config['momentum_threshold']
            
            print(f"🎯 Usando configuración PROFESIONAL específica para {symbol}")
            print(f"   📊 Percentiles: SELL {sell_percentile}%, BUY {buy_percentile}%")
            print(f"   💰 Mínimo rentable: {min_profitable_move*100:.1f}%")
            print(f"   📈 RSI thresholds: SELL > {rsi_sell_threshold}, BUY < {rsi_buy_threshold}")
            print(f"   ⚡ Momentum threshold: {momentum_threshold*100:.1f}%")
        else:
            # Configuración flexible o por defecto (más permisiva)
            sell_percentile = 35
            buy_percentile = 65
            min_profitable_move = 0.003
            rsi_sell_threshold = 65
            rsi_buy_threshold = 35
            momentum_threshold = 0.005
            
            if self.config.get('mode') == 'flexible':
                print(f"🎯 Usando configuración FLEXIBLE para {symbol}")
            else:
                print(f"⚠️ Usando configuración por defecto (más permisiva) para {symbol}")
            
            print(f"   📊 Percentiles: SELL {sell_percentile}%, BUY {buy_percentile}%")
            print(f"   💰 Mínimo rentable: {min_profitable_move*100:.1f}%")
            print(f"   📈 RSI thresholds: SELL > {rsi_sell_threshold}, BUY < {rsi_buy_threshold}")
            print(f"   ⚡ Momentum threshold: {momentum_threshold*100:.1f}%")

        # 🎯 THRESHOLDS DINÁMICOS BASADOS EN PERCENTILES - PROFESIONALES
        sell_threshold = np.percentile(future_returns, sell_percentile)
        buy_threshold = np.percentile(future_returns, buy_percentile)

        # ✅ VERIFICAR QUE LOS THRESHOLDS NO SEAN IGUALES
        if abs(sell_threshold - buy_threshold) < 0.001:
            print(f"⚠️ ADVERTENCIA: Thresholds muy similares, ajustando...")
            sell_threshold = np.percentile(future_returns, 20)  # Más agresivo
            buy_threshold = np.percentile(future_returns, 80)   # Más agresivo

        # ✅ VERIFICAR QUE LOS THRESHOLDS SEAN RAZONABLES
        if sell_threshold >= buy_threshold:
            print(f"⚠️ ADVERTENCIA: Sell threshold >= Buy threshold, ajustando...")
            sell_threshold = np.percentile(future_returns, 15)  # Más agresivo
            buy_threshold = np.percentile(future_returns, 85)   # Más agresivo

        # Ajustar thresholds si son muy pequeños
        if abs(sell_threshold) < min_profitable_move:
            sell_threshold = -min_profitable_move
        if buy_threshold < min_profitable_move:
            buy_threshold = min_profitable_move

        if self.config.get('mode') == 'professional':
            print(f"💡 Thresholds PROFESIONALES calculados:")
        elif self.config.get('mode') == 'flexible':
            print(f"💡 Thresholds FLEXIBLES calculados:")
        else:
            print(f"💡 Thresholds calculados:")
            
        print(f"   📉 SELL threshold: {sell_threshold*100:.3f}% (percentil {sell_percentile})")
        print(f"   📈 BUY threshold: {buy_threshold*100:.3f}% (percentil {buy_percentile})")
        print(f"   💰 Mínimo rentable: {min_profitable_move*100:.1f}% (rentabilidad neta > 0.5%)")

        # ✅ CREAR ETIQUETAS CON CONFIRMACIÓN TÉCNICA
        labels = []

        for i, return_val in enumerate(future_returns):
            # Clasificación base por percentiles
            if return_val <= sell_threshold:
                candidate_label = 0  # SELL
            elif return_val >= buy_threshold:
                candidate_label = 2  # BUY
            else:
                candidate_label = 1  # HOLD

            # 🔧 CONFIRMACIÓN TÉCNICA para mejorar calidad
            try:
                if i < len(features):
                    current_rsi = features['rsi_14'].iloc[i] if 'rsi_14' in features.columns else 50
                    current_macd = features['macd_histogram'].iloc[i] if 'macd_histogram' in features.columns else 0
                else:
                    current_rsi = 50
                    current_macd = 0

                # Filtros de confirmación técnica MÁS PERMISIVOS
                if candidate_label == 0:  # SELL candidato
                    # Confirmar con indicadores bajistas (más permisivo)
                    if current_rsi > rsi_sell_threshold or current_macd > 0:  # Más permisivo
                        label = 0  # SELL confirmado
                    else:
                        label = 1  # HOLD (falta confirmación)
                elif candidate_label == 2:  # BUY candidato
                    # Confirmar con indicadores alcistas (más permisivo)
                    if current_rsi < rsi_buy_threshold or current_macd < 0:  # Más permisivo
                        label = 2  # BUY confirmado
                    else:
                        label = 1  # HOLD (falta confirmación)
                else:
                    # HOLD con posible escalado por momentum MÁS PERMISIVO
                    if i >= 5:
                        momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                        if momentum > momentum_threshold * 0.8 and current_rsi < 60:  # Más permisivo
                            label = 2  # HOLD -> BUY por momentum
                        elif momentum < -momentum_threshold * 0.8 and current_rsi > 40:  # Más permisivo
                            label = 0  # HOLD -> SELL por momentum
                        else:
                            label = 1  # HOLD mantenido
                    else:
                        label = 1  # HOLD

            except:
                # En caso de error, usar clasificación base
                label = candidate_label

            labels.append(label)

        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # 📊 VERIFICAR DISTRIBUCIÓN FINAL
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas BALANCEADAS:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        # ✅ VERIFICAR DISTRIBUCIÓN MÍNIMA PARA ENTRENAMIENTO
        min_samples_per_class = 2
        classes_with_few_samples = []
        
        for i in range(3):  # 0=SELL, 1=HOLD, 2=BUY
            count = label_counts.get(i, 0) or 0
            if count < min_samples_per_class:
                classes_with_few_samples.append(i)
        
        if classes_with_few_samples:
            print(f"⚠️ ADVERTENCIA: Clases con pocas muestras: {classes_with_few_samples}")
            print(f"🔧 Aplicando corrección de distribución...")
            
            # Corregir distribución forzando más muestras en clases con pocos datos
            labels_corrected = []
            for i, return_val in enumerate(future_returns):
                if i < len(labels):
                    label = labels[i]
                    
                    # Si la clase tiene pocas muestras, ser más permisivo
                    if label in classes_with_few_samples:
                        if label == 0:  # SELL
                            # Ser más permisivo con SELL
                            if return_val <= sell_threshold * 1.5:  # Más permisivo
                                labels_corrected.append(0)
                            else:
                                labels_corrected.append(1)  # HOLD
                        elif label == 2:  # BUY
                            # Ser más permisivo con BUY
                            if return_val >= buy_threshold * 0.8:  # Más permisivo
                                labels_corrected.append(2)
                            else:
                                labels_corrected.append(1)  # HOLD
                        else:
                            labels_corrected.append(label)
                    else:
                        labels_corrected.append(label)
                else:
                    labels_corrected.append(1)  # HOLD por defecto
            
            labels = labels_corrected
            
            # Verificar distribución corregida
            label_counts = pd.Series(labels).value_counts().sort_index()
            print("📊 Distribución CORREGIDA:")
            for i, name in enumerate(class_names):
                count = label_counts.get(i, 0) or 0
                pct = (count / total * 100) if total > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")

            # ✅ VERIFICAR SI LA CORRECCIÓN FUNCIONÓ
            classes_with_few_samples_after = []
            for i in range(3):
                count = label_counts.get(i, 0) or 0
                if count < min_samples_per_class:
                    classes_with_few_samples_after.append(i)
            
            if classes_with_few_samples_after:
                print(f"⚠️ ADVERTENCIA: Corrección insuficiente. Aplicando fallback...")
                # Fallback: distribución más permisiva
                labels_fallback = []
                for i, return_val in enumerate(future_returns):
                    if return_val <= np.percentile(future_returns, 35):  # Más permisivo
                        labels_fallback.append(0)  # SELL
                    elif return_val >= np.percentile(future_returns, 65):  # Más permisivo
                        labels_fallback.append(2)  # BUY
                    else:
                        labels_fallback.append(1)  # HOLD
                
                labels = labels_fallback
                label_counts = pd.Series(labels).value_counts().sort_index()
                print("📊 Distribución FALLBACK:")
                for i, name in enumerate(class_names):
                    count = label_counts.get(i, 0) or 0
                    pct = (count / total * 100) if total > 0 else 0
                    print(f"   - {name}: {count} ({pct:.1f}%)")

        # 🎯 VALIDAR QUE LA DISTRIBUCIÓN ES BALANCEADA
        max_class_pct = max([count/total for count in label_counts.values]) * 100
        min_class_pct = min([count/total for count in label_counts.values]) * 100
        balance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')

        if balance_ratio > 3.0:  # Si una clase es >3x otra
            print(f"⚠️ ADVERTENCIA: Distribución aún desbalanceada (ratio: {balance_ratio:.1f})")
        else:
            print(f"✅ Distribución balanceada: ratio max/min = {balance_ratio:.1f}")

        # ✅ ANÁLISIS DE RENTABILIDAD MEJORADO
        self._analyze_profitability_potential(df, labels, symbol, sell_threshold, buy_threshold)

        return df_labeled

    def _analyze_profitability_potential(self, df: pd.DataFrame, labels: list, symbol: str, sell_threshold: float, buy_threshold: float):
        """💰 Análisis de rentabilidad potencial con thresholds dinámicos"""
        try:
            print(f"\n💰 ANÁLISIS DE RENTABILIDAD POTENCIAL - {symbol}")
            print("=" * 60)

            close_prices = df['close'].values
            trading_costs = 0.003  # 0.3%

            profitable_buys = 0
            profitable_sells = 0
            total_buys = 0
            total_sells = 0
            total_profit_buys = 0.0
            total_profit_sells = 0.0

            for i, label in enumerate(labels):
                if i + self.prediction_horizon >= len(close_prices):
                    break

                current_price = close_prices[i]
                future_price = close_prices[i + self.prediction_horizon]
                gross_return = (future_price - current_price) / current_price

                if label == 2:  # BUY
                    total_buys += 1
                    net_return = gross_return - trading_costs
                    if net_return > 0:
                        profitable_buys += 1
                        total_profit_buys += net_return

                elif label == 0:  # SELL
                    total_sells += 1
                    net_return = -gross_return - trading_costs
                    if net_return > 0:
                        profitable_sells += 1
                        total_profit_sells += net_return

            # 📊 ESTADÍSTICAS DE RENTABILIDAD
            buy_win_rate = (profitable_buys / total_buys * 100) if total_buys > 0 else 0
            sell_win_rate = (profitable_sells / total_sells * 100) if total_sells > 0 else 0
            avg_buy_profit = (total_profit_buys / total_buys * 100) if total_buys > 0 else 0
            avg_sell_profit = (total_profit_sells / total_sells * 100) if total_sells > 0 else 0

            print(f"📊 Estadísticas de rentabilidad:")
            print(f"   🟢 BUY: {total_buys} operaciones, {buy_win_rate:.1f}% win rate, {avg_buy_profit:.2f}% avg profit")
            print(f"   🔴 SELL: {total_sells} operaciones, {sell_win_rate:.1f}% win rate, {avg_sell_profit:.2f}% avg profit")

            # 🎯 EVALUACIÓN DE CALIDAD
            total_operations = total_buys + total_sells
            overall_win_rate = ((profitable_buys + profitable_sells) / total_operations * 100) if total_operations > 0 else 0
            total_profit = total_profit_buys + total_profit_sells

            print(f"🎯 Evaluación general:")
            print(f"   📊 Operaciones totales: {total_operations}")
            print(f"   📈 Win rate general: {overall_win_rate:.1f}%")
            print(f"   💰 Profit total: {total_profit*100:.2f}%")

            if overall_win_rate >= 55 and total_profit > 0:
                print(f"✅ EXCELENTE: Modelo rentable y balanceado")
            elif overall_win_rate >= 50 and total_profit > 0:
                print(f"✅ BUENO: Modelo rentable")
            elif overall_win_rate >= 45:
                print(f"⚠️ ACEPTABLE: Win rate bajo pero funcional")
            else:
                print(f"❌ PROBLEMÁTICO: Win rate muy bajo")

        except Exception as e:
            print(f"⚠️ Error en análisis de rentabilidad: {e}")

    async def get_real_market_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """📊 Obtener datos con timeframe configurable"""
        days = days or self.days
        print(f"📊 Obteniendo {days} días de datos {self.timeframe} para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': self.timeframe,  # ✅ USAR TIMEFRAME CONFIGURABLE
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
                        current_start = data[-1][6] + 1
                    else:
                        print(f"❌ Error API: {response.status}")
                        break

                await asyncio.sleep(0.1)

        # Convertir a DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(all_data, columns=columns)

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros {self.timeframe} de {symbol}")
        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparación optimizada"""
        print("🔧 Preparando datos optimizados...")

        features_aligned = features.iloc[:-self.prediction_horizon]
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados: X shape: {X.shape}, y shape: {y.shape}")

        # Class weights balanceados para responsividad
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        # Suavizar weights extremos
        class_weights = np.clip(class_weights, 0.3, 2.5)  # Más permisivo
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_optimized_tcn_model(self, input_shape: tuple):
        """🎯 Modelo TCN LIGERO - <350k parámetros"""

        print("🎯 Creando modelo TCN LIGERO...")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),

            # Normalización inicial
            tf.keras.layers.LayerNormalization(),

            # Bloques TCN LIGEROS
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            # Global pooling y capas densas LIGERAS
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.15),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.15),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Optimizador y learning rate optimizados
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        param_count = model.count_params()
        print(f"✅ Modelo LIGERO creado: {param_count:,} parámetros")
        
        if param_count > 350000:
            print(f"⚠️ ADVERTENCIA: Modelo excede 350k parámetros ({param_count:,})")
        else:
            print(f"✅ Modelo dentro del límite: {param_count:,} < 350,000")

        return model

    async def train_optimized_model(self, symbol: str) -> bool:
        """🎯 Entrenamiento PROFESIONAL con configuración específica por par"""

        # 🎯 APLICAR CONFIGURACIÓN ESPECÍFICA
        if symbol in self.professional_configs and self.config.get('mode') == 'professional':
            # Configuración profesional predefinida
            config = self.professional_configs[symbol]
            self.timeframe = config['timeframe']
            self.lookback_window = config['lookback_window']
            self.prediction_horizon = config['prediction_horizon']
            self.days = config['days']
            
            print(f"\n🎯 ENTRENANDO MODELO PROFESIONAL PARA {symbol}")
            print("=" * 70)
            print(f"🎯 Configuración PROFESIONAL específica:")
            print(f"   ⏰ Timeframe: {self.timeframe}")
            print(f"   📊 Lookback: {self.lookback_window} períodos ({self.lookback_window * int(self.timeframe[0])}min)")
            print(f"   ⏰ Horizonte: {self.prediction_horizon} períodos ({self.prediction_horizon * int(self.timeframe[0])}min)")
            print(f"   📅 Días: {self.days}")
            print(f"   💰 Mínimo rentable: {config['min_profit_threshold']*100:.1f}%")
            print(f"   📈 Selectividad: SELL {config['percentile_sell']}%, BUY {config['percentile_buy']}%")
        else:
            # Configuración flexible o personalizada
            if self.config.get('mode') == 'flexible':
                print(f"\n🎯 ENTRENANDO MODELO FLEXIBLE PARA {symbol}")
                print("=" * 70)
                print(f"🎯 Configuración FLEXIBLE:")
                print(f"   ⏰ Timeframe: {self.timeframe}")
                print(f"   📊 Lookback: {self.lookback_window} períodos ({self.lookback_window * int(self.timeframe[0])}min)")
                print(f"   ⏰ Horizonte: {self.prediction_horizon} períodos ({self.prediction_horizon * int(self.timeframe[0])}min)")
                print(f"   📅 Días: {self.days}")
                print(f"   🎯 Agresividad: {self.aggressiveness}")
            else:
                print(f"\n🎯 ENTRENANDO MODELO PERSONALIZADO PARA {symbol}")
                print("=" * 70)
                print(f"🎯 Configuración PERSONALIZADA:")
                print(f"   ⏰ Timeframe: {self.timeframe}")
                print(f"   📊 Lookback: {self.lookback_window} períodos ({self.lookback_window * int(self.timeframe[0])}min)")
                print(f"   ⏰ Horizonte: {self.prediction_horizon} períodos ({self.prediction_horizon * int(self.timeframe[0])}min)")
                print(f"   📅 Días: {self.days}")
                print(f"   🎯 Agresividad: {self.aggressiveness}")

        try:
            # 1. Obtener datos con timeframe configurable
            df = await self.get_real_market_data(symbol, days=self.days)

            # 2. Calcular features
            print(f"🔄 Calculando features...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            # 3. Etiquetas BALANCEADAS
            df_labeled = self.create_responsive_labels(df, features, symbol)

            # 4. Preparar datos
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # 5. Verificar distribución antes del split
            unique_labels, counts = np.unique(y, return_counts=True)
            min_samples = min(counts)
            
            if min_samples < 2:
                print(f"❌ ERROR: Clase con menos de 2 muestras. Distribución: {dict(zip(unique_labels, counts))}")
                print(f"🔧 Aplicando split sin stratify...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.15, random_state=42
                )
            else:
                print(f"✅ Distribución válida para stratify. Mínimo: {min_samples} muestras por clase")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.15, random_state=42, stratify=y
                )

            # 6. Modelo optimizado
            model = self.create_optimized_tcn_model((X.shape[1], X.shape[2]))

            # ✅ DIRECTORIO CON CONFIGURACIÓN
            if self.config.get('mode') == 'flexible':
                model_dir = f'models/flexible_{self.timeframe}_{symbol.lower()}'
            else:
                model_dir = f'models/professional_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)

            # Callbacks optimizados
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
                tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.7, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(
                    f'{model_dir}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            if self.config.get('mode') == 'flexible':
                print("🚀 Entrenamiento FLEXIBLE...")
            else:
                print("🚀 Entrenamiento PROFESIONAL...")

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=150,
                batch_size=64,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            # 7. Evaluación
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            if self.config.get('mode') == 'flexible':
                print(f"✅ Accuracy FLEXIBLE: {test_acc:.3f}")
            else:
                print(f"✅ Accuracy PROFESIONAL: {test_acc:.3f}")

            # 8. Guardar con configuración
            model.save(f'{model_dir}/model.h5')

            with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            with open(f'{model_dir}/class_weights.pkl', 'wb') as f:
                pickle.dump(class_weights, f)

            # ✅ GUARDAR CONFIGURACIÓN
            config_save = {
                'timeframe': self.timeframe,
                'aggressiveness': self.aggressiveness,
                'lookback_window': self.lookback_window,
                'prediction_horizon': self.prediction_horizon,
                'days': self.days,
                'force_signals': self.force_signals,
                'use_adaptive_thresholds': self.use_adaptive_thresholds
            }
            
            with open(f'{model_dir}/config.pkl', 'wb') as f:
                pickle.dump(config_save, f)

            if self.config.get('mode') == 'flexible':
                print(f"✅ Modelo FLEXIBLE guardado en {model_dir}/")
            else:
                print(f"✅ Modelo PROFESIONAL guardado en {model_dir}/")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False


def get_flexible_configuration():
    """🎯 Configuración FLEXIBLE con múltiples opciones"""

    print("\n🎯 CONFIGURACIÓN FLEXIBLE - MÚLTIPLES OPCIONES")
    print("=" * 60)

    # 1. Símbolo
    available_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT']
    print(f"\n📊 Símbolos disponibles: {', '.join(available_symbols)}")
    symbol = input("🎯 Ingresa el símbolo (ej: BTCUSDT): ").upper().strip()
    if symbol not in available_symbols:
        print(f"⚠️ Símbolo no válido, usando BTCUSDT")
        symbol = 'BTCUSDT'

    # 2. Timeframe
    timeframes = ['1m', '3m', '5m']
    print(f"\n⏰ Timeframes disponibles: {', '.join(timeframes)}")
    timeframe = input("⏰ Ingresa el timeframe (1m/3m/5m): ").lower().strip()
    if timeframe not in timeframes:
        print(f"⚠️ Timeframe no válido, usando 5m")
        timeframe = '5m'

    # 3. Días de entrenamiento
    days_options = [24, 32, 48, 60, 75, 90]
    print(f"\n📅 Días de entrenamiento disponibles: {', '.join(map(str, days_options))}")
    try:
        days = int(input("📅 Número de días: ").strip())
        if days not in days_options:
            print(f"⚠️ Días no válidos, usando 48")
            days = 48
    except ValueError:
        print("⚠️ Días no válidos, usando 48")
        days = 48

    # 4. Horizonte predictivo
    horizons = [6, 8, 12, 15, 18]
    print(f"\n⏰ Horizontes predictivos disponibles: {', '.join(map(str, horizons))}")
    try:
        prediction_horizon = int(input("⏰ Horizonte predictivo: ").strip())
        if prediction_horizon not in horizons:
            print(f"⚠️ Horizonte no válido, usando 12")
            prediction_horizon = 12
    except ValueError:
        print("⚠️ Horizonte no válido, usando 12")
        prediction_horizon = 12

    # 5. Lookback window (basado en timeframe)
    lookback_options = {
        '1m': [12, 24, 32, 48, 60, 72],
        '3m': [12, 24, 32, 48, 60, 72],
        '5m': [12, 24, 32, 48, 60, 72]
    }
    
    available_lookbacks = lookback_options[timeframe]
    print(f"\n📊 Lookback windows disponibles para {timeframe}: {', '.join(map(str, available_lookbacks))}")
    print(f"   💡 Recomendado: {available_lookbacks[2]} ({available_lookbacks[2] * int(timeframe[0])} minutos)")
    
    try:
        lookback_window = int(input("📊 Lookback window: ").strip())
        if lookback_window not in available_lookbacks:
            print(f"⚠️ Lookback no válido, usando {available_lookbacks[2]}")
            lookback_window = available_lookbacks[2]
    except ValueError:
        print(f"⚠️ Lookback no válido, usando {available_lookbacks[2]}")
        lookback_window = available_lookbacks[2]

    # 6. Configuración avanzada
    print(f"\n🔧 Configuración avanzada:")
    advanced = input("🔧 ¿Configurar parámetros avanzados? (s/n): ").lower().strip()

    force_signals = False
    use_adaptive_thresholds = True

    if advanced == 's':
        force_signals_input = input("🔧 ¿Forzar señales en lugar de HOLD? (s/n): ").lower().strip()
        force_signals = force_signals_input == 's'
        print("💡 Nota: En sistema balanceado, forzar señales está deshabilitado por defecto")

        adaptive_input = input("🔧 ¿Usar thresholds adaptativos? (s/n): ").lower().strip()
        use_adaptive_thresholds = adaptive_input == 's'

    # Crear configuración
    config = {
        'pairs': [symbol],
        'timeframe': timeframe,
        'lookback_window': lookback_window,
        'prediction_horizon': prediction_horizon,
        'days': days,
        'force_signals': force_signals,
        'use_adaptive_thresholds': use_adaptive_thresholds,
        'mode': 'flexible'
    }

    print(f"\n✅ CONFIGURACIÓN FLEXIBLE FINAL:")
    print(f"   📊 Símbolo: {symbol}")
    print(f"   ⏰ Timeframe: {timeframe}")
    print(f"   📅 Días: {days}")
    print(f"   ⏰ Horizonte: {prediction_horizon}")
    print(f"   📊 Lookback: {lookback_window} ({lookback_window * int(timeframe[0])} minutos)")
    print(f"   🔥 Forzar señales: {force_signals}")
    print(f"   📊 Thresholds adaptativos: {use_adaptive_thresholds}")

    return config


def get_user_configuration():
    """🎯 Obtener configuración avanzada del usuario"""

    print("\n🎯 CONFIGURACIÓN AVANZADA DEL ENTRENADOR BALANCEADO")
    print("=" * 60)

    # 1. Símbolo
    available_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT']
    print(f"\n📊 Símbolos disponibles: {', '.join(available_symbols)}")
    symbol = input("🎯 Ingresa el símbolo (ej: BTCUSDT): ").upper().strip()
    if symbol not in available_symbols:
        print(f"⚠️ Símbolo no válido, usando BTCUSDT")
        symbol = 'BTCUSDT'

    # 2. Timeframe
    available_timeframes = ['1m', '3m', '5m', '15m', '1h', '4h']
    print(f"\n⏰ Timeframes disponibles: {', '.join(available_timeframes)}")
    timeframe = input("⏰ Ingresa el timeframe (ej: 5m): ").lower().strip()
    if timeframe not in available_timeframes:
        print(f"⚠️ Timeframe no válido, usando 5m")
        timeframe = '5m'

    # 3. Agresividad
    print(f"\n🎯 Niveles de agresividad:")
    print("   - conservative: Menos señales, más precisión")
    print("   - balanced: Balance entre señales y precisión")
    print("   - aggressive: Más señales, menos precisión")
    aggressiveness = input("🎯 Selecciona agresividad (conservative/balanced/aggressive): ").lower().strip()
    if aggressiveness not in ['conservative', 'balanced', 'aggressive']:
        print(f"⚠️ Agresividad no válida, usando balanced")
        aggressiveness = 'balanced'

    # 4. Días de entrenamiento
    try:
        days = int(input("📅 Número de días (ej: 30): ").strip())
        if days <= 0 or days > 365:
            print("⚠️ Días no válidos, usando 30")
            days = 30
    except ValueError:
        print("⚠️ Días no válidos, usando 30")
        days = 30

    # 5. Configuración avanzada
    print(f"\n🔧 Configuración avanzada:")
    advanced = input("🔧 ¿Configurar parámetros avanzados? (s/n): ").lower().strip()

    lookback_window = 24
    prediction_horizon = 6
    force_signals = True
    use_adaptive_thresholds = True

    if advanced == 's':
        try:
            lookback_window = int(input("🔧 Lookback window (ej: 24): ").strip())
            if lookback_window <= 0:
                print("⚠️ Lookback window no válido, usando 24")
                lookback_window = 24
        except ValueError:
            print("⚠️ Lookback window no válido, usando 24")
            lookback_window = 24

        try:
            prediction_horizon = int(input("🔧 Prediction horizon (ej: 6): ").strip())
            if prediction_horizon <= 0:
                print("⚠️ Prediction horizon no válido, usando 6")
                prediction_horizon = 6
        except ValueError:
            print("⚠️ Prediction horizon no válido, usando 6")
            prediction_horizon = 6

        force_signals_input = input("🔧 ¿Forzar señales en lugar de HOLD? (s/n): ").lower().strip()
        force_signals = force_signals_input == 's'
        print("💡 Nota: En sistema balanceado, forzar señales está deshabilitado por defecto")

        adaptive_input = input("🔧 ¿Usar thresholds adaptativos? (s/n): ").lower().strip()
        use_adaptive_thresholds = adaptive_input == 's'

    # Crear configuración
    config = {
        'pairs': [symbol],
        'timeframe': timeframe,
        'lookback_window': lookback_window,
        'prediction_horizon': prediction_horizon,
        'days': days,
        'aggressiveness': aggressiveness,
        'force_signals': force_signals,
        'use_adaptive_thresholds': use_adaptive_thresholds
    }

    print(f"\n✅ CONFIGURACIÓN BALANCEADA FINAL:")
    print(f"   📊 Símbolo: {symbol}")
    print(f"   ⏰ Timeframe: {timeframe}")
    print(f"   🎯 Agresividad: {aggressiveness}")
    print(f"   📅 Días: {days}")
    print(f"   📊 Lookback: {lookback_window}")
    print(f"   ⏰ Horizonte: {prediction_horizon}")
    print(f"   🔥 Forzar señales: {force_signals}")
    print(f"   📊 Thresholds adaptativos: {use_adaptive_thresholds}")

    return config


def get_optimized_configuration():
    """🎯 Configuraciones PROFESIONALES predefinidas"""

    print("\n🎯 CONFIGURACIONES PROFESIONALES")
    print("=" * 50)
    print("1. 🚀 Entrenar BTCUSDT (Profesional)")
    print("2. 📊 Entrenar ETHUSDT (Profesional)")
    print("3. 📈 Entrenar BNBUSDT (Profesional)")
    print("4. ⚡ Entrenar XRPUSDT (Profesional)")
    print("5. 🎯 Entrenar DOTUSDT (Profesional)")
    print("6. 🔥 Entrenar TODOS los pares (Profesional)")
    print("7. ⚙️ Configuración FLEXIBLE (Timeframes, Días, Horizontes)")
    print("8. 🎯 Configuración Personalizada")

    choice = input("\n🎯 Selecciona configuración (1-8): ").strip()

    if choice == '1':
        return {
            'pairs': ['BTCUSDT'],
            'mode': 'professional'
        }
    elif choice == '2':
        return {
            'pairs': ['ETHUSDT'],
            'mode': 'professional'
        }
    elif choice == '3':
        return {
            'pairs': ['BNBUSDT'],
            'mode': 'professional'
        }
    elif choice == '4':
        return {
            'pairs': ['XRPUSDT'],
            'mode': 'professional'
        }
    elif choice == '5':
        return {
            'pairs': ['DOTUSDT'],
            'mode': 'professional'
        }
    elif choice == '6':
        return {
            'pairs': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT'],
            'mode': 'professional_all'
        }
    elif choice == '7':
        return get_flexible_configuration()
    else:
        return get_user_configuration()


async def main():
    """🎯 Entrenador responsivo con configuración flexible"""

    print("🎯 TCN TRAINER V2 PROFESIONAL - MODELOS ESPECÍFICOS POR PAR")
    print("=" * 80)
    print("🚀 Características PROFESIONALES:")
    print("   ✅ Configuraciones específicas por par")
    print("   ✅ Thresholds optimizados por volatilidad")
    print("   ✅ Etiquetado profesional con percentiles dinámicos")
    print("   ✅ Confirmación técnica avanzada (RSI, MACD)")
    print("   ✅ Análisis de rentabilidad profesional")
    print("   ✅ Rentabilidad mínima > 0.5%")
    print("=" * 80)

    # Seleccionar tipo de configuración
    print("\n🎯 TIPO DE CONFIGURACIÓN:")
    print("1. 🚀 Configuración Optimizada (Recomendado)")
    print("2. ⚙️ Configuración Personalizada")

    config_type = input("\n🎯 Selecciona tipo (1-2): ").strip()

    if config_type == '1':
        config = get_optimized_configuration()
    else:
        config = get_user_configuration()

    # Crear trainer con configuración
    trainer = OptimizedTCNTrainer(config)

    # Entrenar modelo(s)
    if config.get('mode') == 'professional_all':
        print(f"\n🔥 ENTRENANDO TODOS LOS PARES PROFESIONALES")
        print("=" * 60)
        
        successful_pairs = []
        failed_pairs = []
        
        for symbol in config['pairs']:
            print(f"\n🎯 Entrenando {symbol}...")
            success = await trainer.train_optimized_model(symbol)
            
            if success:
                successful_pairs.append(symbol)
                print(f"✅ {symbol}: MODELO PROFESIONAL COMPLETADO")
                print(f"🎯 Guardado en: models/professional_{symbol.lower()}/")
            else:
                failed_pairs.append(symbol)
                print(f"❌ {symbol}: ERROR EN ENTRENAMIENTO")
        
        # Resumen final
        print(f"\n📊 RESUMEN FINAL:")
        print(f"✅ Exitosos: {len(successful_pairs)} - {', '.join(successful_pairs)}")
        print(f"❌ Fallidos: {len(failed_pairs)} - {', '.join(failed_pairs)}")
        
        if successful_pairs:
            print(f"\n🎯 Modelos profesionales guardados en:")
            for symbol in successful_pairs:
                print(f"   📁 models/professional_{symbol.lower()}/")
        
        success = len(successful_pairs) > 0
        
    else:
        # Entrenar un solo par
        symbol = config['pairs'][0]
        
        if config.get('mode') == 'flexible':
            print(f"\n🚀 Entrenando modelo flexible para {symbol}...")
            print(f"📊 Configuración flexible activada")
        else:
            print(f"\n🚀 Entrenando modelo profesional para {symbol}...")
            print(f"📊 Configuración profesional activada")

        success = await trainer.train_optimized_model(symbol)

        if success:
            if config.get('mode') == 'flexible':
                print(f"\n✅ {symbol}: MODELO FLEXIBLE COMPLETADO EXITOSAMENTE")
                print(f"🎯 Modelo guardado en: models/flexible_{config['timeframe']}_{symbol.lower()}/")
            else:
                print(f"\n✅ {symbol}: MODELO PROFESIONAL COMPLETADO EXITOSAMENTE")
                print(f"🎯 Modelo guardado en: models/professional_{symbol.lower()}/")
            print(f"📁 Archivos incluidos:")
            print(f"   - best_model.h5 (modelo entrenado)")
            print(f"   - scaler.pkl (normalización)")
            print(f"   - feature_columns.pkl (features)")
            print(f"   - class_weights.pkl (pesos de clases)")
            print(f"   - config.pkl (configuración)")
        else:
            print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO")

    # Preguntar si entrenar más modelos
    train_more = input("\n🤔 ¿Entrenar otro modelo? (s/n): ").lower().strip()

    while train_more == 's':
        config = get_user_configuration()
        trainer = OptimizedTCNTrainer(config)
        symbol = config['pairs'][0]
        timeframe = config['timeframe']

        print(f"\n🚀 Entrenando modelo {timeframe} para {symbol}...")
        success = await trainer.train_optimized_model(symbol)

        if success:
            print(f"\n✅ {symbol}: MODELO BALANCEADO COMPLETADO")
            print(f"🎯 Guardado en: models/balanced_{timeframe}_{symbol.lower()}/")
        else:
            print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO")

        train_more = input("\n🤔 ¿Entrenar otro modelo? (s/n): ").lower().strip()

    print(f"\n🎉 ¡ENTRENAMIENTO PROFESIONAL COMPLETADO!")
    print(f"🎯 Revisa los modelos en el directorio models/")


if __name__ == "__main__":
    asyncio.run(main())
