#!/usr/bin/env python3
"""
🚀 BACKTESTING UNIVERSAL - SELECTOR DE MODELOS
Script para probar cualquier modelo disponible en el directorio models/
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import warnings
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from centralized_features_engine2 import CentralizedFeaturesEngine

warnings.filterwarnings('ignore')

class UniversalBacktester:
    """🎯 Backtester universal para cualquier modelo"""
    
    def __init__(self):
        # Inicializar motor de features centralizado
        self.features_engine = CentralizedFeaturesEngine()
        
        # Configuración de trading
        self.initial_balance = 1000.0  # $1000 USD inicial
        self.trading_fee = 0.001      # 0.1% fee por trade
        self.min_trade_amount = 10.0   # Mínimo $10 por trade
        self.lookback_window = 24      # Default, se auto-ajustará por modelo
        
        # Configuración del modelo actual
        self.model_path = None
        self.symbol = None
        self.timeframe = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.class_weights = None
        
        # Métricas de rendimiento
        self.trades = []
        self.balance_history = []
        self.predictions_history = []
        
        print(f"🚀 Backtester Universal inicializado")
        print(f"💰 Balance inicial: ${self.initial_balance}")
        print(f"💸 Fee de trading: {self.trading_fee*100:.1f}%")
        print(f"🔧 Motor de features: Centralizado")
    
    def discover_models(self) -> List[Dict]:
        """🔍 Descubrir automáticamente todos los modelos disponibles"""
        
        print("🔍 Descubriendo modelos disponibles...")
        
        models_dir = "models"
        if not os.path.exists(models_dir):
            print(f"❌ Directorio {models_dir} no encontrado")
            return []
        
        models = []
        
        for dir_name in os.listdir(models_dir):
            dir_path = os.path.join(models_dir, dir_name)
            
            # Solo directorios
            if not os.path.isdir(dir_path) or dir_name.startswith('.'):
                continue
            
            # Verificar que tenga archivos de modelo
            required_files = ['model.h5', 'scaler.pkl', 'feature_columns.pkl']
            alternative_files = ['best_model.h5']
            
            has_model = False
            model_file = None
            
            # Verificar archivos requeridos
            if os.path.exists(os.path.join(dir_path, 'best_model.h5')):
                has_model = True
                model_file = 'best_model.h5'
            elif os.path.exists(os.path.join(dir_path, 'model.h5')):
                has_model = True
                model_file = 'model.h5'
            
            # Verificar otros archivos necesarios
            has_scaler = os.path.exists(os.path.join(dir_path, 'scaler.pkl'))
            has_features = os.path.exists(os.path.join(dir_path, 'feature_columns.pkl'))
            
            if has_model and has_scaler and has_features:
                # Extraer información del nombre
                symbol, timeframe = self._extract_symbol_timeframe(dir_name)
                
                if symbol:
                    # 🔢 Contar parámetros del modelo
                    model_full_path = os.path.join(dir_path, model_file)
                    parameter_count = self._count_model_parameters(model_full_path)
                    
                    models.append({
                        'name': dir_name,
                        'path': dir_path,
                        'symbol': symbol,
                        'timeframe': timeframe or '5m',  # Default 5m
                        'model_file': model_file,
                        'parameters': parameter_count,
                        'complete': True
                    })
                    
                    # Clasificar por tamaño de parámetros
                    if parameter_count > 0:
                        if parameter_count < 50000:
                            size_indicator = "🟢"  # Optimizado
                            size_label = "Opt"
                        elif parameter_count < 200000:
                            size_indicator = "🟡"  # Intermedio
                            size_label = "Med"
                        else:
                            size_indicator = "🔴"  # Posible overfitting
                            size_label = "Big"
                        
                        print(f"   ✅ {dir_name} -> {symbol} ({timeframe or '5m'}) {size_indicator} {parameter_count:,} parámetros")
                    else:
                        print(f"   ✅ {dir_name} -> {symbol} ({timeframe or '5m'}) ❓ parámetros")
                else:
                    print(f"   ⚠️ {dir_name} -> No se pudo extraer símbolo")
            else:
                missing = []
                if not has_model: missing.append("modelo")
                if not has_scaler: missing.append("scaler")
                if not has_features: missing.append("features")
                print(f"   ❌ {dir_name} -> Faltan: {', '.join(missing)}")
        
        print(f"✅ {len(models)} modelos válidos encontrados")
        return models
    
    def _count_model_parameters(self, model_path: str) -> int:
        """🔢 Contar parámetros del modelo"""
        try:
            import tensorflow as tf
            # Suprimir logs de TensorFlow
            tf.get_logger().setLevel('ERROR')
            
            # Cargar modelo temporalmente
            model = tf.keras.models.load_model(model_path, compile=False)
            param_count = model.count_params()
            
            # Limpiar memoria
            del model
            tf.keras.backend.clear_session()
            
            return param_count
        except Exception as e:
            print(f"   ⚠️ Error contando parámetros: {str(e)[:50]}...")
            return 0
    
    def _extract_symbol_timeframe(self, dir_name: str) -> Tuple[Optional[str], Optional[str]]:
        """🔧 Extraer símbolo y timeframe del nombre del directorio"""
        
        # Lista de símbolos conocidos
        known_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT', 
                        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'LINKUSDT', 'MATICUSDT']
        
        # Convertir a mayúsculas para búsqueda
        dir_upper = dir_name.upper()
        
        # Buscar símbolo
        symbol = None
        for s in known_symbols:
            if s in dir_upper:
                symbol = s
                break
        
        # Buscar timeframe
        timeframe = None
        timeframe_patterns = [
            r'_(\d+[mh])_',  # _5m_, _1h_
            r'_(\d+[mh])$',  # _5m, _1h al final
            r'^(\d+[mh])_',  # 5m_, 1h_ al inicio
        ]
        
        for pattern in timeframe_patterns:
            match = re.search(pattern, dir_name.lower())
            if match:
                timeframe = match.group(1)
                break
        
        # Si no encuentra timeframe pero hay _5m en el nombre
        if not timeframe and '5m' in dir_name.lower():
            timeframe = '5m'
        elif not timeframe and '1m' in dir_name.lower():
            timeframe = '1m'
        elif not timeframe and '15m' in dir_name.lower():
            timeframe = '15m'
        elif not timeframe and '1h' in dir_name.lower():
            timeframe = '1h'
        elif not timeframe and '4h' in dir_name.lower():
            timeframe = '4h'
        
        return symbol, timeframe
    
    def select_model(self, models: List[Dict]) -> Optional[Dict]:
        """🎯 Permitir al usuario seleccionar un modelo"""
        
        if not models:
            print("❌ No hay modelos disponibles")
            return None
        
        print(f"\n🎯 MODELOS DISPONIBLES:")
        print("=" * 80)
        
        for i, model in enumerate(models, 1):
            # Clasificar por tamaño de parámetros
            parameters = model.get('parameters', 0)
            if parameters > 0:
                if parameters < 50000:
                    size_indicator = "🟢"  # Optimizado
                    size_label = "Optimizado"
                elif parameters < 200000:
                    size_indicator = "🟡"  # Intermedio
                    size_label = "Intermedio"
                else:
                    size_indicator = "🔴"  # Posible overfitting
                    size_label = "Alto riesgo"
                
                param_text = f"{size_indicator} {parameters:,} parámetros ({size_label})"
            else:
                param_text = "❓ Parámetros desconocidos"
            
            print(f"{i:2d}. {model['name']}")
            print(f"    📊 Símbolo: {model['symbol']}")
            print(f"    ⏰ Timeframe: {model['timeframe']}")
            print(f"    📁 Archivo: {model['model_file']}")
            print(f"    🔢 {param_text}")
            print()
        
        # Leyenda de indicadores
        print("🔢 LEYENDA DE PARÁMETROS:")
        print("   🟢 Optimizado: < 50K parámetros (Recomendado)")
        print("   🟡 Intermedio: 50K-200K parámetros (Moderado)")  
        print("   🔴 Alto riesgo: > 200K parámetros (Posible overfitting)")
        print("=" * 80)
        
        while True:
            try:
                choice = input(f"🎯 Selecciona modelo (1-{len(models)}): ").strip()
                if choice.lower() in ['q', 'quit', 'salir']:
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected = models[choice_num - 1]
                    print(f"\n✅ Modelo seleccionado: {selected['name']}")
                    print(f"   📊 {selected['symbol']} - {selected['timeframe']}")
                    return selected
                else:
                    print(f"❌ Selecciona un número entre 1 y {len(models)}")
            except ValueError:
                print("❌ Ingresa un número válido")
    
    def load_model_components(self, model_info: Dict) -> bool:
        """📂 Cargar modelo y componentes"""
        
        try:
            print(f"📂 Cargando modelo {model_info['name']}...")
            
            self.model_path = model_info['path']
            self.symbol = model_info['symbol']
            self.timeframe = model_info['timeframe']
            
            # Cargar modelo
            model_file_path = os.path.join(self.model_path, model_info['model_file'])
            self.model = tf.keras.models.load_model(model_file_path)
            
            # 🔧 AUTO-DETECTAR LOOKBACK_WINDOW del modelo
            input_shape = self.model.input_shape
            if len(input_shape) >= 2:
                detected_lookback = input_shape[1]  # (None, timesteps, features)
                if detected_lookback != self.lookback_window:
                    print(f"🔧 Auto-ajustando lookback_window: {self.lookback_window} → {detected_lookback}")
                    self.lookback_window = detected_lookback
            
            print(f"✅ {model_info['model_file']} cargado ({self.model.count_params():,} parámetros)")
            print(f"🔢 Input shape: {input_shape}")
            print(f"⏰ Lookback window: {self.lookback_window} timesteps")
            
            # Cargar scaler
            with open(os.path.join(self.model_path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler cargado")
            
            # Cargar feature columns
            with open(os.path.join(self.model_path, 'feature_columns.pkl'), 'rb') as f:
                self.feature_columns = pickle.load(f)
            print(f"✅ Feature columns cargadas: {len(self.feature_columns)} features")
            
            # Cargar class weights (opcional)
            try:
                with open(os.path.join(self.model_path, 'class_weights.pkl'), 'rb') as f:
                    self.class_weights = pickle.load(f)
                print("✅ Class weights cargados")
            except:
                print("⚠️ Class weights no encontrados (opcional)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    async def get_historical_data(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """📊 Obtener datos históricos del símbolo del modelo"""
        
        print(f"📊 Obteniendo {days} días de datos históricos de {self.symbol}...")
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        base_url = "https://api.binance.com"
        
        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': self.symbol,
                'interval': self.timeframe,
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
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
        
        print(f"✅ Obtenidos {len(df)} registros históricos")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear features usando motor centralizado"""
        
        print("🔧 Calculando features usando MOTOR CENTRALIZADO...")
        
        try:
            # Usar el motor centralizado
            features_df = self.features_engine.calculate_features(
                df=df, 
                feature_set='tcn_definitivo'
            )
            
            # Extraer solo las features del modelo
            if self.feature_columns:
                available_features = [f for f in self.feature_columns if f in features_df.columns]
                missing_features = [f for f in self.feature_columns if f not in features_df.columns]
                
                if missing_features:
                    print(f"⚠️ Features faltantes: {len(missing_features)}")
                    # Agregar features faltantes con valor 0
                    for feature in missing_features:
                        features_df[feature] = 0.0
                
                # Seleccionar exactamente las features del modelo
                features_final = features_df[self.feature_columns].copy()
            else:
                # Usar todas las features TCN definitivo
                tcn_features = self.features_engine.feature_sets['tcn_definitivo']
                features_final = features_df[tcn_features].copy()
            
            print(f"✅ {len(features_final.columns)} features calculadas")
            return features_final
            
        except Exception as e:
            print(f"❌ Error en motor centralizado: {e}")
            # Fallback básico
            features_fallback = pd.DataFrame(index=df.index)
            if self.feature_columns:
                for col in self.feature_columns:
                    features_fallback[col] = 0.0
            else:
                # Crear 66 features básicas
                for i in range(66):
                    features_fallback[f'feature_{i}'] = 0.0
            
            print(f"⚠️ Usando {len(features_fallback.columns)} features de fallback")
            return features_fallback
    
    def generate_predictions(self, df: pd.DataFrame, features: pd.DataFrame, confidence_threshold: float = 0.6) -> List[Dict]:
        """🎯 Generar predicciones para backtesting"""
        
        print(f"🎯 Generando predicciones (threshold: {confidence_threshold:.0%})...")
        
        # Usar el lookback_window auto-detectado del modelo
        lookback_window = self.lookback_window
        
        # Normalizar features
        features_scaled = self.scaler.transform(features)
        
        predictions = []
        
        for i in range(lookback_window, len(features_scaled)):
            # Crear secuencia temporal
            sequence = features_scaled[i-lookback_window:i]
            sequence = sequence.reshape(1, lookback_window, -1)
            
            # Predicción
            pred_proba = self.model.predict(sequence, verbose=0)[0]
            pred_class = np.argmax(pred_proba)
            confidence = float(np.max(pred_proba))
            
            # Aplicar threshold de confianza
            if confidence < confidence_threshold:
                pred_class = 1  # HOLD si baja confianza
            
            # Información de la predicción
            prediction_info = {
                'timestamp': df.index[i],
                'price': df['close'].iloc[i],
                'prediction': pred_class,  # 0=SELL, 1=HOLD, 2=BUY
                'probabilities': {
                    'SELL': float(pred_proba[0]),
                    'HOLD': float(pred_proba[1]),
                    'BUY': float(pred_proba[2])
                },
                'confidence': confidence,
                'filtered': confidence < confidence_threshold
            }
            
            predictions.append(prediction_info)
        
        # Estadísticas
        total_predictions = len(predictions)
        filtered_count = sum(1 for p in predictions if p['filtered'])
        
        print(f"✅ {total_predictions} predicciones generadas")
        print(f"   📊 Filtradas por baja confianza: {filtered_count} ({filtered_count/total_predictions*100:.1f}%)")
        
        return predictions
    
    def simulate_trading(self, df: pd.DataFrame, predictions: List[Dict]) -> Dict:
        """💰 Simular trading basado en predicciones - VERSIÓN CORREGIDA"""
        
        print("💰 Simulando estrategia de trading...")
        
        cash_balance = self.initial_balance  # Dinero en efectivo
        position_size = 0.0  # Cantidad de activo que tengo
        position_entry_price = 0.0
        
        trades = []
        balance_history = []
        
        for i, pred in enumerate(predictions):
            current_price = pred['price']
            signal = pred['prediction']
            confidence = pred['confidence']
            timestamp = pred['timestamp']
            
            # Lógica de trading
            if position_size == 0:  # Sin posición
                if signal == 2 and cash_balance >= self.min_trade_amount:  # BUY signal
                    # ✅ CORREGIDO: Calcular compra correctamente
                    invest_amount = cash_balance * 0.95  # 95% del efectivo
                    entry_fee = invest_amount * self.trading_fee
                    net_invest = invest_amount - entry_fee  # Dinero neto para comprar
                    
                    position_size = net_invest / current_price  # Cantidad de activo comprado
                    position_entry_price = current_price
                    cash_balance *= 0.05  # Solo queda 5% de efectivo
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'OPEN_LONG',
                        'price': current_price,
                        'size': position_size,
                        'fee': entry_fee,
                        'invest_amount': invest_amount,
                        'confidence': confidence,
                        'cash_after': cash_balance
                    })
            
            else:  # Con posición
                # Cerrar posición en señal contraria o HOLD con alta confianza
                should_close = (
                    signal == 0 or  # SELL signal
                    (signal == 1 and confidence > 0.7)  # HOLD con alta confianza
                )
                
                if should_close:
                    # ✅ CORREGIDO: Calcular venta correctamente
                    sell_value = position_size * current_price  # Valor bruto de venta
                    exit_fee = sell_value * self.trading_fee
                    net_proceeds = sell_value - exit_fee  # Dinero neto recibido
                    
                    # Calcular profit del trade
                    entry_cost = 0
                    for trade in reversed(trades):
                        if trade['action'] == 'OPEN_LONG' and 'invest_amount' in trade:
                            entry_cost = trade['invest_amount']
                            break
                    trade_profit = net_proceeds - entry_cost
                    profit_pct = (current_price - position_entry_price) / position_entry_price * 100
                    
                    # Actualizar balances
                    cash_balance += net_proceeds
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'size': position_size,
                        'fee': exit_fee,
                        'sell_value': sell_value,
                        'net_proceeds': net_proceeds,
                        'profit': trade_profit,
                        'profit_pct': profit_pct,
                        'confidence': confidence,
                        'cash_after': cash_balance
                    })
                    
                    # Reset posición
                    position_size = 0.0
                    position_entry_price = 0.0
            
            # Calcular valor total del portfolio
            if position_size > 0:
                portfolio_value = cash_balance + (position_size * current_price)
            else:
                portfolio_value = cash_balance
            
            balance_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash_balance': cash_balance,
                'position_size': position_size,
                'price': current_price
            })
        
        # Cerrar posición final si está abierta
        final_balance = cash_balance
        if position_size > 0 and len(predictions) > 0:
            final_price = predictions[-1]['price']
            sell_value = position_size * final_price
            exit_fee = sell_value * self.trading_fee
            net_proceeds = sell_value - exit_fee
            
            # Calcular profit del trade final
            entry_cost = 0
            for trade in reversed(trades):
                if trade['action'] == 'OPEN_LONG' and 'invest_amount' in trade:
                    entry_cost = trade['invest_amount']
                    break
            trade_profit = net_proceeds - entry_cost
            profit_pct = (final_price - position_entry_price) / position_entry_price * 100
            
            final_balance = cash_balance + net_proceeds
            
            trades.append({
                'timestamp': predictions[-1]['timestamp'],
                'action': 'CLOSE_FINAL',
                'price': final_price,
                'size': position_size,
                'fee': exit_fee,
                'sell_value': sell_value,
                'net_proceeds': net_proceeds,
                'profit': trade_profit,
                'profit_pct': profit_pct,
                'cash_after': final_balance
            })
        
        self.trades = trades
        self.balance_history = balance_history
        self.predictions_history = predictions
        
        completed_trades = [t for t in trades if t['action'] in ['CLOSE_LONG', 'CLOSE_FINAL']]
        print(f"✅ Simulación completada: {len(completed_trades)} trades completados")
        
        return {
            'final_balance': final_balance,
            'total_trades': len(trades),
            'balance_history': balance_history,
            'trades': trades
        }
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """📊 Calcular métricas de rendimiento"""
        
        print("📊 Calculando métricas de rendimiento...")
        
        trades = results['trades']
        balance_history = results['balance_history']
        final_balance = results['final_balance']
        
        if not trades:
            return {'error': 'No hay trades para analizar'}
        
        # ✅ MÉTRICAS CORREGIDAS
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        
        # Solo trades de cierre (que tienen profit calculado)
        completed_trades = [t for t in trades if t['action'] in ['CLOSE_LONG', 'CLOSE_FINAL'] and 'profit' in t]
        total_trades = len(completed_trades)
        
        if total_trades == 0:
            return {'error': 'No hay trades completados para analizar'}
        
        # Separar trades ganadores y perdedores
        profitable_trades = [t for t in completed_trades if t['profit'] > 0]
        losing_trades = [t for t in completed_trades if t['profit'] <= 0]
        
        win_rate = len(profitable_trades) / total_trades * 100
        
        # Calcular profit/loss totales
        total_profit = sum([t['profit'] for t in profitable_trades])
        total_loss = sum([t['profit'] for t in losing_trades])  # Ya es negativo
        
        # Promedios por trade
        avg_profit = total_profit / len(profitable_trades) if profitable_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        # Profit factor: ganancia total / pérdida total (en valor absoluto)
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
        
        # Drawdown
        portfolio_values = [b['portfolio_value'] for b in balance_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        max_drawdown = np.max(drawdown)
        
        # Sharpe ratio (simplificado)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252*288) if np.std(returns) > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'final_balance': final_balance,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'total_profit': total_profit,
            'total_loss': total_loss,
            'avg_profit_per_trade': avg_profit,
            'avg_loss_per_trade': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
        }
    
    def print_results(self, metrics: Dict, model_info: Dict):
        """📈 Mostrar resultados del backtesting"""
        
        print("\n" + "="*70)
        print(f"📈 RESULTADOS DEL BACKTESTING - {model_info['symbol']}")
        print(f"🏗️ Modelo: {model_info['name']}")
        print("="*70)
        
        if 'error' in metrics:
            print(f"❌ {metrics['error']}")
            return
        
        print(f"💰 RENDIMIENTO GENERAL:")
        print(f"   Balance inicial: ${self.initial_balance:,.2f}")
        print(f"   Balance final: ${metrics['final_balance']:,.2f}")
        print(f"   Retorno total: {metrics['total_return_pct']:+.2f}%")
        print(f"   Máximo drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\n📊 ESTADÍSTICAS DE TRADING:")
        print(f"   Total trades: {metrics['total_trades']}")
        print(f"   Win rate: {metrics['win_rate_pct']:.1f}%")
        print(f"   Trades ganadores: {metrics['profitable_trades']}")
        print(f"   Trades perdedores: {metrics['losing_trades']}")
        
        print(f"\n💵 PROFIT & LOSS (CORREGIDO):")
        print(f"   Profit total: ${metrics['total_profit']:+.2f}")
        print(f"   Loss total: ${metrics['total_loss']:+.2f}")
        print(f"   Avg profit/trade: ${metrics['avg_profit_per_trade']:+.2f}")
        print(f"   Avg loss/trade: ${metrics['avg_loss_per_trade']:+.2f}")
        print(f"   Profit factor: {metrics['profit_factor']:.2f}")
        
        # ✅ VERIFICACIÓN DE CÁLCULOS
        total_pnl = metrics['total_profit'] + metrics['total_loss']
        actual_pnl = metrics['final_balance'] - self.initial_balance
        print(f"\n🔍 VERIFICACIÓN:")
        print(f"   PnL por trades: ${total_pnl:+.2f}")
        print(f"   PnL real: ${actual_pnl:+.2f}")
        if abs(total_pnl - actual_pnl) < 1.0:
            print(f"   ✅ Cálculos correctos")
        else:
            print(f"   ⚠️ Diferencia: ${abs(total_pnl - actual_pnl):.2f}")
        
        print(f"\n📈 MÉTRICAS AVANZADAS:")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        # Evaluación del rendimiento
        print(f"\n🎯 EVALUACIÓN:")
        if metrics['total_return_pct'] > 10:
            print(f"   ✅ EXCELENTE: Retorno > 10%")
        elif metrics['total_return_pct'] > 5:
            print(f"   ✅ BUENO: Retorno > 5%")
        elif metrics['total_return_pct'] > 0:
            print(f"   ⚠️ MARGINAL: Retorno positivo pero bajo")
        else:
            print(f"   ❌ PÉRDIDAS: Retorno negativo")
        
        if metrics['win_rate_pct'] > 60:
            print(f"   ✅ WIN RATE EXCELENTE: > 60%")
        elif metrics['win_rate_pct'] > 50:
            print(f"   ✅ WIN RATE BUENO: > 50%")
        else:
            print(f"   ⚠️ WIN RATE BAJO: < 50%")
        
        if metrics['profit_factor'] > 1.5:
            print(f"   ✅ PROFIT FACTOR EXCELENTE: > 1.5")
        elif metrics['profit_factor'] > 1.0:
            print(f"   ✅ PROFIT FACTOR POSITIVO: > 1.0")
        else:
            print(f"   ❌ PROFIT FACTOR NEGATIVO: < 1.0")
        
        print("="*70)
    
    async def run_backtest(self, model_info: Dict, days: int = 15, confidence_threshold: float = 0.5):
        """🚀 Ejecutar backtesting completo"""
        
        print(f"🚀 INICIANDO BACKTESTING UNIVERSAL")
        print(f"📊 Modelo: {model_info['name']}")
        print(f"💎 Símbolo: {model_info['symbol']}")
        print(f"⏰ Timeframe: {model_info['timeframe']}")
        print(f"📅 Días: {days}")
        print(f"🎯 Confianza mínima: {confidence_threshold:.0%}")
        print("="*70)
        
        # 1. Cargar modelo
        if not self.load_model_components(model_info):
            return None
        
        # 2. Obtener datos históricos
        df = await self.get_historical_data(days=days)
        if df.empty:
            print("❌ No se pudieron obtener datos históricos")
            return None
        
        # 3. Calcular features
        features = self.create_features(df)
        if features.empty:
            print("❌ Error calculando features")
            return None
        
        # 4. Generar predicciones
        predictions = self.generate_predictions(df, features, confidence_threshold)
        if not predictions:
            print("❌ Error generando predicciones")
            return None
        
        # 5. Simular trading
        results = self.simulate_trading(df, predictions)
        
        # 6. Calcular métricas
        metrics = self.calculate_metrics(results)
        
        # 7. Mostrar resultados
        self.print_results(metrics, model_info)
        
        return {
            'metrics': metrics,
            'results': results,
            'predictions': predictions,
            'data': df,
            'model_info': model_info
        }

async def main():
    """🎯 Función principal de backtesting universal"""
    
    print("🚀 BACKTESTING UNIVERSAL - SELECTOR DE MODELOS")
    print("=" * 70)
    print("🎯 Probar cualquier modelo disponible en el directorio models/")
    print("=" * 70)
    
    # Crear backtester
    backtester = UniversalBacktester()
    
    # Descubrir modelos
    models = backtester.discover_models()
    if not models:
        print("❌ No se encontraron modelos válidos")
        return
    
    # Seleccionar modelo
    selected_model = backtester.select_model(models)
    if not selected_model:
        print("👋 ¡Hasta luego!")
        return
    
    # Configuración del backtest
    print(f"\n⚙️ CONFIGURACIÓN DEL BACKTEST:")
    print("1. 🔥 Backtest rápido (7 días)")
    print("2. 📊 Backtest estándar (15 días)")
    print("3. 📈 Backtest extenso (30 días)")
    print("4. ⚙️ Configuración personalizada")
    
    while True:
        choice = input("\n🎯 Selecciona opción (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            break
        print("❌ Selecciona 1, 2, 3 o 4")
    
    if choice == "1":
        days = 7
        confidence_threshold = 0.5
    elif choice == "2":
        days = 15
        confidence_threshold = 0.5
    elif choice == "3":
        days = 30
        confidence_threshold = 0.5
    else:  # Personalizada
        while True:
            try:
                days = int(input("📅 Días de backtesting (5-60): "))
                if 5 <= days <= 60:
                    break
                print("❌ Días debe estar entre 5 y 60")
            except ValueError:
                print("❌ Ingresa un número válido")
        
        while True:
            try:
                confidence_threshold = float(input("🎯 Confianza mínima (0.3-0.8): "))
                if 0.3 <= confidence_threshold <= 0.8:
                    break
                print("❌ Confianza debe estar entre 0.3 y 0.8")
            except ValueError:
                print("❌ Ingresa un número válido")
    
    # Ejecutar backtest
    try:
        results = await backtester.run_backtest(
            model_info=selected_model,
            days=days,
            confidence_threshold=confidence_threshold
        )
        
        if results:
            print(f"\n🎉 ¡BACKTESTING COMPLETADO EXITOSAMENTE!")
            print(f"📊 Datos procesados: {len(results['data'])} velas")
            print(f"🎯 Predicciones: {len(results['predictions'])}")
            if 'metrics' in results and 'total_trades' in results['metrics']:
                print(f"💰 Trades simulados: {results['metrics']['total_trades']}")
            else:
                print(f"💰 Trades simulados: 0")
            
            # Guardar resultados (opcional)
            save_results = input("\n💾 ¿Guardar resultados detallados? (s/n): ").lower().strip()
            if save_results in ['s', 'si', 'yes', 'y']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backtest_{selected_model['symbol'].lower()}_{timestamp}.json"
                
                import json
                with open(filename, 'w') as f:
                    json_results = {
                        'model_info': selected_model,
                        'metrics': results['metrics'],
                        'config': {
                            'days': days,
                            'confidence_threshold': confidence_threshold,
                            'initial_balance': backtester.initial_balance,
                            'trading_fee': backtester.trading_fee
                        }
                    }
                    json.dump(json_results, f, indent=2, default=str)
                
                print(f"💾 Resultados guardados en: {filename}")
            
        else:
            print("\n❌ Error en el backtesting")
            
    except Exception as e:
        print(f"\n❌ Error ejecutando backtesting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 