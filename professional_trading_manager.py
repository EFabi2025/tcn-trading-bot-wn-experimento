#!/usr/bin/env python3
"""
🚀 PROFESSIONAL TRADING MANAGER
Sistema completo de trading algorítmico con todas las funcionalidades profesionales
"""

import asyncio
import aiohttp
import time
import hmac
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Importar nuestros módulos
from advanced_risk_manager import AdvancedRiskManager, Position, RiskLimits
from trading_database import TradingDatabase
from final_real_binance_predictor import OptimizedTCNPredictor, OptimizedBinanceData

load_dotenv()

@dataclass
class BinanceConfig:
    """⚙️ Configuración de Binance"""
    api_key: str
    secret_key: str
    base_url: str
    environment: str

class TradingManagerStatus:
    """📊 Estados del Trading Manager"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class ProfessionalTradingManager:
    """🚀 Trading Manager Profesional Completo"""

    def __init__(self):
        self.status = TradingManagerStatus.STOPPED
        self.start_time = None
        self.last_heartbeat = None

        # Configuración
        self.config = self._load_config()

        # Componentes principales
        self.risk_manager = None
        self.database = None
        self.predictor = None
        self.binance_data = None

        # Estado del sistema
        self.active_positions = {}
        self.trade_count = 0
        self.session_pnl = 0.0
        self.last_prediction_time = None
        self.emergency_mode = False

        # Configuración de trading
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
        self.prediction_interval = 60  # segundos
        self.monitoring_interval = 30  # segundos

        # Métricas en tiempo real
        self.metrics = {
            'uptime_seconds': 0,
            'total_predictions': 0,
            'successful_predictions': 0,
            'api_calls_count': 0,
            'error_count': 0,
            'last_error': None
        }

        # Control de pausa/resume
        self.pause_trading = False
        self.pause_reason = None

        print("🚀 Professional Trading Manager inicializado")

    def _load_config(self) -> BinanceConfig:
        """⚙️ Cargar configuración desde variables de entorno"""
        return BinanceConfig(
            api_key=os.getenv('BINANCE_API_KEY'),
            secret_key=os.getenv('BINANCE_SECRET_KEY'),
            base_url=os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision'),
            environment=os.getenv('ENVIRONMENT', 'testnet')
        )

    async def initialize(self):
        """🚀 Inicializar todos los componentes del sistema"""
        print("🚀 Iniciando Professional Trading Manager...")
        self.status = TradingManagerStatus.STARTING

        try:
            # 1. Inicializar base de datos
            await self._initialize_database()

            # 2. Inicializar Risk Manager
            await self._initialize_risk_manager()

            # 3. Inicializar predictor TCN
            await self._initialize_predictor()

            # 4. Inicializar datos de Binance
            await self._initialize_binance_data()

            # 5. Verificar conectividad
            await self._verify_connectivity()

            # 6. Configurar monitoreo
            await self._setup_monitoring()

            self.start_time = datetime.now()
            self.last_heartbeat = datetime.now()
            self.status = TradingManagerStatus.RUNNING

            # Log inicial
            await self.database.log_event('INFO', 'SYSTEM', 'Professional Trading Manager inicializado correctamente')

            print("✅ Professional Trading Manager iniciado correctamente")

        except Exception as e:
            self.status = TradingManagerStatus.ERROR
            print(f"❌ Error inicializando Trading Manager: {e}")
            await self.database.log_event('ERROR', 'SYSTEM', f'Error inicializando: {e}')
            raise

    async def _initialize_database(self):
        """🗄️ Inicializar sistema de base de datos"""
        print("🗄️ Inicializando base de datos...")
        self.database = TradingDatabase()

        # Limpiar datos antiguos si es necesario
        await self.database.cleanup_old_data(days_to_keep=90)

        print("✅ Base de datos lista")

    async def _initialize_risk_manager(self):
        """🛡️ Inicializar Risk Manager"""
        print("🛡️ Inicializando Risk Manager...")
        self.risk_manager = AdvancedRiskManager(self.config)
        await self.risk_manager.initialize()

        print("✅ Risk Manager configurado")

    async def _initialize_predictor(self):
        """🤖 Inicializar predictor TCN"""
        print("🤖 Inicializando predictor TCN...")
        self.predictor = OptimizedTCNPredictor()

        print("✅ Predictor TCN listo")

    async def _initialize_binance_data(self):
        """📊 Inicializar datos de Binance"""
        print("📊 Inicializando datos de Binance...")
        self.binance_data = OptimizedBinanceData(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key,
            base_url=self.config.base_url
        )

        print("✅ Conexión a Binance configurada")

    async def _verify_connectivity(self):
        """🔗 Verificar conectividad con APIs"""
        print("🔗 Verificando conectividad...")

        # Test Binance API
        try:
            test_data = await self.binance_data.get_klines('BTCUSDT', '1m', 5)
            if test_data.empty:
                raise Exception("No se pudieron obtener datos de test")
            print("✅ Conectividad Binance OK")
        except Exception as e:
            raise Exception(f"Error conectividad Binance: {e}")

        # Test predictor
        try:
            # Test con datos simulados
            test_features = np.random.random((1, 50, 21))
            test_prediction = self.predictor.predict(test_features)
            print("✅ Predictor TCN OK")
        except Exception as e:
            raise Exception(f"Error predictor: {e}")

    async def _setup_monitoring(self):
        """👁️ Configurar sistema de monitoreo"""
        print("👁️ Configurando monitoreo...")

        # Crear tareas de monitoreo en background
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._position_monitor())
        asyncio.create_task(self._metrics_collector())

        print("✅ Monitoreo configurado")

    async def run(self):
        """🎯 Ejecutar loop principal de trading"""
        print("🎯 Iniciando loop principal de trading...")

        while self.status == TradingManagerStatus.RUNNING:
            try:
                # Verificar si está pausado
                if self.pause_trading:
                    await self._handle_pause_state()
                    await asyncio.sleep(10)
                    continue

                # 1. Obtener predicciones para todos los símbolos
                predictions = await self._get_all_predictions()

                # 2. Procesar cada predicción
                for symbol, prediction_data in predictions.items():
                    await self._process_prediction(symbol, prediction_data)

                # 3. Actualizar métricas
                await self._update_metrics()

                # 4. Guardar estado en DB
                await self._save_periodic_metrics()

                # 5. Esperar siguiente ciclo
                await asyncio.sleep(self.prediction_interval)

            except Exception as e:
                await self._handle_error(e)
                await asyncio.sleep(30)  # Pausa en caso de error

    async def _get_all_predictions(self) -> Dict:
        """🔮 Obtener predicciones para todos los símbolos"""
        predictions = {}

        for symbol in self.symbols:
            try:
                # Obtener datos de mercado
                df = await self.binance_data.get_klines(symbol, '1m', 500)

                if len(df) < 500:
                    print(f"⚠️ Datos insuficientes para {symbol}: {len(df)} velas")
                    continue

                # Crear features técnicos
                features = self.binance_data.create_technical_features(df)

                if features is None or len(features) < 50:
                    print(f"⚠️ Features insuficientes para {symbol}")
                    continue

                # Hacer predicción
                prediction_result = self.predictor.predict(features)

                # Obtener precio actual
                current_price = float(df.iloc[-1]['close'])

                predictions[symbol] = {
                    'signal': prediction_result['signal'],
                    'confidence': prediction_result['confidence'],
                    'current_price': current_price,
                    'timestamp': datetime.now(),
                    'features_used': features.shape if hasattr(features, 'shape') else len(features)
                }

                self.metrics['total_predictions'] += 1

                print(f"🔮 {symbol}: {prediction_result['signal']} ({prediction_result['confidence']:.1%})")

            except Exception as e:
                print(f"❌ Error predicción {symbol}: {e}")
                self.metrics['error_count'] += 1
                await self.database.log_event('ERROR', 'PREDICTION', f'Error predicción {symbol}: {e}', symbol)

        self.last_prediction_time = datetime.now()
        return predictions

    async def _process_prediction(self, symbol: str, prediction_data: Dict):
        """⚡ Procesar una predicción individual"""

        signal = prediction_data['signal']
        confidence = prediction_data['confidence']
        current_price = prediction_data['current_price']

        # Skip si es HOLD o baja confianza
        if signal == 'HOLD' or confidence < 0.70:
            return

        # Verificar si ya tenemos posición en este símbolo
        if symbol in self.active_positions:
            await self._manage_existing_position(symbol, prediction_data)
        else:
            await self._consider_new_position(symbol, prediction_data)

    async def _consider_new_position(self, symbol: str, prediction_data: Dict):
        """📈 Considerar nueva posición"""

        signal = prediction_data['signal']
        confidence = prediction_data['confidence']
        current_price = prediction_data['current_price']

        # Verificar límites de riesgo
        can_trade, reason = await self.risk_manager.check_risk_limits_before_trade(
            symbol, signal, confidence
        )

        if not can_trade:
            print(f"❌ Trade rechazado {symbol}: {reason}")
            await self.database.log_event('WARNING', 'RISK', f'Trade rechazado {symbol}: {reason}', symbol)
            return

        # Abrir nueva posición
        position = await self.risk_manager.open_position(symbol, signal, confidence, current_price)

        if position:
            self.active_positions[symbol] = position

            # Guardar en base de datos
            trade_data = {
                'symbol': symbol,
                'side': signal,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'confidence': confidence,
                'strategy': 'TCN_ML',
                'is_active': True,
                'metadata': {
                    'features_used': prediction_data.get('features_used'),
                    'prediction_time': prediction_data['timestamp'].isoformat()
                }
            }

            trade_id = await self.database.save_trade(trade_data)
            position.trade_id = trade_id  # Guardar ID para updates posteriores

            self.trade_count += 1

            # Log del trade
            await self.database.log_event(
                'INFO', 'TRADING',
                f'Nueva posición: {signal} {symbol} @ ${current_price:.4f}',
                symbol
            )

            # Enviar notificación Discord si está configurado
            await self._send_discord_notification(f"🟢 **NUEVA POSICIÓN**\n"
                                                 f"📊 {symbol}: {signal}\n"
                                                 f"💰 Precio: ${current_price:.4f}\n"
                                                 f"🎯 Confianza: {confidence:.1%}\n"
                                                 f"📈 Cantidad: {position.quantity:.6f}")

    async def _manage_existing_position(self, symbol: str, prediction_data: Dict):
        """🔄 Gestionar posición existente"""

        position = self.active_positions[symbol]
        current_price = prediction_data['current_price']

        # Actualizar precio actual
        position.current_price = current_price

        # Calcular PnL actual
        if position.side == 'BUY':
            position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

        position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)

        # Verificar Stop Loss/Take Profit (esto lo hace el risk manager en monitor_positions)
        # Aquí podríamos implementar lógica adicional basada en señales

        # Si la señal cambió drásticamente, considerar cierre
        signal = prediction_data['signal']
        confidence = prediction_data['confidence']

        # Ejemplo: Si teníamos BUY y ahora es SELL con alta confianza, cerrar
        if ((position.side == 'BUY' and signal == 'SELL') or
            (position.side == 'SELL' and signal == 'BUY')) and confidence > 0.80:

            await self._close_position(symbol, "SIGNAL_REVERSAL")

    async def _close_position(self, symbol: str, reason: str):
        """📉 Cerrar posición específica"""

        if symbol not in self.active_positions:
            return

        position = self.active_positions[symbol]
        current_price = await self.risk_manager.get_current_price(symbol)

        # Calcular PnL final
        if position.side == 'BUY':
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

        pnl_usd = (pnl_percent / 100) * (position.quantity * position.entry_price)

        # Actualizar estadísticas del risk manager
        self.session_pnl += pnl_usd

        # Actualizar en base de datos
        if hasattr(position, 'trade_id') and position.trade_id:
            exit_data = {
                'exit_price': current_price,
                'exit_time': datetime.now(),
                'pnl_percent': pnl_percent,
                'pnl_usd': pnl_usd,
                'exit_reason': reason
            }

            await self.database.update_trade_exit(position.trade_id, exit_data)

        # Eliminar de posiciones activas
        del self.active_positions[symbol]

        # Log del cierre
        emoji = "🟢" if pnl_usd > 0 else "🔴"
        await self.database.log_event(
            'INFO', 'TRADING',
            f'Posición cerrada: {symbol} - PnL: {pnl_percent:+.2f}% (${pnl_usd:+.2f})',
            symbol
        )

        # Notificación Discord
        await self._send_discord_notification(f"{emoji} **POSICIÓN CERRADA**\n"
                                             f"📊 {symbol}\n"
                                             f"💰 PnL: {pnl_percent:+.2f}% (${pnl_usd:+.2f})\n"
                                             f"📝 Razón: {reason}")

        print(f"{emoji} Posición {symbol} cerrada: {pnl_percent:+.2f}% (${pnl_usd:+.2f})")

    async def _heartbeat_monitor(self):
        """💓 Monitor de heartbeat del sistema"""
        while self.status in [TradingManagerStatus.RUNNING, TradingManagerStatus.PAUSED]:
            try:
                self.last_heartbeat = datetime.now()

                if self.start_time:
                    self.metrics['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()

                await asyncio.sleep(30)

            except Exception as e:
                print(f"❌ Error en heartbeat: {e}")
                await asyncio.sleep(60)

    async def _position_monitor(self):
        """👁️ Monitor de posiciones para Stop Loss/Take Profit"""
        while self.status in [TradingManagerStatus.RUNNING, TradingManagerStatus.PAUSED]:
            try:
                if not self.pause_trading:
                    await self.risk_manager.monitor_positions()

                    # Sincronizar posiciones con nuestro dict
                    for symbol in list(self.active_positions.keys()):
                        if symbol not in self.risk_manager.active_positions:
                            # La posición fue cerrada por el risk manager
                            del self.active_positions[symbol]

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                print(f"❌ Error monitoreando posiciones: {e}")
                await asyncio.sleep(60)

    async def _metrics_collector(self):
        """📊 Collector de métricas en tiempo real"""
        while self.status in [TradingManagerStatus.RUNNING, TradingManagerStatus.PAUSED]:
            try:
                # Actualizar métricas cada 5 minutos
                await asyncio.sleep(300)
                await self._save_periodic_metrics()

            except Exception as e:
                print(f"❌ Error recolectando métricas: {e}")
                await asyncio.sleep(300)

    async def _save_periodic_metrics(self):
        """💾 Guardar métricas periódicamente"""
        try:
            # Obtener métricas del risk manager
            risk_metrics = self.risk_manager.get_risk_report()

            # Calcular métricas adicionales
            total_trades = await self._get_total_trades_today()
            win_rate = await self._calculate_win_rate()

            metrics_data = {
                'total_balance': risk_metrics['balance']['current'],
                'daily_pnl': risk_metrics['pnl']['daily_usd'],
                'total_pnl': risk_metrics['pnl']['total_usd'],
                'daily_return_percent': risk_metrics['pnl']['daily_percent'],
                'total_return_percent': risk_metrics['pnl']['total_percent'],
                'current_drawdown': risk_metrics['risk_metrics']['current_drawdown'],
                'max_drawdown': 0.0,  # Calcular histórico
                'win_rate': win_rate,
                'profit_factor': 0.0,  # Calcular
                'active_positions_count': len(self.active_positions),
                'total_exposure_usd': risk_metrics['risk_metrics']['exposure_usd'],
                'exposure_percent': risk_metrics['risk_metrics']['exposure_percent'],
                'trades_today': total_trades
            }

            await self.database.save_performance_metrics(metrics_data)

        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")

    async def _get_total_trades_today(self) -> int:
        """📊 Obtener total de trades del día"""
        try:
            trades = await self.database.get_trades_history(days=1)
            return len([t for t in trades if t['entry_time'].startswith(datetime.now().strftime('%Y-%m-%d'))])
        except:
            return 0

    async def _calculate_win_rate(self) -> float:
        """📈 Calcular win rate reciente"""
        try:
            trades = await self.database.get_trades_history(days=7, is_active=False)
            if not trades:
                return 0.0

            winning_trades = len([t for t in trades if t['pnl_usd'] and t['pnl_usd'] > 0])
            return (winning_trades / len(trades)) * 100 if trades else 0.0
        except:
            return 0.0

    async def _handle_pause_state(self):
        """⏸️ Manejar estado de pausa"""
        print(f"⏸️ Sistema pausado: {self.pause_reason}")

        # Continuar monitoreando posiciones incluso en pausa
        await self.risk_manager.monitor_positions()

    async def _handle_error(self, error: Exception):
        """❌ Manejar errores del sistema"""
        self.metrics['error_count'] += 1
        self.metrics['last_error'] = str(error)

        print(f"❌ Error en Trading Manager: {error}")

        # Log crítico
        await self.database.log_event('ERROR', 'SYSTEM', f'Error crítico: {error}')

        # Si es error crítico, activar pausa
        if "critical" in str(error).lower() or self.metrics['error_count'] > 10:
            await self.pause_trading_with_reason(f"Errores críticos: {self.metrics['error_count']}")

    async def _send_discord_notification(self, message: str):
        """📢 Enviar notificación a Discord"""
        try:
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if not webhook_url:
                return

            payload = {
                'content': f"🤖 **Trading Bot**\n{message}\n📅 {datetime.now().strftime('%H:%M:%S')}"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 204:
                        print(f"⚠️ Error enviando notificación Discord: {response.status}")

        except Exception as e:
            print(f"❌ Error Discord: {e}")

    # === MÉTODOS DE CONTROL PÚBLICO ===

    async def pause_trading_with_reason(self, reason: str):
        """⏸️ Pausar trading con razón específica"""
        self.pause_trading = True
        self.pause_reason = reason
        self.status = TradingManagerStatus.PAUSED

        await self.database.log_event('WARNING', 'CONTROL', f'Trading pausado: {reason}')
        await self._send_discord_notification(f"⏸️ **TRADING PAUSADO**\n📝 {reason}")

        print(f"⏸️ Trading pausado: {reason}")

    async def resume_trading(self):
        """▶️ Reanudar trading"""
        self.pause_trading = False
        self.pause_reason = None
        self.status = TradingManagerStatus.RUNNING

        await self.database.log_event('INFO', 'CONTROL', 'Trading reanudado')
        await self._send_discord_notification("▶️ **TRADING REANUDADO**")

        print("▶️ Trading reanudado")

    async def emergency_stop(self):
        """🚨 Parada de emergencia - cerrar todas las posiciones"""
        self.status = TradingManagerStatus.EMERGENCY_STOP
        self.emergency_mode = True

        print("🚨 PARADA DE EMERGENCIA ACTIVADA")

        # Cerrar todas las posiciones
        for symbol in list(self.active_positions.keys()):
            await self._close_position(symbol, "EMERGENCY_STOP")

        # Activar circuit breaker
        await self.risk_manager.activate_circuit_breaker("Emergency stop manual", 120)

        await self.database.log_event('CRITICAL', 'EMERGENCY', 'Parada de emergencia activada')
        await self._send_discord_notification("🚨 **PARADA DE EMERGENCIA**\nTodas las posiciones cerradas")

    async def get_system_status(self) -> Dict:
        """📊 Obtener estado completo del sistema"""
        risk_report = self.risk_manager.get_risk_report() if self.risk_manager else {}

        return {
            'status': self.status,
            'uptime_seconds': self.metrics['uptime_seconds'],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'trading_paused': self.pause_trading,
            'pause_reason': self.pause_reason,
            'emergency_mode': self.emergency_mode,
            'active_positions': len(self.active_positions),
            'session_pnl': self.session_pnl,
            'trade_count': self.trade_count,
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'metrics': self.metrics,
            'risk_metrics': risk_report,
            'symbols_trading': self.symbols,
            'prediction_interval': self.prediction_interval,
            'environment': self.config.environment
        }

    async def update_risk_limits(self, new_limits: Dict):
        """⚙️ Actualizar límites de riesgo en caliente"""
        if self.risk_manager:
            for key, value in new_limits.items():
                if hasattr(self.risk_manager.limits, key):
                    setattr(self.risk_manager.limits, key, value)
                    print(f"🔧 Límite actualizado: {key} = {value}")

            await self.database.log_event('INFO', 'CONFIG', f'Límites de riesgo actualizados: {new_limits}')

    async def shutdown(self):
        """🔄 Apagar sistema de forma segura"""
        print("🔄 Iniciando apagado seguro...")

        # Cambiar estado
        self.status = TradingManagerStatus.STOPPED

        # Cerrar posiciones activas si hay alguna
        if self.active_positions:
            print("📉 Cerrando posiciones activas...")
            for symbol in list(self.active_positions.keys()):
                await self._close_position(symbol, "SYSTEM_SHUTDOWN")

        # Crear backup de la base de datos
        if self.database:
            backup_path = await self.database.backup_database()
            if backup_path:
                print(f"💾 Backup creado: {backup_path}")

        # Log final
        await self.database.log_event('INFO', 'SYSTEM', 'Sistema apagado de forma segura')
        await self._send_discord_notification("🔄 **SISTEMA APAGADO**\nApagado seguro completado")

        print("✅ Apagado seguro completado")

# === FUNCIÓN PRINCIPAL ===
async def main():
    """🎯 Función principal para ejecutar el Trading Manager"""
    manager = ProfessionalTradingManager()

    try:
        # Inicializar
        await manager.initialize()

        # Ejecutar
        await manager.run()

    except KeyboardInterrupt:
        print("\n⏹️ Interrupción manual detectada")
        await manager.shutdown()
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        await manager.emergency_stop()
        await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
