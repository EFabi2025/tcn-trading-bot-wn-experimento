#!/usr/bin/env python3
"""
🚀 TRADING MANAGER - EL CEREBRO DEL BOT
Orquesta todos los módulos para ejecutar la estrategia de trading con TCN.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Módulos de configuración y base de datos
from config import trading_config
from trading_database import TradingDatabase

# Módulos de lógica de trading
from advanced_risk_manager import AdvancedRiskManager
from professional_portfolio_manager import ProfessionalPortfolioManager

# Módulos de predicción y datos
from real_binance_predictor import BinanceDataProvider, RealTCNPredictor

# Módulos de utilidad
from smart_discord_notifier import SmartDiscordNotifier

class TradingManagerStatus:
    """📊 Estados del Trading Manager"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"

class TradingManager:
    """🚀 El Orquestador Principal del Bot de Trading"""
    
    def __init__(self):
        """Inicializa el Trading Manager y todos sus componentes."""
        self.config = trading_config
        self.status = TradingManagerStatus.STOPPED
        self.logger = self._setup_logger()

        # Componentes del sistema (se inicializarán después)
        self.database: TradingDatabase = None
        self.data_provider: BinanceDataProvider = None
        self.tcn_predictor: RealTCNPredictor = None
        self.risk_manager: AdvancedRiskManager = None
        self.portfolio_manager: ProfessionalPortfolioManager = None
        self.discord_notifier: SmartDiscordNotifier = None

        self.active_positions: Dict[str, any] = {}
        self.symbols: list[str] = self.config.TRADING_SYMBOLS

    def _setup_logger(self) -> logging.Logger:
        """Configura un logger estandarizado para el sistema."""
        logger = logging.getLogger("TradingManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    async def initialize(self):
        """Inicializa todos los subsistemas en el orden correcto."""
        self.logger.info("🚀 Iniciando el Trading Manager...")
        self.status = TradingManagerStatus.STARTING
        
        try:
            # 1. Base de datos
            self.database = TradingDatabase()
            self.logger.info("✅ Base de datos inicializada.")

            # 2. Proveedor de datos de mercado
            self.data_provider = BinanceDataProvider()
            await self.data_provider.__aenter__() # Inicia la sesión de aiohttp
            self.logger.info("✅ Proveedor de datos de mercado (BinanceDataProvider) listo.")

            # 3. Predictor TCN
            self.tcn_predictor = RealTCNPredictor()
            self.logger.info("✅ Predictor TCN (RealTCNPredictor) cargado con modelos.")

            # 4. Gestor de Portfolio (necesita el balance inicial)
            self.portfolio_manager = ProfessionalPortfolioManager(self.config, self.symbols, self.logger)
            initial_balance = await self.portfolio_manager.initialize()
            self.logger.info(f"✅ Gestor de Portfolio inicializado. Balance USDT inicial: ${initial_balance:.2f}")

            # 5. Gestor de Riesgo
            self.risk_manager = AdvancedRiskManager()
            await self.risk_manager.initialize(initial_balance)
            self.logger.info("✅ Gestor de Riesgo (AdvancedRiskManager) configurado.")
            
            # 6. Notificador de Discord
            self.discord_notifier = SmartDiscordNotifier()
            self.logger.info("✅ Notificador de Discord listo.")

            # 7. Tareas de monitoreo
            self._setup_monitoring()

            self.status = TradingManagerStatus.RUNNING
            self.logger.info("🎉 ¡Sistema inicializado y listo para operar! Estado: RUNNING.")

        except Exception as e:
            self.logger.critical(f"❌ Error fatal durante la inicialización: {e}", exc_info=True)
            self.status = TradingManagerStatus.ERROR
            await self.shutdown()
            raise

    def _setup_monitoring(self):
        """Configura las tareas de monitoreo en segundo plano."""
        self.logger.info("⚙️ Configurando tareas de monitoreo...")
        asyncio.create_task(self._heartbeat_monitor())
        self.logger.info("✅ Tarea de monitoreo de heartbeat configurada.")

    async def _heartbeat_monitor(self):
        """💖 Envía un "latido" periódico para mostrar que el bot está activo."""
        self.logger.info("💖 Monitor de heartbeat iniciado.")
        while self.status == TradingManagerStatus.RUNNING:
            try:
                self.logger.info("💖 Heartbeat: El bot está vivo y operando.")
                await asyncio.sleep(self.config.HEARTBEAT_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                self.logger.info("💖 Monitor de heartbeat detenido.")
                break
            except Exception as e:
                self.logger.error(f"💥 Error en el monitor de heartbeat: {e}")
                await asyncio.sleep(60)

    async def _display_status_report(self):
        """Muestra un reporte de estado completo y profesional en la consola."""
        try:
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            if not snapshot:
                self.logger.warning("No se pudo obtener el snapshot del portafolio para el reporte.")
                return

            report = self.portfolio_manager.format_tcn_style_report(snapshot)
            
            # Limpiar la consola para un reporte limpio (opcional, puede no funcionar en todas las terminales)
            # os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "🔥" * 30 + " REPORTE DE ESTADO " + "🔥" * 30)
            print(report)
            print("🔥" * 79)

        except Exception as e:
            self.logger.error(f"❌ Error generando el reporte de estado: {e}", exc_info=True)

    async def run(self):
        """Bucle principal de trading."""
        if self.status != TradingManagerStatus.RUNNING:
            self.logger.error("El manager no está en estado RUNNING. No se puede iniciar el bucle.")
            return

        self.logger.info("🎯 Iniciando bucle principal de trading...")
        while self.status == TradingManagerStatus.RUNNING:
            try:
                loop_start_time = datetime.now()
                
                # 1. Mostrar el estado actual del portafolio
                await self._display_status_report()

                # 2. Obtener precios actuales
                prices = await self._get_current_prices()

                # 3. Generar señales basadas en TCN
                signals = await self._generate_tcn_signals(prices)

                # 4. Procesar señales
                if signals:
                    await self._process_signals(signals)
                else:
                    self.logger.info("🤔 No se generaron señales de trading en este ciclo.")

                # 5. Esperar al siguiente ciclo
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                sleep_time = max(0, self.config.CHECK_INTERVAL_SECONDS - loop_duration)
                self.logger.info(f"Ciclo completado en {loop_duration:.2f}s. Durmiendo por {sleep_time:.2f}s.")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                self.logger.info("Bucle de trading cancelado.")
                break
            except Exception as e:
                self.logger.error(f"❌ Error en el bucle principal de trading: {e}", exc_info=True)
                await asyncio.sleep(self.config.CHECK_INTERVAL_SECONDS)

    async def _get_current_prices(self) -> Dict[str, float]:
        """Obtiene los precios actuales para todos los símbolos monitoreados."""
        tasks = [self.data_provider.get_ticker_price(s) for s in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for symbol, result in zip(self.symbols, results):
            if isinstance(result, dict) and 'price' in result:
                prices[symbol] = float(result['price'])
        
        self.logger.info(f"Precios actuales obtenidos para {len(prices)} símbolos.")
        return prices

    async def _generate_tcn_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """Genera una lista de señales de trading a partir de las predicciones de TCN."""
        self.logger.info("🧠 Evaluando señales con modelos TCN...")
        signals_to_process = []

        for symbol in self.symbols:
            try:
                prediction = await self._get_tcn_prediction(symbol)
                if not prediction:
                    continue

                self.logger.info(f"🔮 Predicción para {symbol}: "
                                 f"Señal={prediction['signal']}, "
                                 f"Confianza={prediction['confidence']:.2f}")

                signal = prediction['signal']
                confidence = prediction['confidence']

                is_valid = False
                if signal == 'BUY' and confidence >= self.config.TCN_BUY_CONFIDENCE_THRESHOLD:
                    is_valid = True
                elif signal == 'SELL' and confidence >= self.config.TCN_SELL_CONFIDENCE_THRESHOLD:
                    is_valid = True
                
                if is_valid:
                    self.logger.info(f"✅ Señal VÁLIDA para {symbol} ({signal}) con confianza {confidence:.2f} detectada.")
                    prediction['current_price'] = prices.get(symbol, 0)
                    signals_to_process.append(prediction)
                else:
                    self.logger.info(f"-> Señal para {symbol} ({signal}) no cumple el umbral de confianza. Se ignora.")

            except Exception as e:
                self.logger.error(f"❌ Error generando señal TCN para {symbol}: {e}")

        return signals_to_process

    async def _get_tcn_prediction(self, symbol: str) -> Dict:
        """Obtiene una predicción TCN para un único símbolo."""
        # 1. Obtener datos de mercado (klines)
        klines = await self.data_provider.get_klines(symbol, interval="1m", limit=100)
        if not klines or len(klines) < 50:
            self.logger.warning(f"Datos de klines insuficientes para {symbol}.")
            return None
        
        # 2. Hacer la predicción
        prediction = await self.tcn_predictor.predict_from_real_data(symbol, klines)
        return prediction

    async def _process_signals(self, signals: List[Dict]):
        """Procesa una lista de señales de trading válidas."""
        for signal_data in signals:
            symbol = signal_data['pair']
            signal_type = signal_data['signal']
            
            # Lógica de ejemplo: Por ahora solo se loguea la decisión.
            # La integración con risk_manager para abrir/cerrar posiciones iría aquí.
            self.logger.info(f"ACTION => Procesando señal de {signal_type} para {symbol}.")

            # TODO: Implementar la lógica de gestión de posiciones.
            # - Comprobar si ya existe una posición.
            # - Si es señal de VENTA y hay posición -> considerar cerrar.
            # - Si es señal de COMPRA y no hay posición -> considerar abrir.
            # - Llamar a self.risk_manager.check_risk_limits_before_trade(...)
            # - Llamar a self.risk_manager.open_position(...) o close_position(...)

    async def shutdown(self):
        """Realiza un apagado controlado del sistema."""
        self.logger.info("🔄 Iniciando apagado del sistema...")
        self.status = TradingManagerStatus.STOPPED
        
        if self.data_provider:
            await self.data_provider.__aexit__(None, None, None)
            self.logger.info("-> Sesión del proveedor de datos cerrada.")

        self.logger.info("✅ Sistema apagado correctamente.") 