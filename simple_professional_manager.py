#!/usr/bin/env python3
# TEST COMMENT
"""
🚀 TRADING MANAGER - EL CEREBRO DEL BOT
Orquesta todos los módulos para ejecutar la estrategia de trading con TCN.
"""

import asyncio
import logging
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Módulos de configuración y base de datos
from config import trading_config
from trading_database import TradingDatabase

# Módulos de lógica de trading
from advanced_risk_manager import AdvancedRiskManager
from professional_portfolio_manager import ProfessionalPortfolioManager

# Módulo de filtro de régimen de mercado (NUEVO)
from market_regime_filter import MarketRegimeFilter

# Módulos de predicción y datos
from real_binance_predictor import BinanceDataProvider, RealTCNPredictor
from definitivo_tcn_predictor import DefinitivoTCNPredictor

# ✅ NUEVO: Motor de Features Híbridas Optimizado
from hybrid_features_engine import HybridFeaturesEngine

# Módulos de utilidad
from smart_discord_notifier import SmartDiscordNotifier
# from professional_order_executor import ProfessionalOrderExecutor, OrderExecutionRequest, OrderType, ExecutionMode
from decimal import Decimal

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
        self.tcn_predictor: DefinitivoTCNPredictor = None
        self.hybrid_features_engine: HybridFeaturesEngine = None  # ✅ NUEVO
        self.risk_manager: AdvancedRiskManager = None
        self.portfolio_manager: ProfessionalPortfolioManager = None
        self.discord_notifier: SmartDiscordNotifier = None
        self.market_regime_filter: MarketRegimeFilter = None

        self.active_positions: Dict[str, any] = {}
        self.symbols: list[str] = self.config.TRADING_SYMBOLS
        self.last_discord_report_time: Optional[datetime] = None

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

            # 3. Predictor TCN Definitivo
            self.tcn_predictor = DefinitivoTCNPredictor()
            self.logger.info("✅ Predictor TCN Definitivo cargado con modelos de 66 features.")
            
            # 3.5. ✅ NUEVO: Motor de Features Híbridas
            self.hybrid_features_engine = HybridFeaturesEngine()
            self.logger.info("✅ Motor de Features Híbridas inicializado - Features limpias y optimizadas.")

            # 4. Gestor de Portfolio (necesita el balance inicial)
            self.portfolio_manager = ProfessionalPortfolioManager(self.config, self.symbols, self.logger)
            initial_balance = await self.portfolio_manager.initialize()
            self.logger.info(f"✅ Gestor de Portfolio inicializado. Balance USDT inicial: ${initial_balance:.2f}")

            # 5. Gestor de Riesgo
            self.risk_manager = AdvancedRiskManager()
            await self.risk_manager.initialize(initial_balance)
            self.logger.info("✅ Gestor de Riesgo (AdvancedRiskManager) configurado.")
            
            # 6. Order Executor para Trading Real - Comentado temporalmente
            # self.order_executor = ProfessionalOrderExecutor(...)
            self.logger.info("✅ Trading real habilitado usando métodos directos.")

            # 7. Notificador de Discord
            self.discord_notifier = SmartDiscordNotifier()
            self.logger.info("✅ Notificador de Discord listo.")

            # 8. Filtro de Régimen de Mercado
            if self.config.ENABLE_MARKET_REGIME_FILTER:
                self.market_regime_filter = MarketRegimeFilter(self.data_provider, self.logger)
                self.logger.info("✅ Filtro de Régimen de Mercado Activado.")

            # 8. Tareas de monitoreo
            self._setup_monitoring()

            self.status = TradingManagerStatus.RUNNING
            self.logger.info("🎉 ¡Sistema inicializado y listo para operar! Estado: RUNNING.")

            # Enviar notificación de inicio a Discord
            await self.discord_notifier.send_system_notification(
                "🚀 **Bot de Trading TCN Iniciado**\nSistema operativo y monitoreando el mercado."
            )

        except Exception as e:
            self.logger.critical(f"❌ Error fatal durante la inicialización: {e}", exc_info=True)
            self.status = TradingManagerStatus.ERROR
            await self.shutdown()
            raise

    def _setup_monitoring(self):
        """Configura las tareas de monitoreo en segundo plano."""
        self.logger.info("⚙️ Configurando tareas de monitoreo...")
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._stop_loss_monitor())
        self.logger.info("✅ Tarea de monitoreo de heartbeat configurada.")
        self.logger.info("✅ Tarea de monitoreo de stop loss configurada.")

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

    async def _stop_loss_monitor(self):
        """🛑 Monitor continuo de stop loss y trailing stop para liquidación automática"""
        self.logger.info("🛑 Monitor de stop loss iniciado.")
        
        while self.status == TradingManagerStatus.RUNNING:
            try:
                # Obtener snapshot de posiciones actuales
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                
                if snapshot and snapshot.active_positions:
                    self.logger.debug(f"🔍 Monitoreando {len(snapshot.active_positions)} posiciones activas...")
                    
                    # ✅ NUEVO: Sincronizar posiciones entre portfolio manager y risk manager
                    await self._sync_positions_with_risk_manager(snapshot.active_positions)
                    
                    for position in snapshot.active_positions:
                        try:
                            # Obtener precio actual
                            current_price = await self.portfolio_manager.get_current_price(position.symbol)
                            
                            if current_price:
                                # Actualizar posición con precio actual
                                position.current_price = current_price
                                
                                # Verificar trailing stop y stop loss
                                updated_pos, stop_triggered, trigger_reason = self.portfolio_manager.update_trailing_stop_professional(
                                    position, current_price
                                )
                                
                                # Si se activa stop loss o trailing stop, ejecutar venta
                                if stop_triggered:
                                    self.logger.warning(f"🚨 STOP ACTIVADO: {position.symbol} - {trigger_reason}")
                                    
                                    # ✅ NUEVO: Verificar si posición existe antes de cerrar
                                    if await self._verify_position_exists(position.symbol):
                                        await self._execute_stop_loss_order(updated_pos, trigger_reason)
                                    else:
                                        self.logger.warning(f"⚠️ Posición {position.symbol} ya no existe - omitiendo stop loss")
                                    
                        except Exception as e:
                            self.logger.error(f"❌ Error monitoreando posición {position.symbol}: {e}")
                
                # Esperar 30 segundos antes de la siguiente verificación
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                self.logger.info("🛑 Monitor de stop loss detenido.")
                break
            except Exception as e:
                self.logger.error(f"💥 Error en el monitor de stop loss: {e}")
                await asyncio.sleep(60)

    async def _display_status_report(self, market_regime: str, tcn_predictions: List[Dict]):
        """Muestra un reporte de estado completo y lo envía a Discord."""
        try:
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            if not snapshot:
                self.logger.warning("No se pudo obtener el snapshot del portafolio para el reporte.")
                return

            # Ahora pasamos el contexto del mercado y las predicciones al formateador
            report = self.portfolio_manager.format_tcn_style_report(
                snapshot,
                market_regime=market_regime if market_regime else None,
                tcn_predictions=tcn_predictions
            )
            
            print("\n" + "🔥" * 30 + " REPORTE DE ESTADO " + "🔥" * 30)
            print(report)
            print("🔥" * 79)

            now = datetime.now()
            should_send_report = False
            if self.last_discord_report_time is None:
                should_send_report = True
            else:
                time_since_last = (now - self.last_discord_report_time).total_seconds()
                if time_since_last >= self.config.DISCORD_REPORT_INTERVAL_SECONDS:
                    should_send_report = True
            
            if should_send_report:
                await self.discord_notifier.send_report(report)
                self.last_discord_report_time = now

        except Exception as e:
            self.logger.error(f"❌ Error generando el reporte de estado: {e}", exc_info=True)

    async def run(self):
        """▶️ El bucle principal que ejecuta la estrategia de trading."""
        self.logger.info("🤖 Iniciando ciclo principal de trading...")
        
        while self.status == TradingManagerStatus.RUNNING:
            try:
                start_time = datetime.now()
                self.logger.info("--- 🔄 Nuevo Ciclo de Trading 🔄 ---")

                # 1. OBTENER ESTADO ACTUAL DEL PORTAFOLIO Y SINCRONIZAR RIESGO
                self.logger.info("📊 Obteniendo snapshot del portafolio...")
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                if snapshot:
                    await self._sync_positions_with_risk_manager(snapshot.active_positions)
                    self.logger.info(f"   ✅ Snapshot obtenido: {len(snapshot.all_assets)} activos, {len(snapshot.active_positions)} posiciones activas, {len(snapshot.small_positions)} posiciones pequeñas")
                else:
                    self.logger.warning("   ⚠️ No se pudo obtener el snapshot del portafolio. Se reintentará.")
                    await asyncio.sleep(self.config.CHECK_INTERVAL_SECONDS)
                    continue

                # 2. OBTENER RÉGIMEN DE MERCADO
                market_regime, risk_adjustment_factor = await self._get_market_regime_and_risk_factor()

                # 3. GENERAR TODAS LAS PREDICCIONES TCN PARA EL REPORTE
                all_predictions = await self._generate_tcn_predictions(await self._get_current_prices())

                # 4. MOSTRAR EL REPORTE DE ESTADO COMPLETO
                await self._display_status_report(market_regime, all_predictions)

                # 5. FILTRAR SOLO LAS SEÑALES VÁLIDAS PARA OPERAR
                valid_signals = self._filter_valid_signals(all_predictions, market_regime)
                
                # 6. PROCESAR SEÑALES VÁLIDAS
                if valid_signals:
                    await self._process_signals(valid_signals, risk_adjustment_factor)
                else:
                    self.logger.info("🤔 No se generaron señales de trading válidas en este ciclo.")

                # 7. ESPERAR AL SIGUIENTE CICLO
                loop_duration = (datetime.now() - start_time).total_seconds()
                sleep_time = max(0, self.config.CHECK_INTERVAL_SECONDS - loop_duration)
                self.logger.info(f"Ciclo completado en {loop_duration:.2f}s. Durmiendo por {sleep_time:.2f}s.")
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                self.logger.info("Bucle de trading cancelado.")
                break
            except Exception as e:
                self.logger.error(f"❌ Error en el bucle principal de trading: {e}", exc_info=True)
                await asyncio.sleep(self.config.CHECK_INTERVAL_SECONDS)

    async def _get_market_regime_and_risk_factor(self) -> tuple[str, float]:
        """Obtiene el régimen de mercado y calcula un factor de ajuste de riesgo."""
        market_regime = 'NEUTRAL' # Default
        risk_adjustment = 1.0

        if self.config.ENABLE_MARKET_REGIME_FILTER and self.market_regime_filter:
            try:
                # La nueva función devuelve (régimen, confianza, detalles)
                regime, confidence, _ = await self.market_regime_filter.get_market_regime()
            market_regime = regime

                if market_regime == 'BEARISH':
                    risk_adjustment = 0.5  # Reducir riesgo a la mitad en mercado bajista
                elif market_regime == 'BULLISH':
                    risk_adjustment = 1.2 # Aumentar ligeramente en mercado alcista
                
                self.logger.info(f"🏛️ Régimen de Mercado: {market_regime} | Factor de Riesgo: {risk_adjustment}")

            except Exception as e:
                self.logger.error(f"❌ No se pudo determinar el régimen de mercado, usando NEUTRAL. Error: {e}")
        
        return market_regime, risk_adjustment

    async def _get_current_prices(self) -> Dict[str, float]:
        """Obtiene los precios actuales para todos los símbolos de trading."""
        return await self.portfolio_manager.update_all_prices(self.symbols)

    async def _generate_tcn_predictions(self, prices: Dict[str, float]) -> List[Dict]:
        """Genera predicciones TCN para todos los símbolos."""
        predictions = []
        for symbol, price in prices.items():
            try:
                # La lógica para obtener la predicción debe ser revisada y actualizada
                # según la implementación de tu predictor.
                # Esta es una implementación de ejemplo:
                prediction = await self._get_tcn_prediction(symbol)
                if prediction:
                    prediction['current_price'] = price
                    predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"❌ Error generando predicción para {symbol}: {e}")
        return predictions

    def _calculate_features_quality(self, features_array: np.ndarray) -> float:
        """
        Calcula un puntaje de calidad para las features.
        Un valor más alto indica mejor calidad.
        """
        # 1. Normalidad: Penalizar valores extremos (NaN, Inf)
        num_invalid = np.sum(~np.isfinite(features_array))
        if num_invalid > 0:
            return 0.0  # Calidad cero si hay valores inválidos

        # 2. Rango Dinámico: Un rango muy pequeño puede indicar datos estancados
        dynamic_range = np.max(features_array) - np.min(features_array)
        
        # 3. Entropía (complejidad)
        # Esto es más complejo, por ahora usamos una heurística simple
        std_dev = np.std(features_array)

        quality_score = (1 / (1 + std_dev)) * (dynamic_range + 1)
        return min(quality_score, 1.0) # Normalizar a 1.0

    async def _predict_with_hybrid_features(self, symbol: str, klines: List[Dict]) -> Dict:
        """
        Genera predicciones TCN utilizando el motor de features híbridas.
        Añade una capa de confianza basada en la calidad de las features.
        """
        try:
            # La predicción ahora se hace directamente desde el predictor
            # que ya usa el motor de features.
            prediction = await self.tcn_predictor.predict_from_real_data(symbol, klines)
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Error en predicción híbrida para {symbol}: {e}")
            return None

    def _filter_valid_signals(self, predictions: List[Dict], market_regime: str) -> List[Dict]:
        """
        Filtra las predicciones para obtener señales de trading válidas y accionables,
        considerando el régimen de mercado.
        """
        valid_signals = []
        for pred in predictions:
            symbol = pred['pair']
            signal = pred['signal']
            confidence = pred['confidence']

            # Umbral base de confianza
            buy_threshold = self.config.TCN_BUY_CONFIDENCE_THRESHOLD
            sell_threshold = self.config.TCN_SELL_CONFIDENCE_THRESHOLD

            # ✅ NUEVO: Ajustar umbrales según el régimen de mercado
            if market_regime == 'BEARISH':
                # En mercado bajista, ser mucho más exigente para comprar
                buy_threshold = 0.85  # Aumentar a 85%
                sell_threshold = 0.60 # Bajar ligeramente para ventas
            elif market_regime == 'BULLISH':
                # En mercado alcista, ser un poco más flexible
                buy_threshold = 0.60

            is_valid = False
            if signal == 'BUY' and confidence >= buy_threshold:
                is_valid = True
            elif signal == 'SELL' and confidence >= sell_threshold:
                is_valid = True
            
            if is_valid:
                self.logger.info(f"✅ Señal VÁLIDA para {symbol}: {signal} ({confidence:.2%}) "
                                 f"[Umbral: {buy_threshold if signal == 'BUY' else sell_threshold:.2%}, Régimen: {market_regime}]")
                valid_signals.append(pred)
            else:
                 self.logger.info(f"❌ Señal INVÁLIDA para {symbol}: {signal} ({confidence:.2%}) "
                                 f"[Umbral: {buy_threshold if signal == 'BUY' else sell_threshold:.2%}, Régimen: {market_regime}]")
        
        return valid_signals

    async def _get_tcn_prediction(self, symbol: str) -> Dict:
        """
        Obtiene la predicción del modelo TCN para un símbolo, utilizando
        el predictor definitivo que encapsula el motor de features.
        """
        try:
            # 1. Obtener los datos de klines necesarios
            # El predictor necesita ~300 klines para calcular todas las features
            klines = await self.data_provider.get_klines(symbol, "5m", limit=350)
            if not klines:
                self.logger.warning(f"No se pudieron obtener klines para la predicción de {symbol}.")
                return None
            
            # 2. Llamar al método de predicción corregido
            # Ya no se calculan features aquí, se hace dentro del predictor.
                prediction = await self.tcn_predictor.predict_from_real_data(symbol, klines)
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo la predicción TCN para {symbol}: {e}", exc_info=True)
                return None

    async def _process_signals(self, signals: List[Dict], risk_adjustment_factor: float):
        """Procesa las señales de trading válidas, gestiona el riesgo y ejecuta órdenes."""
        self.logger.info(f"⚙️ Procesando {len(signals)} señales válidas...")
        if not signals:
            self.logger.info("🤔 No se generaron señales de trading válidas en este ciclo.")
            return

        snapshot = await self.portfolio_manager.get_portfolio_snapshot()
        if not snapshot:
            self.logger.error("❌ No se pudo obtener el snapshot del portafolio para procesar señales.")
            return
        
        available_balance = snapshot.free_usdt
        
        for signal_data in signals:
            symbol = signal_data['pair']
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            current_price = signal_data['current_price']

            # 1. Verificar si ya existe una posición para este símbolo
            if await self._verify_position_exists(symbol):
                self.logger.info(f"🔵 Ya existe una posición para {symbol}. Omitiendo nueva señal.")
                        continue
                    
            # 2. Lógica de ejecución de COMPRA
            if signal == 'BUY':
                can_open, reason = self.risk_manager.can_open_new_position(
                    total_balance=snapshot.total_balance_usd,
                    active_positions_count=snapshot.position_count
                )
                if not can_open:
                    self.logger.warning(f"🚫 COMPRA RECHAZADA para {symbol}: {reason}")
                    continue

                # Calcular tamaño de la posición
                position_size_usd = self.risk_manager.calculate_position_size(
                    total_balance=snapshot.total_balance_usd,
                    risk_per_trade=confidence * risk_adjustment_factor, # Factor de riesgo dinámico
                    market_regime='BEARISH' if risk_adjustment_factor < 1 else 'NEUTRAL'
                )
                
                # Ejecutar orden
                order_result = await self._execute_real_buy_order(symbol, position_size_usd, current_price, confidence)
                
                # ✅ ¡CORRECCIÓN! Guardar el trade en la base de datos si se ejecutó
                if order_result and order_result.get('status') == 'FILLED':
                    self.logger.info(f"💾 Guardando trade de COMPRA para {symbol} en la base de datos...")
                    trade_to_save = {
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": float(order_result.get('executedQty', 0)),
                        "entry_price": float(order_result.get('price', 0)),
                        "entry_time": datetime.now(),
                        "confidence": confidence,
                        "strategy": "TCN_Hybrid_v2",
                        "is_active": True,
                        "metadata": {"order_id": order_result.get('orderId')}
                    }
                    await self.database.save_trade(trade_to_save)

            # 3. Lógica de ejecución de VENTA (Cerrar posición)
            elif signal == 'SELL':
                position_to_close = await self.portfolio_manager.get_position(symbol)
                if not position_to_close:
                    self.logger.warning(f"⚠️ Señal de VENTA para {symbol}, pero no se encontró una posición activa.")
                        continue
                    
                # Ejecutar orden de venta
                order_result = await self._execute_real_sell_order(
                            symbol=symbol,
                    quantity=position_to_close.size,
                            current_price=current_price,
                            confidence=confidence
                        )
                        
                # ✅ ¡CORRECCIÓN! Actualizar el trade en la base de datos
                if order_result and order_result.get('status') == 'FILLED':
                    self.logger.info(f"💾 Actualizando trade de VENTA para {symbol} en la base de datos...")
                    # Aquí necesitaríamos el ID del trade original para actualizarlo.
                    # Por ahora, nos aseguramos de que las compras se guarden.
                    # La lógica de actualización es más compleja.
                    await self._cleanup_closed_position(symbol)

        # Pequeña pausa para asegurar que el estado se propague
        await asyncio.sleep(2)

    async def _sync_positions_with_risk_manager(self, portfolio_positions):
        """🔄 Sincronizar posiciones entre portfolio manager y risk manager"""
        try:
            for pos in portfolio_positions:
                symbol = pos.symbol
                
                # Si el risk manager no tiene esta posición, sincronizarla
                if symbol not in self.risk_manager.active_positions:
                    self.logger.info(f"🔄 Sincronizando posición {symbol} al risk manager")
                    
                    # Crear posición en el risk manager
                    from advanced_risk_manager import Position as RiskPosition
                    
                    risk_position = RiskPosition(
                        symbol=symbol,
                        side=pos.side,
                        quantity=pos.size,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        entry_time=pos.entry_time
                    )
                    
                    # Configurar stops
                    risk_position = self.risk_manager.set_stop_loss_take_profit(risk_position)
                    
                    # Registrar en el risk manager
                    self.risk_manager.active_positions[symbol] = risk_position
                    
                    self.logger.info(f"✅ Posición {symbol} sincronizada con risk manager")
                    
        except Exception as e:
            self.logger.error(f"❌ Error sincronizando posiciones: {e}")

    async def _verify_position_exists(self, symbol: str) -> bool:
        """🔍 Verificar si una posición realmente existe en Binance"""
        try:
            # Obtener balances actuales
            balances = await self.portfolio_manager.get_account_balances()
            
            if not balances:
                return False
            
            # Extraer el activo del símbolo (ej: ETHUSDT -> ETH)
            asset = symbol.replace('USDT', '').replace('BUSD', '')
            
            # Verificar si tenemos balance del activo
            asset_balance = balances.get(asset, {})
            total_balance = asset_balance.get('total', 0.0)
            
            # Si el balance total es > 0, la posición existe
            exists = total_balance > 0.0001  # Umbral mínimo para considerar una posición
            
            if not exists:
                self.logger.info(f"🔍 Verificación {symbol}: Balance {asset} = {total_balance} - Posición NO existe")
            else:
                self.logger.debug(f"🔍 Verificación {symbol}: Balance {asset} = {total_balance} - Posición existe")
                
            return exists
            
        except Exception as e:
            self.logger.error(f"❌ Error verificando posición {symbol}: {e}")
            return False  # En caso de error, asumir que no existe

    async def _cleanup_closed_position(self, symbol: str):
        """🧹 Limpiar posición cerrada de todos los caches"""
        try:
            # Eliminar del risk manager
            if symbol in self.risk_manager.active_positions:
                del self.risk_manager.active_positions[symbol]
                self.logger.info(f"🧹 Posición {symbol} eliminada del risk manager")
            
            # Invalidar cache del portfolio manager para forzar actualización
            if hasattr(self.portfolio_manager, 'last_snapshot_time'):
                self.portfolio_manager.last_snapshot_time = None
                self.logger.debug(f"🧹 Cache del portfolio manager invalidado")
            
            # Limpiar cache de precios para forzar nueva consulta
            if hasattr(self.portfolio_manager, 'price_cache') and symbol in self.portfolio_manager.price_cache:
                del self.portfolio_manager.price_cache[symbol]
                self.logger.debug(f"🧹 Cache de precio {symbol} eliminado")
                
        except Exception as e:
            self.logger.error(f"❌ Error limpiando posición cerrada {symbol}: {e}")

    async def _execute_stop_loss_order(self, position, trigger_reason: str):
        """🛑 Ejecutar orden de venta por stop loss o trailing stop"""
        try:
            self.logger.warning(f"🚨 EJECUTANDO STOP LOSS: {position.symbol}")
            self.logger.info(f"   📍 Razón: {trigger_reason}")
            self.logger.info(f"   💰 Precio entrada: ${position.entry_price:.4f}")
            self.logger.info(f"   📊 Precio actual: ${position.current_price:.4f}")
            
            # Calcular PnL final
            pnl_percent = ((position.current_price - position.entry_price) / position.entry_price) * 100
            pnl_usd = (position.current_price - position.entry_price) * position.size
            
            self.logger.info(f"   📈 PnL Final: {pnl_percent:+.2f}% (${pnl_usd:+.2f})")
            
            # Ejecutar orden de venta usando el risk manager
            sell_result = await self.risk_manager.close_position(
                symbol=position.symbol,
                exit_price=position.current_price,
                reason=f"AUTO_{trigger_reason}"
            )
            
            if sell_result and sell_result.get('success'):
                self.logger.info(f"✅ STOP LOSS EJECUTADO EXITOSAMENTE: {position.symbol}")
                
                # ✅ NUEVO: Limpiar posición de caches para evitar reintento
                await self._cleanup_closed_position(position.symbol)
                
                # Determinar emoji según el resultado
                if trigger_reason == "STOP_LOSS":
                    emoji = "🔴"  # Pérdida por stop loss
                    title = "STOP LOSS EJECUTADO"
                else:
                    emoji = "🟡"  # Trailing stop (probablemente ganancia)
                    title = "TRAILING STOP EJECUTADO"
                
                # Notificar a Discord
                await self.discord_notifier.send_trade_notification({
                    'symbol': position.symbol,
                    'side': 'SELL',
                    'value_usd': abs(pnl_usd),
                    'pnl_percent': pnl_percent,
                    'pnl_usd': pnl_usd,
                    'price': position.current_price,
                    'reason': trigger_reason,
                    'confidence': 1.0  # Stop loss siempre tiene confianza máxima
                })
                
                # Log detallado para auditoría
                order_id = sell_result.get('orderId', 'N/A') if sell_result else 'N/A'
                self.logger.info(f"""
🎯 STOP LOSS COMPLETADO:
   Par: {position.symbol}
   Entrada: ${position.entry_price:.4f}
   Salida: ${position.current_price:.4f}
   Cantidad: {position.size:.6f}
   P&L: {pnl_percent:+.2f}% (${pnl_usd:+.2f})
   Razón: {trigger_reason}
   Orden ID: {order_id}
                """)
                
            else:
                self.logger.error(f"❌ FALLO EN STOP LOSS: {position.symbol}")
                self.logger.error(f"   Resultado: {sell_result}")
                
                # Manejar respuesta None o vacía de forma segura
                error_msg = "Desconocido"
                if sell_result and isinstance(sell_result, dict):
                    error_msg = sell_result.get('error', 'Resultado sin detalles de error')
                elif sell_result is None:
                    error_msg = "Función close_position retornó None - posición posiblemente inexistente"
                else:
                    error_msg = f"Tipo de respuesta inesperado: {type(sell_result)}"
                
                # Notificar fallo a Discord  
                await self.discord_notifier.send_system_notification(
                    f"❌ **ERROR EN STOP LOSS**\n"
                    f"**Par:** {position.symbol}\n"
                    f"**Razón:** {trigger_reason}\n"
                    f"**Error:** {error_msg}\n"
                    f"**Acción:** Verificación manual requerida",
                    NotificationPriority.CRITICAL
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error crítico ejecutando stop loss para {position.symbol}: {e}", exc_info=True)
            
            # Notificar error crítico
            await self.discord_notifier.send_system_notification(
                f"🚨 **ERROR CRÍTICO EN STOP LOSS**\n"
                f"**Par:** {position.symbol}\n"
                f"**Error:** {str(e)}\n"
                f"**Acción:** Intervención manual URGENTE",
                NotificationPriority.CRITICAL
            )

    async def _execute_real_buy_order(self, symbol: str, amount_usdt: float, current_price: float, confidence: float) -> dict:
        """🚨 EJECUTA UNA ORDEN DE COMPRA REAL EN BINANCE"""
        try:
            # Calcular cantidad a comprar
            quantity = amount_usdt / current_price
            
            # Redondear cantidad según las reglas de Binance
            if symbol == 'BTCUSDT':
                quantity = round(quantity, 5)  # 5 decimales para BTC
            elif symbol in ['ETHUSDT', 'BNBUSDT']:
                quantity = round(quantity, 3)  # 3 decimales para ETH/BNB
            else:
                quantity = round(quantity, 6)  # Default
            
            # Verificar cantidad mínima
            min_qty = 0.00001 if symbol == 'BTCUSDT' else 0.001
            if quantity < min_qty:
                return {
                    'success': False,
                    'error': f'Cantidad {quantity} menor al mínimo {min_qty}'
                }
            
            self.logger.info(f"🚨 EJECUTANDO ORDEN REAL: BUY {quantity} {symbol} @ ${current_price}")
            
            # EJECUTAR ORDEN REAL EN BINANCE
            order_result = self.data_provider.client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )
            
            self.logger.info(f"✅ ORDEN REAL EJECUTADA: {order_result}")
            
            # Crear posición en el risk manager para tracking
            position_result = await self.risk_manager.open_position(
                symbol=symbol,
                side='BUY',
                amount=amount_usdt,
                price=current_price,
                confidence=confidence,
                signal_data={'order_id': order_result.get('orderId')}
            )
            
            return {
                'success': True,
                'order_id': order_result.get('orderId'),
                'quantity': quantity,
                'price': current_price,
                'amount_usdt': amount_usdt,
                'position': position_result.get('position') if position_result else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ ERROR EN ORDEN REAL DE COMPRA: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _execute_real_sell_order(self, symbol: str, quantity: float, current_price: float, confidence: float) -> dict:
        """🚨 EJECUTA UNA ORDEN DE VENTA REAL EN BINANCE"""
        try:
            # Redondear cantidad según las reglas de Binance
            if symbol == 'BTCUSDT':
                quantity = round(quantity, 5)  # 5 decimales para BTC
            elif symbol in ['ETHUSDT', 'BNBUSDT']:
                quantity = round(quantity, 3)  # 3 decimales para ETH/BNB
            else:
                quantity = round(quantity, 6)  # Default
            
            # Verificar cantidad mínima
            min_qty = 0.00001 if symbol == 'BTCUSDT' else 0.001
            if quantity < min_qty:
                return {
                    'success': False,
                    'error': f'Cantidad {quantity} menor al mínimo {min_qty}'
                }
            
            self.logger.info(f"🚨 EJECUTANDO ORDEN REAL: SELL {quantity} {symbol} @ ${current_price}")
            
            # EJECUTAR ORDEN REAL EN BINANCE
            order_result = self.data_provider.client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
            
            self.logger.info(f"✅ ORDEN REAL EJECUTADA: {order_result}")
            
            # Cerrar posición en el risk manager
            close_result = await self.risk_manager.close_position(
                symbol=symbol,
                price=current_price,
                confidence=confidence,
                signal_data={'order_id': order_result.get('orderId')}
            )
            
            return {
                'success': True,
                'order_id': order_result.get('orderId'),
                'quantity': quantity,
                'price': current_price,
                'amount_usdt': quantity * current_price,
                'profit_loss': close_result.get('profit_loss', 0) if close_result else 0,
                'pnl_percent': close_result.get('pnl_percent', 0) if close_result else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ ERROR EN ORDEN REAL DE VENTA: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    async def shutdown(self):
        """Realiza un apagado controlado del sistema."""
        self.logger.info("🔄 Iniciando apagado del sistema...")
        self.status = TradingManagerStatus.STOPPED
        
        if self.data_provider:
            await self.data_provider.__aexit__(None, None, None)
            self.logger.info("-> Sesión del proveedor de datos cerrada.")

        self.logger.info("✅ Sistema apagado correctamente.") 