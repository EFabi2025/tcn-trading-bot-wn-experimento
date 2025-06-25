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
from market_regime_filter import MarketRegimeFilter, MarketRegime

# Módulos de predicción y datos
from real_binance_predictor import BinanceDataProvider, RealTCNPredictor
from definitivo_tcn_predictor import DefinitivoTCNPredictor

# ✅ NUEVO: Motor de Features Híbridas Optimizado
from hybrid_features_engine import HybridFeaturesEngine

# Módulos de utilidad
from smart_discord_notifier import SmartDiscordNotifier, NotificationPriority

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
            
            # 6. Notificador de Discord
            self.discord_notifier = SmartDiscordNotifier()
            self.logger.info("✅ Notificador de Discord listo.")

            # 7. Filtro de Régimen de Mercado
            if self.config.ENABLE_MARKET_REGIME_FILTER:
                self.market_regime_filter = MarketRegimeFilter(self.data_provider, self.logger)
                self.logger.info("✅ Filtro de Régimen de Mercado Activado.")

            # 8. Tareas de monitoreo
            self._setup_monitoring()

            self.status = TradingManagerStatus.RUNNING
            self.logger.info("🎉 ¡Sistema inicializado y listo para operar! Estado: RUNNING.")

            # Enviar notificación de inicio a Discord
            await self.discord_notifier.send_system_notification(
                "🚀 **Bot de Trading TCN Iniciado**\nSistema operativo y monitoreando el mercado.",
                priority=NotificationPriority.HIGH
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

    async def _display_status_report(self, market_regime: MarketRegime, tcn_predictions: List[Dict]):
        """Muestra un reporte de estado completo y lo envía a Discord."""
        try:
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            if not snapshot:
                self.logger.warning("No se pudo obtener el snapshot del portafolio para el reporte.")
                return

            # Ahora pasamos el contexto del mercado y las predicciones al formateador
            report = self.portfolio_manager.format_tcn_style_report(
                snapshot,
                market_regime=market_regime.value if market_regime else None,
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
        """Bucle principal de trading, ahora reestructurado."""
        if self.status != TradingManagerStatus.RUNNING:
            self.logger.error("El manager no está en estado RUNNING. No se puede iniciar el bucle.")
            return

        self.logger.info("🎯 Iniciando bucle principal de trading...")
        while self.status == TradingManagerStatus.RUNNING:
            try:
                loop_start_time = datetime.now()
                
                # --- NUEVO ORDEN ---
                # 1. Obtener precios y contexto de mercado
                prices = await self._get_current_prices()
                market_regime, risk_adjustment_factor = await self._get_market_regime_and_risk_factor()

                # 2. Generar TODAS las predicciones TCN para el reporte
                all_predictions = await self._generate_tcn_predictions(prices)

                # 3. Mostrar el reporte de estado COMPLETO
                await self._display_status_report(market_regime, all_predictions)

                # 4. Filtrar solo las señales válidas para operar
                valid_signals = self._filter_valid_signals(all_predictions, market_regime)
                
                # 5. Procesar señales válidas
                if valid_signals:
                    await self._process_signals(valid_signals, risk_adjustment_factor)
                else:
                    self.logger.info("🤔 No se generaron señales de trading válidas en este ciclo.")

                # 6. Esperar al siguiente ciclo
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

    async def _get_market_regime_and_risk_factor(self) -> tuple[MarketRegime, float]:
        """Obtiene el régimen de mercado y el factor de ajuste de riesgo asociado."""
        market_regime = MarketRegime.RANGING
        risk_adjustment_factor = 1.0

        if self.config.ENABLE_MARKET_REGIME_FILTER and self.market_regime_filter:
            regime, details = await self.market_regime_filter.get_market_regime()
            market_regime = regime
            self.logger.info(f"🏛️ Régimen de mercado detectado: {market_regime.value} ({details.get('reason', 'N/A')})")
            
            if market_regime == MarketRegime.HIGH_VOLATILITY:
                risk_adjustment_factor = 0.5
                self.logger.warning(
                    f"🔥 ALTA VOLATILIDAD. El tamaño de las posiciones se reducirá en un 50% (Factor de ajuste: {risk_adjustment_factor})."
                )
        return market_regime, risk_adjustment_factor

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

    async def _generate_tcn_predictions(self, prices: Dict[str, float]) -> List[Dict]:
        """Genera una lista con TODAS las predicciones de TCN para el reporte."""
        self.logger.info("🧠 Generando predicciones con modelos TCN...")
        all_predictions = []

        for symbol in self.symbols:
            try:
                prediction = await self._get_tcn_prediction(symbol)
                if prediction:
                    prediction['current_price'] = prices.get(symbol, 0)
                    all_predictions.append(prediction)
                    
                    # ✅ NUEVO: Mostrar información del motor de features usado
                    engine = prediction.get('features_engine', 'unknown')
                    quality = prediction.get('features_quality', 0.0)
                    engine_info = f" [{engine}"
                    if engine == 'hybrid_optimized':
                        engine_info += f", Q:{quality:.2f}"
                    engine_info += "]"
                    
                    self.logger.info(f"🔮 Predicción para {symbol}: Señal={prediction['signal']}, Confianza={prediction['confidence']:.2f}{engine_info}")

            except Exception as e:
                self.logger.error(f"❌ Error generando predicción TCN para {symbol}: {e}")

        return all_predictions

    def _calculate_features_quality(self, features_array: np.ndarray) -> float:
        """🔍 Calcular puntuación de calidad de features híbridas"""
        try:
            if features_array is None or len(features_array.shape) != 2:
                return 0.0
            
            # Métricas de calidad
            nan_ratio = np.isnan(features_array).sum() / features_array.size
            inf_ratio = np.isinf(features_array).sum() / features_array.size
            
            # Variabilidad por feature
            std_per_feature = features_array.std(axis=0)
            constant_features_ratio = (std_per_feature < 1e-6).sum() / features_array.shape[1]
            
            # Rango de valores (normalización)
            value_range = features_array.max() - features_array.min()
            normalized_range = min(value_range / 10.0, 1.0)  # Penalizar rangos extremos
            
            # Calcular puntuación (0-1)
            quality_score = (
                (1 - nan_ratio) * 0.3 +           # 30% - sin NaN
                (1 - inf_ratio) * 0.3 +           # 30% - sin Inf
                (1 - constant_features_ratio) * 0.2 +  # 20% - variabilidad
                normalized_range * 0.2             # 20% - rango apropiado
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculando calidad de features: {e}")
            return 0.0

    async def _predict_with_hybrid_features(self, symbol: str, features_array: np.ndarray) -> Dict:
        """🧠 Hacer predicción usando features híbridas precalculadas"""
        try:
            if symbol not in self.tcn_predictor.models:
                self.logger.error(f"❌ Modelo no disponible para {symbol}")
                return None
            
            model = self.tcn_predictor.models[symbol]
            model_input_shape = model.input_shape
            
            # Preparar input según el tipo de modelo
            if len(model_input_shape) == 2:  # Dense model (batch_size, features)
                # Usar última fila de features
                input_data = features_array[-1:, :]  # Shape: (1, 66)
                
            elif len(model_input_shape) == 3:  # LSTM/TCN model (batch_size, timesteps, features)
                timesteps = model_input_shape[1]
                expected_features = model_input_shape[2]
                
                # Verificar que tenemos suficientes timesteps
                if features_array.shape[0] < timesteps:
                    self.logger.error(f"❌ {symbol}: Datos insuficientes para secuencia: {features_array.shape[0]} < {timesteps}")
                    return None
                
                # Tomar últimos timesteps
                sequence_data = features_array[-timesteps:, :]  # Shape: (timesteps, features)
                
                # Ajustar features si es necesario
                if sequence_data.shape[1] != expected_features:
                    if sequence_data.shape[1] < expected_features:
                        padding = np.zeros((sequence_data.shape[0], expected_features - sequence_data.shape[1]))
                        sequence_data = np.concatenate([sequence_data, padding], axis=1)
                    else:
                        sequence_data = sequence_data[:, :expected_features]
                
                input_data = np.expand_dims(sequence_data, axis=0)  # Shape: (1, timesteps, features)
            
            else:
                self.logger.error(f"❌ {symbol}: Shape de modelo no soportado: {model_input_shape}")
                return None
            
            # Hacer predicción
            prediction = model.predict(input_data, verbose=0)
            probabilities = prediction[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]
            
            result = {
                'pair': symbol,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'features_count': input_data.shape[-1],
                'model_type': 'hybrid_definitivo',
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error en predicción híbrida para {symbol}: {e}")
            return None

    def _filter_valid_signals(self, predictions: List[Dict], market_regime: MarketRegime) -> List[Dict]:
        """Filtra la lista de predicciones para obtener solo señales operables."""
        self.logger.info("🚦 Filtrando señales de trading válidas...")
        valid_signals = []
        for pred in predictions:
            signal = pred['signal']
            confidence = pred['confidence']
            symbol = pred['pair']

            if self.config.ENABLE_MARKET_REGIME_FILTER:
                if market_regime == MarketRegime.BEARISH and signal == 'BUY':
                    self.logger.warning(f"🚫 {symbol}: Señal de COMPRA ignorada debido a régimen de mercado BAJISTA.")
                    continue
                if market_regime == MarketRegime.HIGH_VOLATILITY and signal == 'BUY':
                     self.logger.warning(f"🔥 {symbol}: Señal de COMPRA en ALTA VOLATILIDAD, se procede con riesgo reducido.")

            is_valid = False
            if signal == 'BUY' and confidence >= self.config.TCN_BUY_CONFIDENCE_THRESHOLD:
                is_valid = True
            elif signal == 'SELL' and confidence >= self.config.TCN_SELL_CONFIDENCE_THRESHOLD:
                is_valid = True
            
            if is_valid:
                self.logger.info(f"✅ Señal VÁLIDA para {symbol} ({signal}) con confianza {confidence:.2f} detectada.")
                valid_signals.append(pred)
            else:
                self.logger.info(f"-> Señal para {symbol} ({signal}) no cumple el umbral de confianza. Se ignora.")
        
        return valid_signals

    async def _get_tcn_prediction(self, symbol: str) -> Dict:
        """
        ✅ NUEVA IMPLEMENTACIÓN: Obtiene predicción TCN usando Features Híbridas optimizadas
        """
        try:
            # 1. Obtener datos de mercado
            klines = await self.data_provider.get_klines(symbol, interval="1m", limit=100)
            if not klines or len(klines) < 50:
                self.logger.warning(f"Datos de klines insuficientes para {symbol}.")
                return None
            
            # 2. ✅ USAR MOTOR HÍBRIDO: Generar features limpias y optimizadas
            features_array = await self.hybrid_features_engine.compute_features_hybrid(symbol, klines)
            if features_array is None or features_array.shape != (48, 66):
                self.logger.error(f"❌ No se pudieron generar features híbridas para {symbol} - Shape: {features_array.shape if features_array is not None else 'None'}")
                # Fallback al predictor original
                prediction = await self.tcn_predictor.predict_from_real_data(symbol, klines)
                if prediction:
                    prediction['features_engine'] = 'definitivo_fallback'
                return prediction
            
            # 3. Calcular calidad de features
            features_quality = self._calculate_features_quality(features_array)
            
            # 4. Hacer predicción con features híbridas
            prediction = await self._predict_with_hybrid_features(symbol, features_array)
            
            if prediction:
                prediction['features_engine'] = 'hybrid_optimized'
                prediction['features_quality'] = features_quality
                
                # Log de calidad de features
                self.logger.info(f"🔮 {symbol}: Predicción con features híbridas (calidad: {features_quality:.2f})")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Error en predicción híbrida para {symbol}: {e}")
            
            # Fallback seguro al predictor original
            try:
                klines = await self.data_provider.get_klines(symbol, interval="1m", limit=100)
                prediction = await self.tcn_predictor.predict_from_real_data(symbol, klines)
                if prediction:
                    prediction['features_engine'] = 'definitivo_fallback'
                    self.logger.warning(f"⚠️ {symbol}: Usando predictor original como fallback")
                return prediction
            except Exception as fallback_error:
                self.logger.error(f"❌ Error también en fallback para {symbol}: {fallback_error}")
                return None

    async def _process_signals(self, signals: List[Dict], risk_adjustment_factor: float):
        """Procesa una lista de señales de trading válidas."""
        for signal_data in signals:
            symbol = signal_data['pair']
            signal_type = signal_data['signal']
            confidence = signal_data['confidence']
            current_price = signal_data.get('current_price', 0)
            
            self.logger.info(f"ACTION => Procesando señal de {signal_type} para {symbol}.")
            
            try:
                existing_position = await self.portfolio_manager.get_position(symbol)
                
                if signal_type == 'BUY':
                    if existing_position and hasattr(existing_position, 'quantity') and existing_position.quantity > 0:
                        self.logger.info(f"🔄 {symbol}: Ya existe posición LONG, se ignora señal BUY.")
                        continue
                    
                    risk_check = await self.risk_manager.check_risk_limits_before_trade(
                        symbol, 'BUY', current_price, confidence
                    )
                    
                    if risk_check['approved']:
                        trade_amount = self.risk_manager.calculate_position_size(
                            symbol, current_price, confidence, risk_adjustment_factor
                        )
                        
                        if trade_amount and trade_amount > 0:
                            self.logger.info(f"💰 EJECUTANDO COMPRA: {symbol} - ${trade_amount:.2f} @ ${current_price:.2f}")
                            
                            result = await self.risk_manager.open_position(
                                symbol=symbol,
                                side='BUY',
                                amount=trade_amount,
                                price=current_price,
                                confidence=confidence,
                                signal_data=signal_data
                            )
                            
                            if result and result.get('success'):
                                self.logger.info(f"✅ COMPRA EXITOSA: {symbol} - {result}")
                                
                                await self.discord_notifier.send_trade_notification(
                                    f"🟢 **COMPRA EJECUTADA**\n"
                                    f"**Par:** {symbol}\n"
                                    f"**Precio:** ${current_price:.2f}\n"
                                    f"**Cantidad:** ${trade_amount:.2f}\n"
                                    f"**Confianza:** {confidence:.1%}",
                                    priority=NotificationPriority.HIGH
                                )
                            else:
                                self.logger.error(f"❌ FALLO EN COMPRA: {symbol} - {result}")
                        else:
                            self.logger.warning(f"⚠️ {symbol}: Cantidad de compra calculada es 0 o inválida.")
                    else:
                        self.logger.warning(f"🚫 {symbol}: Compra rechazada por gestión de riesgo: {risk_check.get('reason', 'N/A')}")
                
                elif signal_type == 'SELL':
                    if not existing_position or not hasattr(existing_position, 'quantity') or existing_position.quantity <= 0:
                        self.logger.info(f"🔄 {symbol}: No hay posición LONG para vender, se ignora señal SELL.")
                        continue
                    
                    risk_check = await self.risk_manager.check_risk_limits_before_trade(
                        symbol, 'SELL', current_price, confidence
                    )
                    
                    if risk_check['approved']:
                        position_quantity = existing_position.quantity if hasattr(existing_position, 'quantity') else 0
                        self.logger.info(f"💸 EJECUTANDO VENTA: {symbol} - {position_quantity} @ ${current_price:.2f}")
                        
                        result = await self.risk_manager.close_position(
                            symbol=symbol,
                            price=current_price,
                            confidence=confidence,
                            signal_data=signal_data
                        )
                        
                        if result and result.get('success'):
                            self.logger.info(f"✅ VENTA EXITOSA: {symbol} - {result}")
                            
                            profit_loss = result.get('profit_loss', 0)
                            profit_emoji = "🟢" if profit_loss >= 0 else "🔴"
                            
                            await self.discord_notifier.send_trade_notification(
                                f"{profit_emoji} **VENTA EJECUTADA**\n"
                                f"**Par:** {symbol}\n"
                                f"**Precio:** ${current_price:.2f}\n"
                                f"**P&L:** ${profit_loss:.2f}\n"
                                f"**Confianza:** {confidence:.1%}",
                                priority=NotificationPriority.HIGH
                            )
                        else:
                            self.logger.error(f"❌ FALLO EN VENTA: {symbol} - {result}")
                    else:
                        self.logger.warning(f"🚫 {symbol}: Venta rechazada por gestión de riesgo: {risk_check.get('reason', 'N/A')}")
                
                else:
                    self.logger.info(f"🔄 {symbol}: Señal HOLD, mantener posición actual.")
                    
            except Exception as e:
                self.logger.error(f"❌ Error procesando señal {signal_type} para {symbol}: {e}", exc_info=True)

    async def shutdown(self):
        """Realiza un apagado controlado del sistema."""
        self.logger.info("🔄 Iniciando apagado del sistema...")
        self.status = TradingManagerStatus.STOPPED
        
        if self.data_provider:
            await self.data_provider.__aexit__(None, None, None)
            self.logger.info("-> Sesión del proveedor de datos cerrada.")

        self.logger.info("✅ Sistema apagado correctamente.") 