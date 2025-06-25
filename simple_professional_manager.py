#!/usr/bin/env python3
"""
🚀 SIMPLE PROFESSIONAL TRADING MANAGER
Sistema de trading básico sin ML para testing inicial
Integrado con Professional Portfolio Manager para reportes TCN
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
import pandas as pd
from dotenv import load_dotenv

# Importar nuestros módulos de risk y database
from advanced_risk_manager import AdvancedRiskManager, Position, RiskLimits
from trading_database import TradingDatabase

# Importar el módulo de Smart Discord Notifier
from smart_discord_notifier import SmartDiscordNotifier

# ✅ NUEVO: Importar Professional Portfolio Manager
from professional_portfolio_manager import ProfessionalPortfolioManager

# ✅ NUEVO: Importar Portfolio Diversification Manager
from portfolio_diversification_manager import PortfolioDiversificationManager, PortfolioPosition

load_dotenv()

@dataclass
class BinanceConfig:
    """⚙️ Configuración de Binance"""
    api_key: str
    secret_key: str
    base_url: str
    environment: str

@dataclass
class AccountInfo:
    """💰 Información de cuenta de Binance"""
    usdt_balance: float
    total_balance_usd: float
    positions: Dict[str, Dict]
    balances: Dict[str, float]

class TradingManagerStatus:
    """📊 Estados del Trading Manager"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class SimpleProfessionalTradingManager:
    """🚀 Trading Manager Profesional Simplificado"""

    def __init__(self):
        """🚀 Inicializar Trading Manager"""
        print("🚀 Simple Professional Trading Manager inicializado")

        # Configuración básica
        self.config = self._load_config()

        # ✅ CORREGIDO: Solo pares con modelos TCN disponibles
        # Excluir temporalmente ADAUSDT, DOTUSDT, SOLUSDT hasta entrenar modelos
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        # ⚠️ PARES PENDIENTES (sin modelos): ["ADAUSDT", "DOTUSDT", "SOLUSDT"]
        self.excluded_symbols = ["ADAUSDT", "DOTUSDT", "SOLUSDT"]

        print(f"📊 Pares activos: {self.symbols}")
        print(f"⏸️ Pares excluidos (sin modelos): {self.excluded_symbols}")

        self.check_interval = 60  # 1 minuto

        # Estado del sistema
        self.status = TradingManagerStatus.STOPPED
        self.database: Optional[TradingDatabase] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.client = None

        # ✅ NUEVO: Professional Portfolio Manager
        self.portfolio_manager: Optional[ProfessionalPortfolioManager] = None

        # ✅ NUEVO: Portfolio Diversification Manager
        self.diversification_manager = PortfolioDiversificationManager()

        # Balance y trading - ✅ CORREGIDO: Inicializar en 0, obtener de Binance
        self.current_balance = 0.0  # Se actualizará desde Binance
        self.session_pnl = 0.0
        self.trade_count = 0
        # ✅ CORREGIDO: Clave por order_id para múltiples posiciones por símbolo
        self.active_positions: Dict[str, Position] = {}
        self.account_info = None

        # ✅ NUEVO: Portfolio tracking
        self.last_portfolio_snapshot = None
        self.last_tcn_report_time = None

        # Smart Discord Notifier
        self.discord_notifier = SmartDiscordNotifier()

        # 🧠 INICIALIZAR TCN REAL OBLIGATORIO
        self.tcn_predictor = None
        self._initialize_tcn_predictor()

        # Configurar filtros conservadores para evitar spam
        self.discord_notifier.configure_filters(
            min_trade_value_usd=12.0,          # Solo trades > $12
            min_pnl_percent_notify=2.0,        # Solo PnL > 2%
            max_notifications_per_hour=8,      # Max 8/hora
            max_notifications_per_day=40,      # Max 40/día
            suppress_similar_minutes=10,       # 10 min entre similares
            only_profitable_trades=False,      # Notificar pérdidas también
            emergency_only_mode=False          # Todas las prioridades
        )

        # Control de tiempo
        self.last_check_time = None
        self.last_balance_update = None

        # Configuración de trading
        self.monitoring_interval = 30  # segundos

        # Control de pausa/resume
        self.pause_trading = False
        self.pause_reason = None

        self.start_time = None
        self.last_heartbeat = None
        self.emergency_mode = False

        # Precios en tiempo real
        self.current_prices = {}

        # 🔧 CORREGIDO: Métricas unificadas con todas las claves necesarias
        self.metrics = {
            'uptime_seconds': 0,
            'total_checks': 0,
            'successful_checks': 0,
            'api_calls_count': 0,
            'error_count': 0,
            'last_error': None,
            'balance_updates': 0,
            'last_balance_update': None,
            'portfolio_snapshots': 0,
            'tcn_reports_sent': 0,
            'active_positions': 0,
            'session_pnl': 0.0,
            'total_trades': 0,
            'profitable_trades': 0
        }

    def _initialize_tcn_predictor(self):
        """🧠 Inicializar predictor TCN REAL obligatorio"""
        try:
            from tcn_definitivo_predictor import TCNDefinitivoPredictor
            self.tcn_predictor = TCNDefinitivoPredictor()
            print("🎯 Predictor TCN DEFINITIVO inicializado en constructor")
            print(f"   📊 Modelos cargados: {len(self.tcn_predictor.models)}")
            print(f"   🎯 Símbolos: {list(self.tcn_predictor.models.keys())}")
            return True
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: No se pudo inicializar TCN definitivo en constructor: {e}")
            print("🚨 SISTEMA REQUIERE TCN REAL - NO PUEDE CONTINUAR SIN ÉL")
            raise Exception(f"TCN REAL requerido pero falló en constructor: {e}")

    def _load_config(self) -> BinanceConfig:
        """⚙️ Cargar configuración desde variables de entorno"""
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')

        if not api_key or not secret_key:
            raise ValueError("❌ BINANCE_API_KEY y BINANCE_SECRET_KEY son requeridos en .env")

        return BinanceConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision'),
            environment=os.getenv('ENVIRONMENT', 'testnet')
        )

    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma para API de Binance"""
        return hmac.new(
            self.config.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def get_account_info(self) -> Optional[AccountInfo]:
        """💰 Obtener información completa de la cuenta de Binance"""
        try:
            params = {
                'timestamp': int(time.time() * 1000),
                'recvWindow': 10000
            }
            query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
            signature = self._generate_signature(query_string)

            headers = {
                'X-MBX-APIKEY': self.config.api_key
            }

            url = f"{self.config.base_url}/api/v3/account"
            full_url = f"{url}?{query_string}&signature={signature}"

            async with aiohttp.ClientSession() as session:
                async with session.get(full_url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Procesar balances
                        balances = {}
                        usdt_balance = 0.0

                        for balance in data.get('balances', []):
                            asset = balance['asset']
                            free = float(balance['free'])
                            locked = float(balance['locked'])
                            total = free + locked

                            if total > 0:
                                balances[asset] = {
                                    'free': free,
                                    'locked': locked,
                                    'total': total
                                }

                                if asset == 'USDT':
                                    usdt_balance = total

                        # Calcular valor total en USD (aproximado)
                        total_balance_usd = usdt_balance  # Base USDT

                        # Obtener precios para otros activos
                        for asset, balance_info in balances.items():
                            if asset != 'USDT' and balance_info['total'] > 0:
                                try:
                                    # Intentar obtener precio en USDT
                                    price_symbol = f"{asset}USDT"
                                    price = await self.get_current_price(price_symbol)
                                    if price > 0:
                                        total_balance_usd += balance_info['total'] * price
                                except:
                                    pass  # Si no se puede obtener precio, ignorar

                        self.metrics['api_calls_count'] += 1
                        self.metrics['balance_updates'] += 1
                        self.metrics['last_balance_update'] = datetime.now().isoformat()

                        return AccountInfo(
                            usdt_balance=usdt_balance,
                            total_balance_usd=total_balance_usd,
                            positions={},  # Implementar si necesitas posiciones específicas
                            balances=balances
                        )

                    else:
                        error_text = await response.text()
                        raise Exception(f"Error API Binance: {response.status} - {error_text}")

        except Exception as e:
            print(f"❌ Error obteniendo info de cuenta: {e}")
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            return None

    async def update_balance_from_binance(self):
        """🔄 Actualizar balance desde Binance"""
        try:
            account_info = await self.get_account_info()
            if account_info:
                old_balance = self.current_balance
                self.current_balance = account_info.usdt_balance
                self.account_info = account_info
                self.last_balance_update = datetime.now()

                # Solo mostrar cambio si es significativo
                if abs(old_balance - self.current_balance) > 0.01:
                    print(f"💰 Balance actualizado: ${old_balance:.2f} → ${self.current_balance:.2f}")

                return True
        except Exception as e:
            print(f"❌ Error actualizando balance: {e}")
            return False

        return False

    async def initialize(self):
        """🚀 Inicializar todos los componentes del sistema"""
        print("🚀 Iniciando Simple Professional Trading Manager...")
        self.status = TradingManagerStatus.STARTING

        try:
            # 1. Inicializar base de datos
            await self._initialize_database()

            # 2. Obtener balance inicial de Binance - ✅ NUEVO
            print("💰 Obteniendo balance de Binance...")
            await self.update_balance_from_binance()
            if self.current_balance == 0:
                print("⚠️ No se pudo obtener balance de Binance, usando valor por defecto")
                self.current_balance = 100.0  # Fallback mínimo si falla API

            # 3. ✅ NUEVO: Inicializar Professional Portfolio Manager
            print("💼 Inicializando Professional Portfolio Manager...")
            self.portfolio_manager = ProfessionalPortfolioManager(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                base_url=self.config.base_url
            )
            print("✅ Portfolio Manager inicializado")

            # 4. Inicializar Risk Manager
            await self._initialize_risk_manager()

            # ✅ 5. SINCRONIZAR POSICIONES EXISTENTES AL ARRANCAR
            await self._sync_positions_on_startup()

            # 6. Verificar conectividad
            await self._verify_connectivity()

            # 7. Configurar monitoreo
            await self._setup_monitoring()

            self.start_time = time.time()
            self.last_heartbeat = datetime.now()
            self.status = TradingManagerStatus.RUNNING

            # Log inicial
            if self.database:
                await self.database.log_event('INFO', 'SYSTEM', 'Simple Trading Manager inicializado correctamente')

            print("✅ Simple Professional Trading Manager iniciado correctamente")

        except Exception as e:
            self.status = TradingManagerStatus.ERROR
            print(f"❌ Error inicializando Trading Manager: {e}")
            if self.database:
                await self.database.log_event('ERROR', 'SYSTEM', f'Error inicializando: {e}')
            raise

    async def _sync_positions_on_startup(self):
        """🔄 Sincronizar estado de posiciones activas al arrancar."""
        print("🔄 Sincronizando posiciones existentes al inicio...")
        try:
            # Verificar que portfolio_manager esté inicializado
            if not self.portfolio_manager:
                print("   ⚠️ Portfolio manager no inicializado, saltando sincronización")
                return

            # Obtener el estado real del portafolio desde el exchange
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            if not snapshot or not snapshot.active_positions:
                print("   ✅ No se encontraron posiciones activas en el exchange.")
                return

            print(f"   🔍 Encontradas {len(snapshot.active_positions)} posiciones en el exchange. Sincronizando con DB...")

            # Verificar que database esté inicializado
            if not self.database:
                print("   ❌ Database no inicializado, no se puede sincronizar")
                return

            # Reconstruir el estado interno de self.active_positions
            synced_count = 0
            for portfolio_pos in snapshot.active_positions:
                order_id = portfolio_pos.order_id
                if not order_id:
                    print(f"      ⚠️ Advertencia: Posición para {portfolio_pos.symbol} no tiene ID de orden. Se omite.")
                    continue

                # Intento 1: Buscar por ID de orden (para trades nuevos y ya vinculados)
                db_trade = await self.database.get_trade_by_order_id(order_id)

                # Intento 2 (Fallback): Si no se encuentra, es un trade antiguo. Intentar vincularlo.
                if not db_trade:
                    print(f"      🔧 Intentando vincular trade antiguo para {portfolio_pos.symbol} con ID de orden {order_id}...")
                    unlinked_trade = await self.database.get_last_unlinked_buy_trade(portfolio_pos.symbol)
                    if unlinked_trade:
                        # Vincularlo en la DB para futuras ejecuciones
                        await self.database.update_trade_order_id(unlinked_trade['id'], order_id)
                        # Usar este trade para la sesión actual
                        db_trade = unlinked_trade
                    else:
                        print(f"      ❌ No se encontró un trade de compra sin vincular para {portfolio_pos.symbol}.")


                if db_trade:
                    # Reconstruir el objeto Position
                    reconstructed_pos = Position(
                        symbol=db_trade['symbol'],
                        side='BUY',
                        quantity=float(db_trade['quantity']),
                        entry_price=float(db_trade['entry_price']),
                        entry_time=pd.to_datetime(db_trade['entry_time']),
                        stop_loss=float(db_trade['stop_loss']) if db_trade['stop_loss'] else None,
                        take_profit=float(db_trade['take_profit']) if db_trade['take_profit'] else None,
                        trade_id=db_trade['id'], # ID interno de la DB
                        current_price=portfolio_pos.current_price,
                        pnl_percent=portfolio_pos.unrealized_pnl_percent,
                        pnl_usd=portfolio_pos.unrealized_pnl_usd,
                    )

                    self.active_positions[order_id] = reconstructed_pos
                    synced_count += 1
                    print(f"      ✅ Sincronizada posición para {reconstructed_pos.symbol} con ID de orden {order_id}.")
                else:
                    print(f"      ⚠️ Advertencia: Posición con ID de orden {order_id} existe en el exchange pero no se encontró un trade correspondiente en la DB.")

            print(f"   👍 Sincronización completa. {synced_count} posiciones activas cargadas en el estado del bot.")

        except Exception as e:
            print(f"❌ Error fatal durante la sincronización de posiciones: {e}")
            # Decidimos no continuar si no podemos sincronizar el estado, para evitar operaciones incorrectas.
            raise e

    def _get_positions_for_symbol(self, symbol: str) -> List[Position]:
        """Helper: Obtiene todas las posiciones activas para un símbolo específico."""
        # ✅ CORREGIDO: Usar portfolio_manager.position_registry en lugar de active_positions
        return [pos for pos in self.portfolio_manager.position_registry.values() if pos.symbol == symbol]

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

    async def _verify_connectivity(self):
        """🔗 Verificar conectividad con APIs"""
        print("🔗 Verificando conectividad...")

        # Test Binance API simple
        try:
            price = await self.get_current_price('BTCUSDT')
            if price > 0:
                print(f"✅ Conectividad Binance OK - BTC: ${price:.2f}")
            else:
                raise Exception("No se pudo obtener precio de test")
        except Exception as e:
            raise Exception(f"Error conectividad Binance: {e}")

    async def get_current_price(self, symbol: str) -> float:
        """💲 Obtener precio actual de un símbolo"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol}
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except Exception as e:
            print(f"❌ Error obteniendo precio {symbol}: {e}")
        return 0.0

    async def _setup_monitoring(self):
        """🛠️ Configurar tareas de monitoreo del sistema"""
        print("🛠️ Configurando sistema de monitoreo...")

        try:
            # Monitoreo de latido (heartbeat)
            asyncio.create_task(self._heartbeat_monitor())
            print("   ✅ Heartbeat monitor iniciado")

            # ✅ NUEVO: Monitoreo de posiciones con trailing stops cada 30 segundos
            async def _position_monitor_loop():
                """Loop de monitoreo de posiciones con trailing stops - ALTA FRECUENCIA"""
                while self.status == TradingManagerStatus.RUNNING:
                    try:
                        await self._position_monitor()
                        await asyncio.sleep(30)  # ✅ ALTA FRECUENCIA: Cada 30 segundos para TCN
                    except Exception as e:
                        print(f"❌ Error en monitor de posiciones: {e}")
                        await asyncio.sleep(60)

            asyncio.create_task(_position_monitor_loop())
            print("   ✅ Position monitor con trailing stops iniciado")

            # Recolector de métricas
            asyncio.create_task(self._metrics_collector())
            print("   ✅ Metrics collector iniciado")

        except Exception as e:
            print(f"❌ Error configurando monitoreo: {e}")
            raise

    async def run(self):
        """🎯 Ejecutar loop principal de trading

        ✅ ALTA FRECUENCIA TCN: Actualizaciones frecuentes para trading en tiempo real
        - Bucle principal: cada 60 segundos (genera señales TCN, updates balance)
        - Monitor de posiciones: cada 30 segundos (máxima frecuencia TCN)
        - Reporte TCN: cada 5 minutos (completo)
        - PRIORIDAD: Datos frescos > logs limpios
        """
        print("🎯 Iniciando loop principal de trading...")

        # ✅ NUEVO: Contador de errores por ciclo para reset automático
        consecutive_errors = 0
        max_consecutive_errors = 5  # Reducido de 10 a 5
        error_reset_time = 300  # 5 minutos para reset de errores
        last_successful_cycle = datetime.now()

        while self.status == TradingManagerStatus.RUNNING:
            try:
                loop_start_time = datetime.now()

                # Verificar si está pausado
                if self.pause_trading:
                    await self._handle_pause_state()
                    await asyncio.sleep(10)
                    continue

                # ✅ MEJORADO: Reset automático de contador de errores
                if consecutive_errors > 0:
                    time_since_success = (datetime.now() - last_successful_cycle).total_seconds()
                    if time_since_success > error_reset_time:
                        consecutive_errors = 0
                        print(f"🔄 Reset automático de errores después de {error_reset_time/60:.1f} minutos")

                # ✅ NUEVO: Generar reporte TCN cada 5 minutos
                await self._generate_tcn_report_if_needed()

                # ✅ MEJORADO: Mostrar información profesional en tiempo real
                await self._display_professional_info()

                # 1. Actualizar balance cada 5 minutos
                time_since_balance_update = None
                if self.last_balance_update:
                    time_since_balance_update = (datetime.now() - self.last_balance_update).total_seconds()

                if not self.last_balance_update or (time_since_balance_update is not None and time_since_balance_update > 300):  # 5 minutos
                    print("🔄 Actualizando balance desde Binance...")
                    await self.update_balance_from_binance()

                # 2. Obtener precios actuales
                prices = await self._get_current_prices()
                self.current_prices = prices

                # ✅ NUEVO: Actualizar PnL de posiciones existentes
                await self._update_positions_pnl(prices)

                # 3. Generar señales usando modelo TCN REAL
                signals = await self._generate_tcn_signals(prices)

                # 4. Procesar cada señal
                for symbol, signal_data in signals.items():
                    await self._process_signal(symbol, signal_data)

                # 5. Actualizar métricas
                await self._update_metrics()

                # 6. Guardar estado en DB
                await self._save_periodic_metrics()

                # ✅ MEJORADO: Marcar ciclo exitoso
                consecutive_errors = 0
                last_successful_cycle = datetime.now()

                # ✅ NUEVO: Mostrar resumen cada ciclo
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                print(f"⏱️ Ciclo completado en {loop_duration:.1f}s")

                # 7. Esperar siguiente ciclo
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                consecutive_errors += 1
                await self._handle_error_improved(e, consecutive_errors, max_consecutive_errors)

                # ✅ MEJORADO: Pausa adaptativa basada en el número de errores
                if consecutive_errors <= 2:
                    await asyncio.sleep(10)  # Pausa corta para errores menores
                elif consecutive_errors <= 4:
                    await asyncio.sleep(20)  # Pausa media
                else:
                    await asyncio.sleep(30)  # Pausa larga solo para errores críticos

    async def _handle_error_improved(self, error: Exception, consecutive_errors: int, max_consecutive_errors: int):
        """❌ Manejo mejorado de errores del sistema"""
        error_msg = f"Error en loop principal: {error}"
        error_type = type(error).__name__

        print(f"❌ {error_msg} (Error #{consecutive_errors})")
        print(f"🔍 Tipo de error: {error_type}")

        if self.database:
            await self.database.log_event('ERROR', 'SYSTEM', f"{error_msg} | Consecutivos: {consecutive_errors}")

        # ✅ MEJORADO: Manejo inteligente de diferentes tipos de errores
        if "timeout" in str(error).lower() or "connection" in str(error).lower():
            print("🌐 Error de conectividad detectado - continuando con pausa corta")
            return

        if "rate limit" in str(error).lower():
            print("⏳ Rate limit detectado - pausando 60 segundos")
            await asyncio.sleep(60)
            return

        # ✅ MEJORADO: Solo pausar después de muchos errores críticos
        if consecutive_errors >= max_consecutive_errors:
            print(f"🚨 {consecutive_errors} errores consecutivos - pausando sistema temporalmente")
            await self.pause_trading_with_reason(f"Demasiados errores consecutivos ({consecutive_errors})")

            # ✅ NUEVO: Auto-reanudar después de 10 minutos
            await asyncio.sleep(600)  # 10 minutos
            if self.status == TradingManagerStatus.PAUSED:
                print("🔄 Auto-reanudando trading después de pausa por errores")
                await self.resume_trading()

        # Actualizar métricas de errores
        self.metrics['error_count'] += 1
        self.metrics['last_error'] = {
            'type': error_type,
            'message': str(error)[:100],
            'timestamp': datetime.now().isoformat()
        }

    async def _generate_tcn_report_if_needed(self):
        """📊 Generar reporte TCN cada 5 minutos"""
        try:
            now = datetime.now()

            # Verificar si es hora de generar reporte (cada 5 minutos)
            should_generate = False

            if self.last_tcn_report_time is None:
                should_generate = True
            else:
                time_since_last = (now - self.last_tcn_report_time).total_seconds()
                if time_since_last >= 300:  # 5 minutos
                    should_generate = True

            if should_generate:
                print("📊 Generando reporte TCN profesional...")

                # Verificar que portfolio_manager esté inicializado
                if not self.portfolio_manager:
                    print("⚠️ Portfolio manager no inicializado para reporte TCN")
                    return

                # Obtener snapshot del portafolio
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                self.last_portfolio_snapshot = snapshot
                self.metrics['portfolio_snapshots'] += 1

                # Generar reporte TCN
                tcn_report = self.portfolio_manager.format_tcn_style_report(snapshot)

                # ✅ NUEVO: Agregar reporte de modelos TCN
                tcn_models_report = await self._generate_tcn_models_section()

                # ✅ NUEVO: Agregar reporte de diversificación
                diversification_report = await self._generate_diversification_section(snapshot)

                # Combinar reportes
                full_report = tcn_report + tcn_models_report + diversification_report

                # Mostrar en consola
                print("\n" + "="*80)
                print("🎯 REPORTE TCN PROFESSIONAL")
                print("="*80)
                print(full_report)
                print("="*80)

                # Enviar a Discord si está configurado
                if hasattr(self, 'discord_notifier'):
                    await self._send_tcn_discord_notification(full_report)
                    self.metrics['tcn_reports_sent'] += 1

                self.last_tcn_report_time = now

        except Exception as e:
            print(f"❌ Error generando reporte TCN: {e}")

    async def _generate_tcn_models_section(self) -> str:
        """🤖 Generar sección de estado de modelos TCN"""
        try:
            models_section = f"""

🤖 **ESTADO DE MODELOS TCN**
"""

            # ✅ CORREGIDO: Inicializar predictor TCN si no existe (igual que en _generate_tcn_signals)
            if not hasattr(self, 'tcn_predictor'):
                try:
                    from tcn_definitivo_predictor import TCNDefinitivoPredictor
                    self.tcn_predictor = TCNDefinitivoPredictor()
                    print("🎯 Predictor TCN DEFINITIVO inicializado para reporte Discord")
                except Exception as e:
                    models_section += f"❌ **Error inicializando predictor**: {str(e)[:50]}...\n"
                    return models_section

            # Obtener precios actuales para las predicciones (reutilizar si ya los tenemos)
            current_prices = {}
            for symbol in self.symbols:
                try:
                    price = await self.get_current_price(symbol)
                    if price > 0:
                        current_prices[symbol] = price
                except Exception as e:
                    print(f"⚠️ Error obteniendo precio {symbol} para Discord: {e}")

            if not current_prices:
                models_section += "⚠️ **Sin datos de precios para análisis**\n"
                return models_section

            # ✅ NUEVO: Analizar contexto de mercado para Discord
            try:
                market_context = await self._analyze_market_context(current_prices)
                regime = market_context['regime']
                regime_confidence = market_context['confidence']
                market_score = market_context['score']
                volatility = market_context['volatility_level']

                # Emoji según régimen
                regime_emoji = {
                    'BULLISH': '🟢',
                    'BEARISH': '🔴',
                    'NEUTRAL': '🟡'
                }.get(regime, '⚪')

                # Emoji según volatilidad
                vol_emoji = {
                    'HIGH': '⚡',
                    'MEDIUM': '📊',
                    'LOW': '😴'
                }.get(volatility, '📊')

                models_section += f"""
🌍 **RÉGIMEN DE MERCADO**
{regime_emoji} **{regime}** (Conf: {regime_confidence:.1%}) {vol_emoji} Vol: {volatility}
📊 Score: {market_score:+.3f} | 🔗 Correlación: {market_context.get('correlation_strength', 0):.2f}

"""
            except Exception as e:
                models_section += f"⚠️ **Contexto de mercado**: Error al analizar ({str(e)[:30]}...)\n\n"

            # Generar predicciones para cada símbolo
            for symbol in self.symbols:
                try:
                    if symbol not in current_prices:
                        models_section += f"❌ **{symbol}**: Sin precio disponible\n"
                        continue

                    # Obtener predicción del modelo
                    prediction = None
                    if hasattr(self.tcn_predictor, 'predict_symbol'):
                        prediction = self.tcn_predictor.predict_symbol(symbol)

                    if prediction:
                        signal = prediction['signal']
                        confidence = prediction['confidence']
                        probabilities = prediction.get('probabilities', {})

                        # Emoji según la señal
                        signal_emoji = {
                            'BUY': '🟢',
                            'SELL': '🔴',
                            'HOLD': '🟡'
                        }.get(signal, '⚪')

                        # Formato de confianza con color
                        conf_status = "🔥" if confidence >= 0.80 else "✅" if confidence >= 0.70 else "⚠️"

                        models_section += f"{signal_emoji} **{symbol}**: {signal} ({conf_status} {confidence:.1%})\n"

                        # Mostrar distribución de probabilidades si están disponibles
                        if probabilities:
                            buy_prob = probabilities.get('BUY', 0)
                            hold_prob = probabilities.get('HOLD', 0)
                            sell_prob = probabilities.get('SELL', 0)
                            models_section += f"   📊 BUY:{buy_prob:.1%} | HOLD:{hold_prob:.1%} | SELL:{sell_prob:.1%}\n"

                        # Precio actual
                        current_price = current_prices[symbol]
                        models_section += f"   💰 Precio: ${current_price:,.4f}\n"

                    else:
                        models_section += f"❌ **{symbol}**: Error en predicción\n"

                except Exception as e:
                    models_section += f"❌ **{symbol}**: Error ({str(e)[:30]}...)\n"
                    continue

            # Agregar timestamp del análisis
            models_section += f"\n⏰ Análisis: {datetime.now().strftime('%H:%M:%S')}\n"

            return models_section

        except Exception as e:
            print(f"⚠️ Error generando sección de modelos TCN: {e}")
            return f"\n🤖 **MODELOS TCN:** Error al generar análisis ({str(e)[:30]}...)\n"

    async def _generate_diversification_section(self, snapshot) -> str:
        """🎯 Generar sección de diversificación para el reporte"""
        try:
            # Convertir posiciones a formato PortfolioPosition
            current_positions = []
            for pos in snapshot.active_positions:
                portfolio_pos = PortfolioPosition(
                    symbol=pos.symbol,
                    quantity=pos.quantity,  # ✅ CORREGIDO: usar 'quantity' en lugar de 'size'
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    value_usd=pos.market_value,  # ✅ CORREGIDO: usar 'market_value' en lugar de 'value_usd'
                    percentage=(pos.market_value / snapshot.total_balance_usd * 100) if snapshot.total_balance_usd > 0 else 0,
                    category=self.diversification_manager.diversification_config['SYMBOL_CATEGORIES'].get(pos.symbol, 'UNKNOWN'),
                    age_minutes=int((datetime.now() - pos.entry_time).total_seconds() / 60),
                    pnl_percent=pos.unrealized_pnl_percent  # ✅ CORREGIDO: usar 'unrealized_pnl_percent'
                )
                current_positions.append(portfolio_pos)

            # Generar análisis de diversificación
            analysis = await self.diversification_manager.analyze_portfolio_diversification(current_positions)

            # Crear sección del reporte
            diversification_section = f"""

🎯 **ANÁLISIS DE DIVERSIFICACIÓN**
📊 **Score:** {analysis.diversification_score:.1f}/100
"""

            # Concentraciones por símbolo
            if analysis.symbol_concentrations:
                diversification_section += "\n**📈 CONCENTRACIÓN POR SÍMBOLO:**\n"
                for symbol, conc in sorted(analysis.symbol_concentrations.items(), key=lambda x: x[1], reverse=True):
                    status = "🔴" if conc > 40 else "🟡" if conc > 35 else "🟢"
                    diversification_section += f"{status} {symbol}: {conc:.1f}%\n"

            # Concentraciones por categoría
            if analysis.category_concentrations:
                diversification_section += "\n**🏷️ CONCENTRACIÓN POR CATEGORÍA:**\n"
                for category, conc in sorted(analysis.category_concentrations.items(), key=lambda x: x[1], reverse=True):
                    status = "🔴" if conc > 60 else "🟢"
                    diversification_section += f"{status} {category}: {conc:.1f}%\n"

            # Alertas importantes
            if analysis.over_concentrated_symbols or analysis.over_concentrated_categories:
                diversification_section += "\n**⚠️ ALERTAS:**\n"
                for symbol in analysis.over_concentrated_symbols:
                    conc = analysis.symbol_concentrations[symbol]
                    diversification_section += f"🚨 {symbol} sobre-concentrado: {conc:.1f}%\n"

                for category in analysis.over_concentrated_categories:
                    conc = analysis.category_concentrations[category]
                    diversification_section += f"🚨 Categoría {category} sobre-concentrada: {conc:.1f}%\n"

            # Recomendaciones principales (máximo 3)
            if analysis.recommendations and len(analysis.recommendations) > 0:
                diversification_section += "\n**💡 RECOMENDACIONES:**\n"
                for rec in analysis.recommendations[:3]:
                    diversification_section += f"• {rec}\n"

            return diversification_section

        except Exception as e:
            print(f"⚠️ Error generando sección de diversificación: {e}")
            return "\n🎯 **DIVERSIFICACIÓN:** Error al generar análisis\n"

    async def _display_professional_info(self):
        """📺 Mostrar información profesional mejorada"""
        try:
            # Verificar que start_time esté inicializado
            if not self.start_time:
                print("⚠️ Start time no inicializado")
                return

            uptime_minutes = (time.time() - self.start_time) / 60

            # Obtener snapshot actualizado del portafolio
            if self.portfolio_manager:
                try:
                    current_snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                    self.last_portfolio_snapshot = current_snapshot
                except Exception as e:
                    print(f"⚠️ Error obteniendo snapshot: {e}")
                    current_snapshot = self.last_portfolio_snapshot
            else:
                current_snapshot = None

            print("🔥" * 80)
            print(f"🕐 {datetime.now().strftime('%H:%M:%S')} | ⏱️ Uptime: {uptime_minutes:.1f}min | 🎯 Trading Manager Professional")

            if current_snapshot:
                print(f"💼 PORTAFOLIO: ${current_snapshot.total_balance_usd:.2f} USDT")
                print(f"💰 USDT Libre: ${current_snapshot.free_usdt:.2f}")

                pnl_sign = "+" if current_snapshot.total_unrealized_pnl >= 0 else ""
                pnl_emoji = "📈" if current_snapshot.total_unrealized_pnl >= 0 else "📉"
                print(f"{pnl_emoji} PnL No Realizado: ${pnl_sign}{current_snapshot.total_unrealized_pnl:.2f}")

                print(f"🎯 Posiciones Activas: {current_snapshot.position_count}/{current_snapshot.max_positions}")

                # ✅ MEJORADO: Mostrar posiciones con información de múltiples entradas
                if current_snapshot.active_positions:
                    print("📈 POSICIONES:")

                    # Agrupar posiciones por símbolo
                    positions_by_symbol = {}
                    for pos in current_snapshot.active_positions:
                        if pos.symbol not in positions_by_symbol:
                            positions_by_symbol[pos.symbol] = []
                        positions_by_symbol[pos.symbol].append(pos)

                    for symbol, positions in positions_by_symbol.items():
                        if len(positions) == 1:
                            # Una sola posición
                            pos = positions[0]
                            pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                            pnl_color = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"

                            print(f"   {pnl_color} {symbol}: ${pos.entry_price:,.4f} → ${pos.current_price:,.4f} ({pnl_sign}{pos.unrealized_pnl_percent:.2f}% = ${pnl_sign}{pos.unrealized_pnl_usd:.2f})")
                        else:
                            # Múltiples posiciones - mostrar resumen + total
                            total_pnl = sum(p.unrealized_pnl_usd for p in positions)
                            total_value = sum(p.market_value for p in positions)
                            pnl_sign = "+" if total_pnl >= 0 else ""
                            pnl_color = "🟢" if total_pnl >= 0 else "🔴"

                            print(f"   {pnl_color} {symbol} ({len(positions)} pos): ${total_value:.2f} (${pnl_sign}{total_pnl:.2f})")

                            # Mostrar detalle de cada posición individual
                            for i, pos in enumerate(positions, 1):
                                pos_pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                                duration_str = f"{pos.duration_minutes}min" if pos.duration_minutes < 60 else f"{pos.duration_minutes//60}h"
                                print(f"      #{i}: {pos.quantity:.6f} @ ${pos.entry_price:,.2f} ({pos_pnl_sign}{pos.unrealized_pnl_percent:.1f}%) {duration_str}")

                # Mostrar principales activos
                print("🪙 ACTIVOS PRINCIPALES:")
                main_assets = [asset for asset in current_snapshot.all_assets
                             if asset.usd_value >= 1.0 and asset.symbol != 'USDT'][:5]

                for asset in main_assets:
                    print(f"   🪙 {asset.symbol}: {asset.total:.6f} (${asset.usd_value:.2f})")

                if current_snapshot.free_usdt > 0:
                    print(f"   💵 USDT: ${current_snapshot.free_usdt:.2f}")
            else:
                print(f"💼 PORTAFOLIO: ${self.current_balance:.2f} USDT")
                print(f"💰 USDT Libre: ${self.current_balance:.2f}")
                print(f"📈 PnL No Realizado: $+0.00")
                print(f"🎯 Posiciones Activas: 0/5")
                print("📈 POSICIONES: Ninguna")

            # Mostrar métricas
            print(f"📊 MÉTRICAS: API calls: {self.metrics.get('api_calls_count', 0)} | Errores: {self.metrics.get('error_count', 0)} | Reportes TCN: {self.metrics.get('tcn_reports_sent', 0)}")

            print("🔥" * 80)

        except Exception as e:
            print(f"❌ Error en display: {e}")

    async def _send_tcn_discord_notification(self, tcn_report: str):
        """💬 Enviar reporte TCN a Discord"""
        try:
            if not tcn_report or len(tcn_report.strip()) == 0:
                print("⚠️ Reporte TCN vacío, saltando Discord")
                return

            from smart_discord_notifier import NotificationPriority

            # Enviar reporte completo con prioridad alta
            result = await self.discord_notifier.send_system_notification(
                tcn_report,
                NotificationPriority.HIGH
            )

            if result and hasattr(result, 'status_code'):
                if result.status_code == 204:
                    print("✅ Discord: Reporte TCN enviado (204 OK)")
                elif result.status_code == 200:
                    print("✅ Discord: Reporte TCN enviado (200 OK)")
                else:
                    print(f"⚠️ Discord: Status {result.status_code}")

        except Exception as e:
            print(f"❌ Discord error: {e}")

    async def _update_positions_pnl(self, prices: Dict[str, float]):
        """📈 Actualizar PnL de todas las posiciones activas"""
        for symbol, position in self.active_positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.current_price = current_price

                # Calcular PnL actualizado
                if position.side == 'BUY':
                    position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                else:
                    position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

                position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)

    async def _get_current_prices(self) -> Dict[str, float]:
        """💰 Obtener precios actuales de todos los símbolos"""
        prices = {}

        print("🔄 Obteniendo precios actuales...")

        for symbol in self.symbols:
            try:
                price = await self.get_current_price(symbol)
                if price > 0:
                    prices[symbol] = price
                    self.metrics['successful_checks'] += 1
                    print(f"   ✅ {symbol}: ${price:.4f}")
                else:
                    print(f"   ❌ {symbol}: Sin precio")

                self.metrics['total_checks'] += 1

            except Exception as e:
                print(f"   ❌ Error obteniendo precio {symbol}: {e}")
                self.metrics['error_count'] += 1
                if self.database:
                    await self.database.log_event('ERROR', 'MARKET_DATA', f'Error precio {symbol}: {e}', symbol)

        self.last_check_time = datetime.now()
        self.metrics['api_calls_count'] += len(self.symbols)

        return prices

    async def _generate_tcn_signals(self, prices: Dict[str, float]) -> Dict:
        """
        🧠 Genera señales de trading (BUY/SELL) basadas en la confianza del modelo TCN.
        ---
        CORREGIDO: Asegura que tanto las señales BUY como SELL se añadan a la cola de procesamiento.
        Versión híbrida que combina la simplicidad de Gemini con la funcionalidad actual.
        """
        signals = {}

        # Inicializar last_signals si no existe
        if not hasattr(self, 'last_signals'):
            self.last_signals = {}

        # ✅ VERIFICAR que TCN está inicializado (debe estar desde constructor)
        if not hasattr(self, 'tcn_predictor') or self.tcn_predictor is None:
            print("❌ ERROR CRÍTICO: TCN no está inicializado")
            print("🚨 SISTEMA REQUIERE TCN REAL - REINTENTANDO INICIALIZACIÓN")
            self._initialize_tcn_predictor()

        # ✅ NUEVO: Analizar contexto de mercado como capa de seguridad
        try:
            market_context = await self._analyze_market_context(prices)
            print(f"🌍 CONTEXTO DE MERCADO: {market_context['regime']} (Score: {market_context['score']:.2f}, Confianza: {market_context['confidence']:.1%})")
        except Exception as e:
            print(f"⚠️ Error analizando contexto de mercado: {e}")
            # Usar contexto neutral por defecto
            market_context = {
                'regime': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.0,
                'market_fear_factor': 0.5,
                'trend_strength': 0.0,
                'volatility_level': 'MEDIUM'
            }

        # Obtener umbral de confianza
        threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70')) * 100  # Convertir a porcentaje

        # ✅ VERSIÓN HÍBRIDA: Combinar lógica actual con simplicidad de Gemini
        for symbol in self.symbols:
            current_price = prices.get(symbol)
            if not current_price:
                continue

            try:
                print(f"🔍 Analizando {symbol} con modelo TCN...")

                # Generar predicción TCN
                prediction = None
                if self.tcn_predictor and hasattr(self.tcn_predictor, 'predict_symbol'):
                    prediction = self.tcn_predictor.predict_symbol(symbol)
                else:
                    # Fallback - sin predicción
                    print(f"  ❌ TCN predictor no disponible para {symbol}")
                    continue

                if not prediction:
                    print(f"  ❌ No se pudo generar predicción para {symbol}")
                    continue

                signal = prediction['signal']
                confidence_level = prediction['confidence'] * 100  # Convertir a porcentaje

                # ✅ NUEVO: Aplicar filtro de contexto de mercado
                filtered_signal, context_reason = self._apply_market_context_filter(
                    signal, confidence_level, market_context, symbol
                )

                if filtered_signal != signal:
                    print(f"🛡️ FILTRO DE CONTEXTO aplicado en {symbol}: {signal} → {filtered_signal} ({context_reason})")
                    signal = filtered_signal

                # ✅ CORREGIDO: Procesar señales válidas independientemente de si han cambiado
                # Actualizar registro de última señal
                self.last_signals[symbol] = signal
                print(f"💡 Señal TCN para {symbol}: {signal} (Confianza: {confidence_level:.2f}%) (Umbral: {threshold:.1f}%)")

                # 💡 **CORRECCIÓN DE LÓGICA CRÍTICA** - Procesar señales con confianza suficiente
                # Esta lógica asegura que AMBAS señales (BUY y SELL) sean correctamente procesadas
                if (signal == 'BUY' and confidence_level >= threshold) or signal == 'SELL':
                    log_emoji = "📈" if signal == "BUY" else "📉"
                    log_action = "COMPRA" if signal == "BUY" else "VENTA"
                    print(f"{log_emoji} Oportunidad de {log_action} detectada para {symbol}. Preparando para posible operación.")

                    # Verificaciones adicionales específicas para BUY
                    if signal == 'BUY':
                        # ✅ CORREGIDO: Verificar límite de 3 posiciones por símbolo
                        existing_positions = self._get_positions_for_symbol(symbol)
                        if len(existing_positions) >= 3:
                            print(f"  ⏸️ Señal BUY ignorada - Máximo 3 posiciones alcanzado para {symbol} (actual: {len(existing_positions)})")
                            continue

                        # Verificar balance suficiente
                        min_position_value = 11.0  # Valor por defecto si risk_manager no está disponible
                        if self.risk_manager and hasattr(self.risk_manager, 'limits'):
                            min_position_value = self.risk_manager.limits.min_position_value_usdt

                        if self.current_balance < min_position_value:
                            print(f"  💰 Señal BUY generada (solo análisis) - Balance insuficiente para trade")
                            continue

                    elif signal == 'SELL':
                        # Verificar si tenemos posición para vender
                        existing_positions = self._get_positions_for_symbol(symbol)
                        if len(existing_positions) == 0:
                            print(f"  ⏸️ Señal SELL ignorada - No hay posición que vender en {symbol}")
                            continue

                    # ✅ SEÑAL VÁLIDA - Este bloque ahora se ejecuta para ambas señales
                    signals[symbol] = {
                        'signal': signal,
                        'price': current_price,
                        'confidence': confidence_level,
                        'timestamp': datetime.utcnow(),
                        'current_price': current_price,
                        'reason': 'TCN_MODEL_PREDICTION',
                        'available_usdt': self.current_balance,
                        'probabilities': prediction.get('probabilities', {}),
                        'balance_sufficient': self.current_balance >= (self.risk_manager.limits.min_position_value_usdt if self.risk_manager and self.risk_manager.limits else 11.0),
                        # ✅ NUEVO: Información del contexto de mercado
                        'market_context': market_context,
                        'context_filter_applied': filtered_signal != prediction['signal']
                    }
                    print(f"  ✅ SEÑAL AÑADIDA A LA COLA: {symbol} {signal} ({confidence_level:.1f}%)")

            except Exception as e:
                print(f"  ❌ Error procesando {symbol}: {e}")
                continue

        if signals:
            print(f"🎯 Total señales TCN generadas: {len(signals)}")
        else:
            print("📊 No se generaron señales TCN válidas en este ciclo")

        return signals

    async def _analyze_market_context(self, prices: Dict[str, float]) -> Dict:
        """
        🌍 Analizar contexto general de mercado como capa de seguridad adicional

        Evalúa múltiples factores para determinar el régimen de mercado:
        - Tendencia de dominancia de BTC
        - Correlación entre activos principales
        - Índice de miedo/codicia (Fear & Greed simulado)
        - Volatilidad del mercado
        - Fortaleza relativa de las altcoins vs BTC

        Returns:
            Dict con régimen, score, confianza y factores de riesgo
        """
        try:
            print("🔍 Analizando contexto de mercado...")

            # Obtener datos históricos para análisis de tendencia
            market_data = {}
            for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
                try:
                    # Obtener últimas 100 velas de 1h para análisis macro
                    url = f"https://api.binance.com/api/v3/klines"
                    params = {
                        'symbol': symbol,
                        'interval': '1h',
                        'limit': 100
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                klines = await response.json()
                                market_data[symbol] = [float(k[4]) for k in klines]  # Precios de cierre
                except Exception as e:
                    print(f"  ⚠️ Error obteniendo datos para {symbol}: {e}")
                    market_data[symbol] = [prices.get(symbol, 0)] * 100  # Fallback

            # 1. ANÁLISIS DE TENDENCIA DOMINANTE (BTC como líder)
            btc_prices = market_data.get('BTCUSDT', [])
            if len(btc_prices) >= 50:
                # Tendencia de corto y largo plazo
                short_avg = sum(btc_prices[-10:]) / 10  # Últimas 10 horas
                medium_avg = sum(btc_prices[-24:]) / 24  # Últimas 24 horas
                long_avg = sum(btc_prices[-50:]) / 50   # Últimas 50 horas

                current_btc = btc_prices[-1]

                # Score de tendencia (-1 = muy bearish, +1 = muy bullish)
                trend_score = 0
                trend_score += 0.4 * ((current_btc - short_avg) / short_avg)   # 40% peso corto plazo
                trend_score += 0.35 * ((current_btc - medium_avg) / medium_avg) # 35% peso medio plazo
                trend_score += 0.25 * ((current_btc - long_avg) / long_avg)     # 25% peso largo plazo

                # Normalizar a rango [-1, 1]
                trend_score = max(-1, min(1, trend_score * 10))
            else:
                trend_score = 0

            # 2. ANÁLISIS DE CORRELACIÓN (Mercado unificado vs disperso)
            correlation_strength = 0
            try:
                if len(market_data) >= 2:
                    # Calcular correlación entre BTC, ETH y BNB
                    import numpy as np

                    btc_returns = np.diff(market_data['BTCUSDT']) / market_data['BTCUSDT'][:-1]
                    eth_returns = np.diff(market_data['ETHUSDT']) / market_data['ETHUSDT'][:-1]
                    bnb_returns = np.diff(market_data['BNBUSDT']) / market_data['BNBUSDT'][:-1]

                    # Correlación promedio
                    btc_eth_corr = np.corrcoef(btc_returns, eth_returns)[0, 1]
                    btc_bnb_corr = np.corrcoef(btc_returns, bnb_returns)[0, 1]
                    eth_bnb_corr = np.corrcoef(eth_returns, bnb_returns)[0, 1]

                    correlation_strength = np.mean([btc_eth_corr, btc_bnb_corr, eth_bnb_corr])
                    correlation_strength = max(0, min(1, correlation_strength))  # Clamp [0,1]
            except Exception as e:
                print(f"  ⚠️ Error calculando correlación: {e}")
                correlation_strength = 0.5  # Neutral

            # 3. ÍNDICE DE MIEDO/CODICIA SIMULADO
            try:
                # Basado en volatilidad y momentum
                btc_volatility = np.std(np.diff(btc_prices[-24:]) / btc_prices[-25:-1]) if len(btc_prices) >= 25 else 0

                # Convertir volatilidad a fear factor (alta vol = miedo, baja vol = codicia)
                fear_factor = min(1, btc_volatility * 100)  # Escalar volatilidad
                fear_factor = max(0, min(1, fear_factor))   # Clamp [0,1]
            except Exception as e:
                fear_factor = 0.5  # Neutral

            # 4. FORTALEZA DE ALTCOINS vs BTC (Dominancia)
            altcoin_strength = 0
            try:
                # Comparar rendimiento de ETH y BNB vs BTC en últimas 24h
                if len(market_data['BTCUSDT']) >= 24 and len(market_data['ETHUSDT']) >= 24:
                    btc_change_24h = (btc_prices[-1] - btc_prices[-24]) / btc_prices[-24]
                    eth_change_24h = (market_data['ETHUSDT'][-1] - market_data['ETHUSDT'][-24]) / market_data['ETHUSDT'][-24]
                    bnb_change_24h = (market_data['BNBUSDT'][-1] - market_data['BNBUSDT'][-24]) / market_data['BNBUSDT'][-24]

                    # Altcoins superando a BTC = bullish, underperformance = bearish
                    eth_relative = eth_change_24h - btc_change_24h
                    bnb_relative = bnb_change_24h - btc_change_24h

                    altcoin_strength = (eth_relative + bnb_relative) / 2
                    altcoin_strength = max(-0.5, min(0.5, altcoin_strength))  # Clamp [-0.5, 0.5]
            except Exception as e:
                altcoin_strength = 0

            # 5. SCORE COMPUESTO FINAL
            # Pesos: Tendencia BTC (50%), Correlación (20%), Fear (15%), Altcoins (15%)
            composite_score = (
                0.50 * trend_score +           # Tendencia dominante
                0.20 * (correlation_strength - 0.5) * 2 +  # Correlación (centralizada en 0)
                0.15 * (0.5 - fear_factor) * 2 +           # Fear invertido (bajo miedo = bullish)
                0.15 * altcoin_strength * 2                # Fortaleza altcoins
            )

            # 6. CLASIFICACIÓN DE RÉGIMEN
            if composite_score > 0.15:
                regime = 'BULLISH'
                confidence = min(0.95, 0.5 + abs(composite_score))
            elif composite_score < -0.15:
                regime = 'BEARISH'
                confidence = min(0.95, 0.5 + abs(composite_score))
            else:
                regime = 'NEUTRAL'
                confidence = 0.5 + (0.15 - abs(composite_score)) / 0.15 * 0.3

            # 7. NIVEL DE VOLATILIDAD
            if fear_factor > 0.7:
                volatility_level = 'HIGH'
            elif fear_factor < 0.3:
                volatility_level = 'LOW'
            else:
                volatility_level = 'MEDIUM'

            context = {
                'regime': regime,
                'score': composite_score,
                'confidence': confidence,
                'market_fear_factor': fear_factor,
                'trend_strength': abs(trend_score),
                'volatility_level': volatility_level,
                'correlation_strength': correlation_strength,
                'altcoin_strength': altcoin_strength,
                'btc_trend_score': trend_score,
                'components': {
                    'btc_trend': trend_score,
                    'correlation': correlation_strength,
                    'fear_greed': 1 - fear_factor,  # Invertir para mostrar greed
                    'altcoin_performance': altcoin_strength
                }
            }

            print(f"  📊 Tendencia BTC: {trend_score:.3f}")
            print(f"  🔗 Correlación mercado: {correlation_strength:.3f}")
            print(f"  😨 Factor miedo: {fear_factor:.3f}")
            print(f"  🚀 Fortaleza altcoins: {altcoin_strength:.3f}")
            print(f"  ⚡ Volatilidad: {volatility_level}")

            return context

        except Exception as e:
            print(f"❌ Error en análisis de contexto: {e}")
            # Contexto neutral por defecto
            return {
                'regime': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.0,
                'market_fear_factor': 0.5,
                'trend_strength': 0.0,
                'volatility_level': 'MEDIUM',
                'correlation_strength': 0.5,
                'altcoin_strength': 0.0,
                'btc_trend_score': 0.0,
                'components': {}
            }

    def _apply_market_context_filter(self, signal: str, confidence: float, market_context: Dict, symbol: str) -> tuple:
        """
        🛡️ Aplicar filtro de contexto de mercado como capa de seguridad adicional

        Args:
            signal: Señal original del modelo TCN ('BUY', 'SELL', 'HOLD')
            confidence: Confianza de la señal (0-100)
            market_context: Contexto de mercado analizado
            symbol: Símbolo siendo analizado

        Returns:
            tuple: (señal_filtrada, razón_del_filtro)
        """
        try:
            regime = market_context['regime']
            market_score = market_context['score']
            market_confidence = market_context['confidence']
            fear_factor = market_context['market_fear_factor']
            volatility = market_context['volatility_level']

            original_signal = signal
            filter_reason = ""

            # 🔴 FILTROS BEARISH - Restricciones de seguridad en mercado bajista (OPTIMIZADO INTEGRAL)
            if regime == 'BEARISH' and market_confidence > 0.7:
                if signal == 'BUY':
                    # ✅ UMBRALES DIFERENCIADOS POR ACTIVO EN BEARISH
                    required_confidence = {
                        'BTCUSDT': 85,   # BTC líder - umbral moderado
                        'ETHUSDT': 82,   # ETH principal altcoin - umbral medio
                        'BNBUSDT': 80    # BNB exchange token - umbral más bajo
                    }.get(symbol, 88)  # Otros activos: 88%

                    if confidence >= required_confidence:
                        filter_reason = f"{symbol.replace('USDT', '')} permitido en BEARISH por alta confianza ({confidence:.1f}% > {required_confidence}%)"
                    else:
                        signal = 'HOLD'
                        filter_reason = f"Mercado BEARISH fuerte (score: {market_score:.2f}) - {symbol} BUY requiere >{required_confidence}% confianza"

                elif signal == 'SELL':
                    # En bearish, SELL es más seguro - reducir umbral ligeramente
                    if confidence < 60:  # Reducir de 70% a 60% para SELL en bearish
                        signal = 'HOLD'
                        filter_reason = f"SELL en BEARISH requiere >60% confianza"
                    else:
                        filter_reason = f"SELL favorecido en mercado BEARISH"

            # 🟢 FILTROS BULLISH - Aprovechar momentum alcista
            elif regime == 'BULLISH' and market_confidence > 0.7:
                if signal == 'BUY':
                    # En bullish fuerte, relajar ligeramente el umbral para BUY
                    if confidence < 65:  # Reducir de 70% a 65% para BUY en bullish
                        signal = 'HOLD'
                        filter_reason = f"BUY en BULLISH requiere >65% confianza"
                    else:
                        filter_reason = f"BUY favorecido en mercado BULLISH"

                elif signal == 'SELL':
                    # En bullish, ser más cauteloso con SELL
                    if confidence < 80:  # Subir umbral para SELL en bullish
                        signal = 'HOLD'
                        filter_reason = f"Mercado BULLISH (score: {market_score:.2f}) - SELL requiere >80% confianza"

            # 🟡 FILTROS DE VOLATILIDAD - Ajustar según volatilidad del mercado (OPTIMIZADO INTEGRAL)
            if volatility == 'HIGH' and fear_factor > 0.8:
                # ✅ UMBRALES DIFERENCIADOS POR ACTIVO EN ALTA VOLATILIDAD
                volatility_thresholds = {
                    'BTCUSDT': 78,   # BTC más estable - umbral menor
                    'ETHUSDT': 75,   # ETH volátil - umbral medio
                    'BNBUSDT': 72    # BNB exchange - umbral menor
                }

                if signal == 'BUY':
                    required_vol_confidence = volatility_thresholds.get(symbol, 80)
                    if confidence < required_vol_confidence:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad (miedo: {fear_factor:.2f}) - {symbol} BUY requiere >{required_vol_confidence}% confianza"
                elif signal == 'SELL' and confidence < 70:  # Reducido de 75% a 70%
                    signal = 'HOLD'
                    filter_reason = f"Alta volatilidad - SELL requiere >70% confianza"

            # 🔵 FILTROS ESPECÍFICOS POR ACTIVO
            if symbol == 'BTCUSDT':
                # ✅ OPTIMIZADO: BTC como líder - permitir señales fuertes
                btc_trend = market_context.get('btc_trend_score', 0)
                if signal == 'BUY' and btc_trend < -0.4:  # Solo en tendencia MUY bajista
                    if confidence < 80:  # Reducido de 82% a 80%
                        signal = 'HOLD'
                        filter_reason = f"BTC en tendencia muy bajista fuerte (trend: {btc_trend:.2f}) - BUY requiere >80% confianza"
                    else:
                        filter_reason = f"BTC BUY permitido pese a tendencia bajista por alta confianza ({confidence:.1f}%)"

            elif symbol in ['ETHUSDT', 'BNBUSDT']:
                # ✅ ALTCOINS OPTIMIZADO - Umbrales diferenciados por underperformance
                altcoin_strength = market_context.get('altcoin_strength', 0)
                if signal == 'BUY' and altcoin_strength < -0.2:  # Altcoins underperforming
                    # Umbrales específicos para cada altcoin
                    altcoin_thresholds = {
                        'ETHUSDT': 73,   # ETH principal altcoin - umbral medio
                        'BNBUSDT': 72    # BNB exchange token - umbral menor
                    }

                    required_alt_confidence = altcoin_thresholds.get(symbol, 75)

                    if confidence >= required_alt_confidence:
                        filter_reason = f"{symbol.replace('USDT', '')} permitido por alta confianza ({confidence:.1f}% > {required_alt_confidence}%) pese a underperformance"
                    else:
                        signal = 'HOLD'
                        filter_reason = f"Altcoins underperforming vs BTC - {symbol} BUY requiere >{required_alt_confidence}% confianza"

            # 📊 LOG DEL FILTRO APLICADO
            if signal != original_signal:
                print(f"  🛡️ FILTRO CONTEXTO: {symbol} {original_signal}→{signal}")
                print(f"      Régimen: {regime} (conf: {market_confidence:.1%})")
                print(f"      Razón: {filter_reason}")

            return signal, filter_reason

        except Exception as e:
            print(f"❌ Error aplicando filtro de contexto: {e}")
            return signal, f"Error en filtro: {str(e)}"

    async def _process_signal(self, symbol: str, signal_data: Dict):
        """⚡ Procesar una señal individual - CON DEBUG DETALLADO"""

        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        balance_sufficient = signal_data.get('balance_sufficient', True)

        print(f"🔍 PROCESANDO SEÑAL: {symbol} {signal} ({confidence:.1f}%)")

        # Skip si es HOLD
        if signal == 'HOLD':
            print(f"  ⏸️ Señal HOLD ignorada para {symbol}")
            return

        # Verificar si el balance es suficiente para nuevas posiciones BUY
        min_position_value = 11.0  # Valor por defecto
        if self.risk_manager and hasattr(self.risk_manager, 'limits'):
            min_position_value = self.risk_manager.limits.min_position_value_usdt

        print(f"  💰 Balance check: ${self.current_balance:.2f} vs min ${min_position_value:.2f}")

        if signal == 'BUY' and not balance_sufficient:
            print(f"  ❌ Señal BUY {symbol} BLOQUEADA - Balance insuficiente")
            return

        # ✅ CORREGIDO: Verificar posiciones existentes para múltiples posiciones por símbolo
        existing_positions = self._get_positions_for_symbol(symbol)
        print(f"  📊 Posiciones existentes en {symbol}: {len(existing_positions)}/3")

        # Si es señal SELL, gestionar posiciones existentes
        if signal == 'SELL' and existing_positions:
            print(f"  🔄 Señal SELL - Gestionando {len(existing_positions)} posición(es) existente(s) para {symbol}")
            await self._manage_existing_position(symbol, signal_data)

        # Si es señal BUY, considerar nueva posición (independiente de posiciones existentes)
        elif signal == 'BUY':
            if len(existing_positions) >= 3:
                print(f"  ❌ Señal BUY BLOQUEADA - Máximo 3 posiciones alcanzado para {symbol} ({len(existing_positions)}/3)")
                return
            else:
                print(f"  📈 Señal BUY - Considerando nueva posición para {symbol} (será {len(existing_positions)+1}/3)")
                await self._consider_new_position(symbol, signal_data)

    async def _consider_new_position(self, symbol: str, signal_data: Dict):
        """📈 Considerar nueva posición con diversificación - CON DEBUG DETALLADO"""

        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']

        print(f"    🚀 EVALUANDO NUEVA POSICIÓN: {symbol} {signal} ({confidence:.1f}%)")

        # 🔧 CORRECCIÓN: SELL no debe crear nuevas posiciones
        if signal == 'SELL':
            print(f"    ❌ BLOQUEADO: Señal SELL - No hay posición existente que vender en {symbol}")
            return

        # ✅ NUEVO: Verificar diversificación del portafolio ANTES de risk management
        print(f"    🎯 PASO 1: Verificando diversificación para {symbol}...")
        try:
            await self._check_portfolio_diversification_before_trade(symbol, signal_data)
            print(f"    ✅ PASO 1: Diversificación OK para {symbol}")
        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e):
                print(f"    ❌ PASO 1: DIVERSIFICACIÓN BLOQUEÓ: {symbol}: {str(e)}")
                await self._send_discord_notification(f"❌ **DIVERSIFICACIÓN BLOQUEÓ**: {symbol}: {str(e)}")
                return  # Salir sin ejecutar el trade
            else:
                print(f"    ⚠️ PASO 1: Error verificando diversificación para {symbol}: {e}")
                # Continuar con el trade si es un error técnico

        # Verificar límites de riesgo
        print(f"    🛡️ PASO 2: Verificando límites de riesgo para {symbol}...")
        can_trade = True
        reason = ""

        if self.risk_manager:
            print(f"    🛡️ PASO 2: Risk manager disponible, ejecutando check_risk_limits_before_trade...")
            can_trade, reason = await self.risk_manager.check_risk_limits_before_trade(
                symbol, signal, confidence
            )
            print(f"    🛡️ PASO 2: Risk Manager resultado: can_trade={can_trade}, reason='{reason}'")
        else:
            print("    ⚠️ PASO 2: Risk manager no disponible, usando verificaciones básicas")

        if not can_trade:
            print(f"    ❌ PASO 2: RISK MANAGER BLOQUEÓ: {symbol}: {reason}")
            await self._send_discord_notification(f"❌ **RISK MANAGER BLOQUEÓ**: {symbol}: {reason}")
            if self.database:
                await self.database.log_event('WARNING', 'RISK', f'Trade rechazado {symbol}: {reason}', symbol)
            return

        # Abrir nueva posición
        print(f"    💰 PASO 3: Intentando abrir posición para {symbol}...")
        position = None
        if self.risk_manager:
            print(f"    💰 PASO 3: Risk manager disponible, ejecutando open_position...")
            position = await self.risk_manager.open_position(symbol, signal, confidence, current_price)
            if position:
                print(f"    ✅ PASO 3: POSICIÓN CREADA: {symbol} - Order ID: {position.order_id}")
            else:
                print(f"    ❌ PASO 3: NO SE PUDO CREAR POSICIÓN: {symbol} - Risk manager devolvió None")
                await self._send_discord_notification(f"❌ **NO SE PUDO CREAR POSICIÓN**: {symbol} - Risk manager falló")
        else:
            print("    ❌ PASO 3: Risk manager no disponible, no se puede abrir posición")
            await self._send_discord_notification(f"❌ **RISK MANAGER NO DISPONIBLE**: {symbol}")
            return

        if position:
            print(f"    ✅ PASO 4: Posición creada exitosamente, guardando en registros...")

            # ✅ CORREGIDO: Usar trade_id (que ahora es el order_id) como clave
            self.active_positions[position.order_id] = position
            print(f"    ✅ PASO 4: Posición guardada en active_positions con ID: {position.order_id}")

            # Guardar en base de datos, incluyendo el order_id de Binance
            trade_data = {
                'symbol': symbol,
                'side': signal,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'confidence': confidence,
                'strategy': 'TCN_MODEL_SIGNALS',
                'is_active': True,
                'metadata': {
                    'signal_reason': signal_data.get('reason'),
                    'signal_time': signal_data['timestamp'].isoformat()
                },
                'order_id': position.order_id # ✅ CRÍTICO: Guardar el ID de la orden de Binance
            }
            print(f"    ✅ PASO 4: Trade data preparado para DB")

            # El ID interno de la DB se genera automáticamente, no necesitamos guardarlo aquí.
            if self.database:
                await self.database.save_trade(trade_data)
                print(f"    ✅ PASO 4: Trade guardado en base de datos")
            else:
                print("    ⚠️ PASO 4: Database no disponible, trade no guardado en DB")

            self.trade_count += 1
            print(f"    ✅ PASO 4: Trade count incrementado a: {self.trade_count}")

            # Log del trade
            if self.database:
                await self.database.log_event(
                    'INFO', 'TRADING',
                    f'Nueva posición: {signal} {symbol} @ ${current_price:.4f}',
                    symbol
                )

            # Enviar notificación Discord si está configurado
            trade_notification_data = {
                'symbol': symbol,
                'side': signal,
                'value_usd': position.quantity * position.entry_price,
                'price': current_price,
                'confidence': confidence,
                'pnl_percent': 0,
                'pnl_usd': 0,
                'reason': 'NEW_POSITION'
            }

            # Usar Smart Discord Notifier para trades
            if hasattr(self, 'discord_notifier'):
                await self.discord_notifier.send_trade_notification(trade_notification_data)
                print(f"    ✅ PASO 5: Notificación enviada vía Smart Discord Notifier")
            else:
                await self._send_discord_notification(f"🟢 **NUEVA POSICIÓN**\n"
                                                     f"📊 {symbol}: {signal}\n"
                                                     f"💰 Precio: ${current_price:.4f}\n"
                                                     f"🎯 Confianza: {confidence:.1%}\n"
                                                     f"📈 Cantidad: {position.quantity:.6f}")
                print(f"    ✅ PASO 5: Notificación enviada vía Discord básico")

            print(f"    🎉 ÉXITO TOTAL: Nueva posición creada para {symbol} - Proceso completado exitosamente")
        else:
            print(f"    ❌ FALLO FINAL: position es None después de todos los pasos - INVESTIGAR RISK MANAGER")

    async def _manage_existing_position(self, symbol: str, signal_data: Dict):
        """🔄 Gestionar posición existente"""

        existing_positions = self._get_positions_for_symbol(symbol)
        if not existing_positions:
            return

        current_price = signal_data['current_price']
        signal = signal_data['signal']
        confidence = signal_data['confidence']

        # Actualizar PnL de todas las posiciones para este símbolo
        for position in existing_positions:
            position.current_price = current_price
            if position.side == 'BUY':
                position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            else: # Futuros
                position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
            position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)

        # Si la señal es de venta, cerrar TODAS las posiciones para este símbolo
        if signal == 'SELL':
            print(f"🔥 Señal de VENTA para {symbol}. Cerrando {len(existing_positions)} posición(es).")
            for position in existing_positions:
                await self._close_position(position.order_id, "SIGNAL_SELL")

        # Lógica de reversión (ej. de BUY a SELL con alta confianza)
        reversal_threshold = float(os.getenv('SIGNAL_REVERSAL_THRESHOLD', '0.85'))
        if confidence > reversal_threshold:
            for position in existing_positions:
                if (position.side == 'BUY' and signal == 'SELL') or (position.side == 'SELL' and signal == 'BUY'):
                    await self._close_position(position.order_id, "SIGNAL_REVERSAL")

    async def _close_position(self, order_id: str, reason: str):
        """📉 Cerrar posición específica por ID de orden"""

        # ✅ CORREGIDO: Buscar en portfolio_manager.position_registry
        position = self.portfolio_manager.position_registry.get(order_id)
        if not position:
            print(f"ℹ️ Intento de cerrar posición {order_id} omitido. No encontrada en registry.")
            return

        # ✅ NUEVO: Verificar que no esté ya marcada para cierre
        if hasattr(position, 'is_closing') and position.is_closing:
            print(f"ℹ️ Posición {order_id} ya está siendo cerrada.")
            return

        # ✅ CRÍTICO: Marcar como en proceso de cierre
        position.is_closing = True

        print(f"👇 Iniciando cierre para {position.symbol} (ID Orden: {order_id}) por motivo: {reason}")

        # Ahora el resto de la función puede proceder de forma segura.
        symbol = position.symbol
        current_price = await self.get_current_price(symbol)

        # Ejecutar orden REAL de venta en Binance
        print(f"   💸 Ejecutando orden de VENTA REAL para {position.quantity} de {symbol} a ${current_price:.4f}")

        # ✅ CRÍTICO: Ejecutar orden real de cierre
        close_order_result = await self._execute_sell_order(position)
        if close_order_result:
            print(f"🎉 Orden de cierre ejecutada: ID {close_order_result['orderId']}")
            # Usar precio real de la orden ejecutada si está disponible
            if 'fills' in close_order_result and len(close_order_result['fills']) > 0:
                current_price = float(close_order_result['fills'][0]['price'])
        else:
            print(f"⚠️ No se pudo ejecutar orden de cierre, usando precio de mercado")

        # Calcular PnL final
        if position.side == 'BUY':
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

        pnl_usd = (pnl_percent / 100) * (position.quantity * position.entry_price)

        # Actualizar estadísticas
        self.session_pnl += pnl_usd

        # Actualizar en base de datos
        if hasattr(position, 'position_id') and position.position_id:
            exit_data = {
                'exit_price': current_price,
                'exit_time': datetime.now(),
                'pnl_percent': pnl_percent,
                'pnl_usd': pnl_usd,
                'exit_reason': reason
            }
            await self.database.update_trade_exit(position.position_id, exit_data)

        # ✅ CORREGIDO: Remover del portfolio_manager.position_registry
        if order_id in self.portfolio_manager.position_registry:
            del self.portfolio_manager.position_registry[order_id]
            print(f"🗑️ Posición {order_id} eliminada del registry")

        # Log y notificación
        color = "🟢" if pnl_usd > 0 else "🔴"
        await self.database.log_event(
            'INFO', 'TRADING',
            f'Posición cerrada: {symbol} - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})',
            symbol
        )

        await self._send_discord_notification(f"{color} **POSICIÓN CERRADA**\n"
                                             f"📊 {symbol}: {position.side}\n"
                                             f"📈 PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})\n"
                                             f"🔄 Razón: {reason}")

        print(f"📉 Posición cerrada: {symbol} (ID de orden: {order_id}) - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})")

    async def _check_portfolio_diversification_before_trade(self, symbol: str, signal_data: Dict):
        """
        🎯 Verificar diversificación antes de ejecutar trade
        ---
        VERSIÓN SIMPLIFICADA: Para bots con pocos pares (BTC, ETH, BNB)
        Solo aplica restricciones básicas sin bloquear oportunidades
        """

        try:
            # ✅ NUEVO: Con solo 3 pares, usar lógica simplificada
            print(f"🎯 Verificación de diversificación simplificada para {symbol}")

            # Obtener posiciones actuales solo para información
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            # Contar posiciones por símbolo
            positions_by_symbol = {}
            total_exposure_usd = 0.0

            for pos in snapshot.active_positions:
                if pos.symbol not in positions_by_symbol:
                    positions_by_symbol[pos.symbol] = []
                positions_by_symbol[pos.symbol].append(pos)
                # ✅ CORREGIDO: Usar valor de entrada (inversión original), no market_value actual
                entry_value = pos.quantity * pos.entry_price
                total_exposure_usd += entry_value

            # ✅ REGLAS SIMPLIFICADAS PARA POCOS PARES:

            # 1. Máximo 3 posiciones por símbolo (configuración correcta)
            existing_positions_count = len(positions_by_symbol.get(symbol, []))
            if existing_positions_count >= 3:
                print(f"🚫 DIVERSIFICACIÓN: Máximo 3 posiciones por símbolo alcanzado para {symbol}")
                print(f"   📊 Posiciones actuales en {symbol}: {existing_positions_count}")
                raise Exception(f"Trade bloqueado por diversificación: Máximo 3 posiciones por símbolo en {symbol}")

            # 2. Lógica inteligente de exposición basada en contexto
            confidence = signal_data['confidence']
            position_size_percent = min(15.0, confidence * 20)
            position_size_usd = (self.current_balance * position_size_percent / 100)

            new_total_exposure = total_exposure_usd + position_size_usd
            exposure_percent = (new_total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0

            # ✅ LÓGICA INTELIGENTE: Límite dinámico basado en confianza y número de pares
            max_exposure = 90.0  # Base: 90%

            # Aumentar límite si la confianza es muy alta
            if confidence >= 80.0:
                max_exposure = 95.0  # Permitir hasta 95% con alta confianza

            # Reducir límite solo si ya tenemos muchas posiciones distribuidas
            total_symbols_with_positions = len(positions_by_symbol)
            if total_symbols_with_positions >= 3:  # Si ya tenemos posiciones en los 3 pares
                max_exposure = 85.0  # Ser más conservador

            if exposure_percent > max_exposure:
                print(f"🚫 DIVERSIFICACIÓN: Exposición total muy alta")
                print(f"   📊 Exposición actual: {(total_exposure_usd/self.current_balance)*100:.1f}%")
                print(f"   📊 Nueva exposición: {exposure_percent:.1f}% > {max_exposure:.0f}%")
                print(f"   💰 Balance actual: ${self.current_balance:.2f}")
                print(f"   💰 Inversión total: ${total_exposure_usd:.2f}")
                print(f"   💰 Nueva inversión: ${position_size_usd:.2f}")
                print(f"   🎯 Confianza: {confidence:.1f}% | Pares activos: {total_symbols_with_positions}/3")
                raise Exception(f"Trade bloqueado por diversificación: Exposición total > {max_exposure:.0f}%")

            # 3. ✅ PERMITIR: Concentración en un solo par si es rentable
            # Con solo 3 pares, es normal tener concentración temporal

            # ✅ INFORMACIÓN: Solo mostrar estado sin bloquear
            print(f"✅ DIVERSIFICACIÓN: Trade permitido para {symbol}")
            print(f"   📊 Posiciones en {symbol}: {existing_positions_count}/3")
            print(f"   💰 Exposición total: {exposure_percent:.1f}%/{max_exposure:.0f}%")
            print(f"   🎯 Tamaño propuesto: ${position_size_usd:.2f} ({position_size_percent:.1f}%)")
            print(f"   🔥 Confianza: {confidence:.1f}% | Pares activos: {len(positions_by_symbol)}/3")

            # ✅ OPCIONAL: Análisis informativo cada 5 trades (no bloquea)
            if self.trade_count % 5 == 0 and len(snapshot.active_positions) > 0:
                print(f"📊 RESUMEN DE PORTAFOLIO:")
                for sym, positions in positions_by_symbol.items():
                    total_value = sum(pos.market_value for pos in positions)
                    percentage = (total_value / snapshot.total_balance_usd) * 100 if snapshot.total_balance_usd > 0 else 0
                    print(f"   {sym}: {len(positions)} posición(es), ${total_value:.2f} ({percentage:.1f}%)")

        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e):
                # Solo bloquear en casos extremos (>3 posiciones por par o >90% exposición)
                print(f"🚫 {str(e)}")
                await self.database.log_event('WARNING', 'DIVERSIFICATION', str(e), symbol)

                # Notificación Discord más suave
                await self._send_discord_notification(
                    f"⚠️ **DIVERSIFICACIÓN: LÍMITE ALCANZADO**\n"
                    f"📊 {symbol}: {signal_data['signal']}\n"
                    f"💡 {str(e).replace('Trade bloqueado por diversificación: ', '')}\n"
                    f"🎯 Confianza: {signal_data['confidence']:.1%}"
                )

                raise  # Re-lanzar solo bloqueos reales
            else:
                # Errores técnicos no deben bloquear trades
                print(f"⚠️ Error técnico en diversificación (ignorado): {e}")
                print(f"✅ Continuando con el trade para {symbol}")

    async def _heartbeat_monitor(self):
        """💓 Monitor de latido del sistema"""
        while self.status == TradingManagerStatus.RUNNING:
            try:
                # Verificar conectividad cada 5 minutos
                await asyncio.sleep(300)

                # Ping a Binance
                test_price = await self.get_current_price("BTCUSDT")
                if test_price <= 0:
                    raise Exception("No se pudo obtener precio de BTC")

                # Log heartbeat
                await self.database.log_event('INFO', 'SYSTEM', f'Heartbeat OK - BTC: ${test_price:.2f}')

            except Exception as e:
                await self.database.log_event('ERROR', 'SYSTEM', f'Heartbeat failed: {e}')
                await asyncio.sleep(60)

    async def _position_monitor(self):
        """🔍 Monitoreo continuo de posiciones y gestión de riesgo - ALTA FRECUENCIA"""
        try:
            # ✅ MÁXIMA FRECUENCIA: Datos frescos cada 30 segundos para TCN
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            if not snapshot.active_positions:
                print("   📊 Sin posiciones activas para monitorear")
                return

            print(f"🔍 Monitoreando {len(snapshot.active_positions)} posición(es)...")

            # 2. Actualizar precios para cada posición - FRECUENCIA MÁXIMA
            symbols_to_update = list(set([pos.symbol for pos in snapshot.active_positions]))
            current_prices = await self.portfolio_manager.update_all_prices(symbols_to_update)

            positions_to_close = []
            trailing_updates = []

            # 3. ✅ NUEVO: Procesar cada posición individualmente con trailing stop
            for i, position in enumerate(snapshot.active_positions):
                try:
                    current_price = current_prices.get(position.symbol, position.current_price)

                    # Actualizar precio actual en la posición
                    position.current_price = current_price

                    # 🔄 Recalcular PnL con precio actual
                    if position.side == 'BUY':
                        entry_value = position.quantity * position.entry_price
                        current_value = position.quantity * current_price
                        position.unrealized_pnl_usd = current_value - entry_value
                        position.unrealized_pnl_percent = (position.unrealized_pnl_usd / entry_value) * 100 if entry_value > 0 else 0.0
                        position.market_value = current_value

                    # ✅ NUEVO: Aplicar trailing stop profesional
                    updated_position, stop_triggered, trigger_reason = self.portfolio_manager.update_trailing_stop_professional(
                        position, current_price
                    )

                    # ✅ CRÍTICO: Actualizar la posición en el snapshot Y en el registry
                    snapshot.active_positions[i] = updated_position

                    # ✅ PERSISTENCIA: Actualizar también en el registry para mantener estado
                    if updated_position.order_id and updated_position.order_id in self.portfolio_manager.position_registry:
                        self.portfolio_manager.position_registry[updated_position.order_id] = updated_position

                    # Si se actualiza el trailing, registrar el cambio
                    if hasattr(updated_position, 'trailing_stop_active') and updated_position.trailing_stop_active:
                        if updated_position.trailing_movements > position.trailing_movements:
                            trailing_updates.append(f"📈 {updated_position.symbol} Pos #{updated_position.order_id}: Trail movido a ${updated_position.trailing_stop_price:.4f}")

                    # Verificar condiciones de cierre
                    should_close, close_reason = await self._check_position_exit_conditions(updated_position, current_price)

                    if stop_triggered or should_close:
                        reason = trigger_reason if stop_triggered else close_reason
                        positions_to_close.append((updated_position, reason))

                        print(f"🛑 Marcando para cierre: {updated_position.symbol} Pos #{updated_position.order_id} - {reason}")

                except Exception as e:
                    print(f"❌ Error monitoreando {position.symbol}: {e}")
                    continue

            # 4. Mostrar actualizaciones de trailing stops
            if trailing_updates:
                print("📈 ACTUALIZACIONES TRAILING STOPS:")
                for update in trailing_updates:
                    print(f"   {update}")

            # 5. Cerrar posiciones marcadas
            if positions_to_close:
                await self._close_positions_batch(positions_to_close)
            else:
                print("   ✅ Todas las posiciones dentro de parámetros")

                # Mostrar resumen de trailing stops activos
                active_trailing = [pos for pos in snapshot.active_positions
                                 if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active]

                if active_trailing:
                    print(f"   📈 Trailing stops activos: {len(active_trailing)}")
                    for pos in active_trailing:
                        protection = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price * 100) if pos.trailing_stop_price else 0
                        print(f"      {pos.symbol} Pos #{pos.order_id}: ${pos.trailing_stop_price:.4f} (+{protection:.2f}%)")

        except Exception as e:
            print(f"❌ Error en monitoreo de posiciones: {e}")

    async def _check_position_exit_conditions(self, position, current_price: float) -> tuple:
        """🛡️ Verificar condiciones de salida para una posición"""
        try:
            # Calcular PnL actual
            if position.side == 'BUY':
                pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

            # ✅ TRADICIONAL: Stop Loss y Take Profit (solo si trailing no está activo)
            if hasattr(position, 'trailing_stop_active') and not position.trailing_stop_active:
                # Stop Loss tradicional
                if pnl_percent <= -3.0:
                    return True, f"STOP_LOSS_TRADICIONAL (-{abs(pnl_percent):.2f}%)"

                # Take Profit tradicional
                if pnl_percent >= 6.0:
                    return True, f"TAKE_PROFIT_TRADICIONAL (+{pnl_percent:.2f}%)"

            # ✅ CIRCUITO: Pérdida máxima diaria
            if await self._daily_loss_exceeds_limit():
                return True, "CIRCUIT_BREAKER_DAILY_LOSS"

            return False, ""

        except Exception as e:
            print(f"❌ Error verificando condiciones de salida: {e}")
            return False, ""

    async def _close_positions_batch(self, positions_and_reasons: List[Tuple]) -> None:
        """🚀 Cerrar múltiples posiciones en lote - CON ÓRDENES REALES"""
        try:
            print(f"🔥 Iniciando cierre de {len(positions_and_reasons)} posición(es)...")

            for position, reason in positions_and_reasons:
                try:
                    print(f"🛑 CERRANDO POSICIÓN {position.symbol} Pos #{position.order_id}:")
                    print(f"   📍 Entrada: ${position.entry_price:.4f}")
                    print(f"   💰 Actual: ${position.current_price:.4f}")
                    print(f"   📊 PnL: {position.unrealized_pnl_percent:.2f}% (${position.unrealized_pnl_usd:.2f})")
                    print(f"   🏷️ Razón: {reason}")

                    # ✅ EJECUTAR ORDEN REAL DE CIERRE EN BINANCE
                    order_result = await self._execute_sell_order(position)

                    if order_result:
                        # Usar precio real de ejecución
                        real_close_price = float(order_result.get('fills', [{}])[0].get('price', position.current_price))
                        real_quantity = float(order_result.get('executedQty', position.quantity))

                        # Calcular PnL real con precio de ejecución
                        if position.side == 'BUY':
                            real_pnl_percent = ((real_close_price - position.entry_price) / position.entry_price) * 100
                        else:
                            real_pnl_percent = ((position.entry_price - real_close_price) / position.entry_price) * 100

                        real_pnl_usd = (real_pnl_percent / 100) * (real_quantity * position.entry_price)

                        print(f"✅ ORDEN REAL EJECUTADA:")
                        print(f"   🆔 Order ID: {order_result.get('orderId')}")
                        print(f"   💲 Precio real: ${real_close_price:.4f}")
                        print(f"   📊 PnL real: {real_pnl_percent:.2f}% (${real_pnl_usd:.2f})")

                        # Actualizar métricas con datos reales
                        self.session_pnl += real_pnl_usd

                    else:
                        print(f"❌ Error ejecutando orden real - usando datos estimados")
                        # Fallback a datos estimados si falla la orden
                        real_pnl_percent = position.unrealized_pnl_percent
                        real_pnl_usd = position.unrealized_pnl_usd
                        self.session_pnl += real_pnl_usd

                    # Logging de la operación
                    await self.database.log_event(
                        'TRADE',
                        'POSITION_CLOSED',
                        f"{position.symbol}: {reason} - PnL: {real_pnl_percent:.2f}% - Order: {order_result.get('orderId', 'FAILED') if order_result else 'FAILED'}"
                    )

                    # Actualizar métricas
                    self.metrics['total_trades'] += 1
                    if real_pnl_usd > 0:
                        self.metrics['profitable_trades'] += 1

                    print(f"✅ Posición {position.symbol} cerrada exitosamente")

                except Exception as e:
                    print(f"❌ Error cerrando {position.symbol}: {e}")
                    await self.database.log_event('ERROR', 'TRADING', f'Error cerrando posición {position.symbol}: {e}')
                    continue

            print(f"🎯 Proceso de cierre completado")

        except Exception as e:
            print(f"❌ Error en cierre de posiciones en lote: {e}")

    async def _execute_sell_order(self, position) -> Optional[Dict]:
        """🔥 EJECUTAR ORDEN REAL DE VENTA EN BINANCE"""
        try:
            # Determinar lado de la orden de cierre
            close_side = 'SELL' if position.side == 'BUY' else 'BUY'

            # Obtener precio actual para la orden
            current_price = await self.get_current_price(position.symbol)

            # ✅ NUEVO: Verificar balance real antes de ejecutar
            asset_symbol = position.symbol.replace('USDT', '')  # BTC, ETH, BNB
            available_balance = await self._get_available_balance(asset_symbol)

            if available_balance <= 0:
                print(f"❌ No hay balance disponible de {asset_symbol} para vender")
                return None

            # ✅ CORREGIDO: Usar el menor entre la cantidad de la posición y el balance disponible
            max_sellable = min(position.quantity, available_balance)

            print(f"📊 Verificación de balance {asset_symbol}:")
            print(f"   Posición: {position.quantity:.8f}")
            print(f"   Disponible: {available_balance:.8f}")
            print(f"   A vender: {max_sellable:.8f}")

            # Preparar parámetros de orden
            timestamp = int(time.time() * 1000)

            # Ajustar cantidad según filtros del símbolo
            adjusted_quantity = await self._adjust_quantity_for_symbol(position.symbol, max_sellable)

            params = {
                'symbol': position.symbol,
                'side': close_side,
                'type': 'MARKET',  # Orden de mercado para cierre inmediato
                'quantity': f"{adjusted_quantity:.8f}".rstrip('0').rstrip('.'),
                'timestamp': timestamp,
                'recvWindow': 10000
            }

            # Crear signature
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            params['signature'] = signature

            # Headers de autenticación
            headers = {
                'X-MBX-APIKEY': self.config.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            print(f"📡 Ejecutando orden de cierre: {close_side} {params['quantity']} {position.symbol}")

            # Ejecutar orden POST /api/v3/order
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/order"

                async with session.post(url, data=params, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"🎉 ORDEN DE CIERRE EJECUTADA: {result['orderId']}")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ Error Binance API: {response.status} - {error_text}")
                        return None

        except Exception as e:
            print(f"❌ ERROR ejecutando orden de cierre: {e}")
            return None

    async def _adjust_quantity_for_symbol(self, symbol: str, quantity: float) -> float:
        """🔧 Ajustar cantidad según filtros del símbolo"""
        try:
            # Obtener información del símbolo
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.base_url}/api/v3/exchangeInfo"
                params = {'symbol': symbol}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'symbols' in data and len(data['symbols']) > 0:
                            symbol_info = data['symbols'][0]

                            # Buscar filtro LOT_SIZE
                            for filter_info in symbol_info.get('filters', []):
                                if filter_info['filterType'] == 'LOT_SIZE':
                                    step_size = float(filter_info['stepSize'])
                                    min_qty = float(filter_info['minQty'])

                                    # ✅ CORREGIDO: Ajustar cantidad correctamente al step size
                                    if quantity < min_qty:
                                        print(f"⚠️ Cantidad {quantity:.8f} menor que mínimo {min_qty:.8f}")
                                        return 0.0  # No se puede vender cantidad menor al mínimo

                                    # Redondear hacia abajo para no exceder la cantidad disponible
                                    adjusted_qty = (quantity // step_size) * step_size

                                    # Verificar que sigue siendo >= mínimo después del ajuste
                                    if adjusted_qty < min_qty:
                                        print(f"⚠️ Cantidad ajustada {adjusted_qty:.8f} menor que mínimo {min_qty:.8f}")
                                        return 0.0

                                    print(f"🔧 Cantidad ajustada: {quantity:.8f} → {adjusted_qty:.8f} (step: {step_size:.8f})")
                                    return adjusted_qty

            # Si no se puede obtener filtros, usar cantidad original
            print(f"⚠️ No se pudieron obtener filtros para {symbol}, usando cantidad original")
            return quantity

        except Exception as e:
            print(f"❌ Error ajustando cantidad para {symbol}: {e}")
            return quantity

    async def _get_available_balance(self, asset: str) -> float:
        """💰 Obtener balance disponible de un activo específico"""
        try:
            # Obtener información de cuenta
            account_info = await self.get_account_info()
            if not account_info:
                print(f"❌ No se pudo obtener información de cuenta para {asset}")
                return 0.0

            # Buscar el balance del activo
            for asset_symbol, balance_info in account_info.balances.items():
                if asset_symbol == asset:
                    available = balance_info.get('free', 0.0) if isinstance(balance_info, dict) else 0.0
                    print(f"💰 Balance disponible {asset}: {available:.8f}")
                    return available

            print(f"⚠️ No se encontró balance para {asset}")
            return 0.0

        except Exception as e:
            print(f"❌ Error obteniendo balance de {asset}: {e}")
            return 0.0

    async def _daily_loss_exceeds_limit(self, max_daily_loss_percent: float = None) -> bool:
        """🚨 Verificar si se ha excedido la pérdida máxima diaria"""
        try:
            # Obtener límite desde .env si no se proporciona
            if max_daily_loss_percent is None:
                max_daily_loss_percent = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10.0'))
            # Obtener snapshot actual
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            # Calcular pérdida porcentual del día
            if snapshot.total_balance_usd > 0:
                daily_pnl_percent = (snapshot.total_unrealized_pnl / snapshot.total_balance_usd) * 100

                if daily_pnl_percent <= -max_daily_loss_percent:
                    print(f"🚨 CIRCUIT BREAKER: Pérdida diaria {daily_pnl_percent:.2f}% >= {max_daily_loss_percent}%")
                    return True

            return False

        except Exception as e:
            print(f"❌ Error verificando pérdida diaria: {e}")
            return False

    async def _metrics_collector(self):
        """📊 Recolector de métricas del sistema"""
        while self.status == TradingManagerStatus.RUNNING:
            try:
                await asyncio.sleep(120)  # Cada 2 minutos

                # Recolectar métricas básicas
                await self._update_metrics()

            except Exception as e:
                await self.database.log_event('ERROR', 'METRICS', f'Error collecting metrics: {e}')
                await asyncio.sleep(60)

    async def _save_periodic_metrics(self):
        """💾 Guardar métricas periódicamente"""
        try:
            # Actualizar balance actual en risk manager
            total_balance = self.current_balance + self.session_pnl
            await self.risk_manager.update_balance(total_balance)

            trades_today = await self._get_total_trades_today()
            win_rate = await self._calculate_win_rate()

            # Calcular exposición total
            total_exposure = 0
            for position in self.active_positions.values():
                if hasattr(position, 'current_price') and position.current_price > 0:
                    total_exposure += position.quantity * position.current_price
                else:
                    total_exposure += position.quantity * position.entry_price

            exposure_percent = (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0

            metrics_data = {
                'timestamp': datetime.now(),
                'total_balance': total_balance,
                'daily_pnl': self.session_pnl,
                'total_pnl': self.session_pnl,  # Para sesión actual
                'daily_return_percent': (self.session_pnl / self.current_balance) * 100 if self.current_balance > 0 else 0,
                'total_return_percent': (self.session_pnl / self.current_balance) * 100 if self.current_balance > 0 else 0,
                'current_drawdown': 0.0,  # Calcular en futuras versiones
                'max_drawdown': 0.0,
                'sharpe_ratio': None,
                'win_rate': win_rate,
                'profit_factor': None,
                'active_positions_count': len(self.active_positions),
                'total_exposure_usd': total_exposure,
                'exposure_percent': exposure_percent,
                'trades_today': trades_today,
                'avg_trade_duration_minutes': None,
                'api_calls_today': self.metrics.get('api_calls_count', 0),
                'error_count_today': self.metrics.get('error_count', 0),
                'last_balance_update': self.metrics.get('last_balance_update', None)
            }

            await self.database.save_performance_metrics(metrics_data)

            # Mostrar resumen de métricas cada 10 ciclos
            if self.metrics['total_checks'] % 10 == 0:
                print(f"\n📊 RESUMEN DE MÉTRICAS:")
                print(f"   📈 Balance total: ${total_balance:.2f}")
                print(f"   💰 PnL sesión: ${self.session_pnl:.2f}")
                print(f"   📊 Trades hoy: {trades_today}")
                print(f"   🎯 Win rate: {win_rate:.1f}%")
                print(f"   💼 Exposición: {exposure_percent:.1f}%")
                print(f"   🔧 API calls: {self.metrics.get('api_calls_count', 0)}")
                print(f"   ❌ Errores: {self.metrics.get('error_count', 0)}")

        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")

    async def _get_total_trades_today(self) -> int:
        """📊 Obtener total de trades de hoy"""
        try:
            trades = await self.database.get_trades_history(days=1)
            return len(trades)
        except:
            return self.trade_count

    async def _calculate_win_rate(self) -> float:
        """🎯 Calcular win rate"""
        try:
            trades = await self.database.get_trades_history(days=7, is_active=False)
            if not trades:
                return 0.0

            wins = sum(1 for trade in trades if trade.get('pnl_usd', 0) > 0)
            return (wins / len(trades)) * 100
        except:
            return 0.0

    async def _update_metrics(self):
        """📈 Actualizar métricas internas"""
        self.metrics['active_positions'] = len(self.active_positions)
        self.metrics['session_pnl'] = self.session_pnl

    async def _handle_pause_state(self):
        """⏸️ Manejar estado de pausa"""
        while self.status == TradingManagerStatus.PAUSED:
            print("⏸️ Sistema pausado - esperando reanudación...")
            await asyncio.sleep(10)

    async def _handle_error(self, error: Exception):
        """❌ Manejar errores del sistema"""
        error_msg = f"Error en loop principal: {error}"
        print(f"❌ {error_msg}")

        await self.database.log_event('ERROR', 'SYSTEM', error_msg)

        # Si hay muchos errores consecutivos, pausar el sistema
        self.metrics['error_count'] += 1

        if self.metrics['error_count'] > 10:
            await self.pause_trading_with_reason("Demasiados errores consecutivos")

    async def _send_discord_notification(self, message: str):
        """💬 Enviar notificación a Discord usando Smart Notifier"""
        try:
            # Importar Smart Discord Notifier si no está disponible
            if not hasattr(self, 'discord_notifier'):
                from smart_discord_notifier import SmartDiscordNotifier, NotificationPriority
                self.discord_notifier = SmartDiscordNotifier()

                # Configurar filtros conservadores para evitar spam
                self.discord_notifier.configure_filters(
                    min_trade_value_usd=12.0,          # Solo trades > $12
                    min_pnl_percent_notify=2.0,        # Solo PnL > 2%
                    max_notifications_per_hour=8,      # Max 8/hora
                    max_notifications_per_day=40,      # Max 40/día
                    suppress_similar_minutes=10,       # 10 min entre similares
                    only_profitable_trades=False,      # Notificar pérdidas también
                    emergency_only_mode=False          # Todas las prioridades
                )

            # Determinar prioridad basada en el mensaje
            from smart_discord_notifier import NotificationPriority

            if "EMERGENCIA" in message or "PARADA" in message:
                priority = NotificationPriority.CRITICAL
            elif "ERROR" in message or "❌" in message:
                priority = NotificationPriority.HIGH
            elif "NUEVA POSICIÓN" in message or "CERRADA" in message:
                priority = NotificationPriority.MEDIUM
            else:
                priority = NotificationPriority.LOW

            # Enviar usando el Smart Notifier
            await self.discord_notifier.send_system_notification(message, priority)

        except Exception as e:
            print(f"⚠️ Error enviando notificación Discord: {e}")

    # Métodos de control del sistema

    async def pause_trading_with_reason(self, reason: str):
        """⏸️ Pausar trading con razón específica"""
        self.status = TradingManagerStatus.PAUSED
        await self.database.log_event('WARNING', 'SYSTEM', f'Trading pausado: {reason}')
        print(f"⏸️ Trading pausado: {reason}")

    async def resume_trading(self):
        """▶️ Reanudar trading"""
        if self.status == TradingManagerStatus.PAUSED:
            self.status = TradingManagerStatus.RUNNING
            await self.database.log_event('INFO', 'SYSTEM', 'Trading reanudado')
            print("▶️ Trading reanudado")

    async def emergency_stop(self):
        """🚨 Parada de emergencia"""
        self.status = TradingManagerStatus.EMERGENCY_STOP

        # Cerrar todas las posiciones activas
        for order_id in list(self.active_positions.keys()):
            await self._close_position(order_id, "EMERGENCY_STOP")

        await self.database.log_event('CRITICAL', 'SYSTEM', 'Parada de emergencia activada')
        print("🚨 PARADA DE EMERGENCIA ACTIVADA")

    async def get_system_status(self) -> Dict:
        """📊 Obtener estado completo del sistema"""

        # Calcular uptime
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = time.time() - self.start_time

        # Calcular exposición total
        total_exposure = 0
        for position in self.active_positions.values():
            if hasattr(position, 'current_price') and position.current_price > 0:
                total_exposure += position.quantity * position.current_price
            else:
                total_exposure += position.quantity * position.entry_price

        return {
            'status': self.status,
            'environment': self.config.environment,
            'symbols_trading': self.symbols,
            'check_interval': self.check_interval,
            'uptime_minutes': uptime_seconds / 60,
            'current_balance_usdt': self.current_balance,
            'session_pnl': self.session_pnl,
            'total_balance': self.current_balance + self.session_pnl,
            'active_positions': len(self.active_positions),
            'total_exposure_usd': total_exposure,
            'exposure_percent': (total_exposure / self.current_balance) * 100 if self.current_balance > 0 else 0,
            'trade_count': self.trade_count,
            'current_prices': self.current_prices,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'last_balance_update': self.last_balance_update.isoformat() if self.last_balance_update else None,
            'account_info': {
                'usdt_balance': self.account_info.usdt_balance if self.account_info else 0,
                'total_balance_usd': self.account_info.total_balance_usd if self.account_info else 0,
                'other_balances': {k: v for k, v in self.account_info.balances.items()
                                 if k != 'USDT' and v['total'] > 0} if self.account_info else {}
            },
            'metrics': self.metrics
        }

    async def shutdown(self):
        """🔄 Apagado controlado del sistema"""
        print("🔄 Iniciando apagado del sistema...")

        self.status = TradingManagerStatus.STOPPED

        # Cerrar posiciones si hay alguna activa
        if self.active_positions:
            print(f"📉 Cerrando {len(self.active_positions)} posiciones activas...")
            for order_id in list(self.active_positions.keys()):
                await self._close_position(order_id, "SYSTEM_SHUTDOWN")

        # Guardar métricas finales
        await self._save_periodic_metrics()

        # Log final
        await self.database.log_event('INFO', 'SYSTEM', 'Sistema apagado correctamente')

        print("✅ Sistema apagado correctamente")

async def main():
    """🎯 Función principal para testing directo"""
    print("🧪 Modo de prueba - Simple Professional Trading Manager")

    manager = SimpleProfessionalTradingManager()
    try:
        await manager.initialize()
        print("✅ Manager inicializado correctamente")

        # Mostrar estado
        status = await manager.get_system_status()
        print(f"📊 Estado: {status}")

    except Exception as e:
        print(f"❌ Error en testing: {e}")
        if manager:
            await manager.emergency_stop()

if __name__ == "__main__":
    asyncio.run(main())
