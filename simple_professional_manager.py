#!/usr/bin/env python3
"""
🚀 SIMPLE PROFESSIONAL TRADING MANAGER
====================================

Gestor de trading automatizado profesional con:
- Sistema TCN avanzado para predicciones
- Gestión inteligente de riesgo y diversificación
- Monitoreo en tiempo real de posiciones
- Notificaciones automáticas vía Discord
- Análisis de contexto de mercado
"""

import os
import sys
import json
import hmac
import hashlib
import asyncio
import aiohttp
import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Importar nuestros módulos de risk y database
from advanced_risk_manager import AdvancedRiskManager, Position, RiskLimits
from trading_database import TradingDatabase

# Importar el módulo de Smart Discord Notifier
from smart_discord_notifier import SmartDiscordNotifier

# ✅ NUEVO: Importar Professional Portfolio Manager
from professional_portfolio_manager import ProfessionalPortfolioManager, Position as PortfolioManagerPosition

# ✅ NUEVO: Importar Portfolio Diversification Manager
from portfolio_diversification_manager import PortfolioDiversificationManager, PortfolioPosition

# ✅ NUEVO: Importar Real Market Data Provider
from real_market_data_provider import RealMarketDataProvider

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
    """🚀 Trading Manager Profesional"""

    def __init__(self):
        """🏗️ Constructor del Trading Manager Profesional"""
        print("🚀 Inicializando Simple Professional Trading Manager...")

        # Estados y configuración
        self.status = TradingManagerStatus.STOPPED
        self.session_pnl = 0.0
        self.current_balance = 0.0
        self.binance_config = self._load_config()

        # ✅ NUEVO: Sistema de cooldown y estabilidad de señales
        self.signal_history = {}  # Historial de señales por símbolo
        self.last_position_action = {}  # Última acción de posición por símbolo
        self.signal_cooldown = {  # Tiempos de enfriamiento por símbolo (en minutos)
            'ETHUSDT': 2,  # ETH: 15 minutos entre cambios de señal
            'BTCUSDT': 2,  # BTC: 10 minutos
            'BNBUSDT': 2,  # BNB: 12 minutos
            'XRPUSDT': 2,  # XRP: 12 minutos
        }
        self.eth_position_protection = {  # Protección específica para ETH
            'last_close_time': None,
            'min_hold_time_minutes': 20,  # Mínimo 20 min antes de cerrar posición ETH
            'consecutive_signals': 0,     # Contador de señales consecutivas
            'signal_confirmation_required': 1 # Requiere 2 señales consecutivas para ETH
        }

        # Configuración de símbolos y gestores
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']

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
        self.trade_count = 0
        # ✅ CORREGIDO: Clave por order_id para múltiples posiciones por símbolo
        self.active_positions: Dict[str, Position] = {}
        self.account_info = None

        # ✅ NUEVO: Sistema de priorización de señales
        self.pending_signals = {}  # {symbol: signal_data} para priorización

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
            min_pnl_percent_notify=0.5,        # Solo PnL > 2%
            max_notifications_per_hour=10,      # Max 8/hora
            max_notifications_per_day=60,      # Max 40/día
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
            import traceback
            print(f"🔍 Traceback completo: {traceback.format_exc()}")
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
            self.binance_config.secret_key.encode('utf-8'),
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
                'X-MBX-APIKEY': self.binance_config.api_key
            }

            url = f"{self.binance_config.base_url}/api/v3/account"
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
                api_key=self.binance_config.api_key,
                secret_key=self.binance_config.secret_key,
                base_url=self.binance_config.base_url
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
        self.risk_manager = AdvancedRiskManager(self.binance_config)
        await self.risk_manager.initialize()
        print("🛡️ Risk Manager inicializado")

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
                url = f"{self.binance_config.base_url}/api/v3/ticker/price"
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
        - Reporte TCN: cada 3 minutos (completo)
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

                if not self.last_balance_update or (time_since_balance_update is not None and time_since_balance_update > 180):  # 5 minutos
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
        """📊 Generar reporte TCN cada 3 minutos"""
        try:
            now = datetime.now()

            # Verificar si es hora de generar reporte (cada 3 minutos)
            should_generate = False

            if self.last_tcn_report_time is None:
                should_generate = True
            else:
                time_since_last = (now - self.last_tcn_report_time).total_seconds()
                if time_since_last >= 180:  # 3 minutos
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

                # Obtener detalles del sistema robusto para Discord
                components = market_context.get('components', {})
                regime_system = components.get('regime_system', 'UNKNOWN')
                pairs_analyzed = components.get('pairs_analyzed', [])

                models_section += f"""
🌍 **RÉGIMEN DE MERCADO** ({regime_system})
{regime_emoji} **{regime}** (Conf: {regime_confidence:.1%}) {vol_emoji} Vol: {volatility}
📊 Score: {market_score:+.3f} | 🔗 Correlación: {market_context.get('correlation_strength', 0):.2f}
🔍 Pares: {len(pairs_analyzed)} analizados | 🎯 Consenso requerido: >60%

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
                    percentage=(pos.market_value / snapshot.total_balance_usd * 1) if snapshot.total_balance_usd > 0 else 0,
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

        # ✅ ADAPTATIVO: Umbral de confianza inteligente según contexto de mercado
        base_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70')) * 100  # Convertir a porcentaje

        # Relajar umbral general en mercados BULLISH claros
        if market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.9:
            threshold = base_threshold * 0.8  # 70% * 0.8 = 56% para mercado muy bullish
            print(f"🎯 UMBRAL ADAPTATIVO: {threshold:.1f}% (base: {base_threshold:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
        elif market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.7:
            threshold = base_threshold * 0.85  # 70% * 0.85 = 59.5% para mercado bullish
            print(f"🎯 UMBRAL ADAPTATIVO: {threshold:.1f}% (base: {base_threshold:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
        else:
            threshold = base_threshold
            print(f"🎯 UMBRAL ESTÁNDAR: {threshold:.1f}% - Mercado {market_context['regime']}")

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

                # ✅ NUEVO: Aplicar filtros de estabilidad y cooldown
                filtered_signal, filter_reason = self._apply_signal_stability_filter(
                    symbol, signal, confidence_level, current_price, market_context
                )

                if filtered_signal != signal:
                    print(f"🛡️ FILTRO DE ESTABILIDAD aplicado en {symbol}: {signal} → {filtered_signal} ({filter_reason})")
                    signal = filtered_signal
                    # Si la señal fue filtrada a HOLD, reducir confianza
                    if signal == 'HOLD':
                        confidence_level = min(confidence_level, 65.0)

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

                # 🔧 **CORRECCIÓN CRÍTICA** - SELL también requiere confianza mínima (ADAPTATIVO)
                # Aplicar umbral de confianza tanto para BUY como para SELL
                base_sell_confidence = float(os.getenv('MIN_SELL_CONFIDENCE_THRESHOLD', '0.75')) * 100

                # ✅ ADAPTATIVO: Relajar umbral SELL en mercados BULLISH para tomar ganancias
                if market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.9:
                    sell_threshold = base_sell_confidence * 0.8  # 75% * 0.8 = 60% para mercado muy bullish
                    print(f"🎯 UMBRAL SELL ADAPTATIVO: {sell_threshold:.1f}% (base: {base_sell_confidence:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
                elif market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.7:
                    sell_threshold = base_sell_confidence * 0.85  # 75% * 0.85 = 63.75% para mercado bullish
                    print(f"🎯 UMBRAL SELL ADAPTATIVO: {sell_threshold:.1f}% (base: {base_sell_confidence:.1f}%) - Mercado BULLISH {market_context['confidence']:.1%}")
                else:
                    sell_threshold = max(base_sell_confidence, threshold * 0.85)  # SELL requiere al menos config% o 85% del umbral BUY

                if (signal == 'BUY' and confidence_level >= threshold) or (signal == 'SELL' and confidence_level >= sell_threshold):
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

                        # ✅ NUEVO: Información sobre umbral SELL aplicado
                        print(f"  📊 Señal SELL con confianza {confidence_level:.1f}% (umbral: {sell_threshold:.1f}%)")
                        print(f"  📋 {len(existing_positions)} posición(es) existente(s) serán evaluadas para cierre")

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

        # ✅ NUEVO: Implementar priorización de BTC en señales simultáneas
        if signals:
            print(f"🎯 Total señales TCN generadas: {len(signals)}")

            # Aplicar priorización: BTC > ETH > BNB > XRP
            prioritized_signals = self._apply_signal_prioritization(signals)

            print(f"⚖️ Señales después de priorización: {len(prioritized_signals)}")
            return prioritized_signals
        else:
            print("📊 No se generaron señales TCN válidas en este ciclo")
            # ✅ INFORMACIÓN: Mostrar por qué no se generaron señales válidas
            print(f"💡 Umbrales aplicados - BUY: {threshold:.1f}% | SELL: {sell_threshold:.1f}%")
            if market_context['regime'] == 'BULLISH' and market_context['confidence'] > 0.7:
                print(f"🎯 Umbrales relajados por mercado BULLISH ({market_context['confidence']:.1%})")
            return signals

    def _apply_signal_prioritization(self, signals: Dict) -> Dict:
        """
        🏆 SISTEMA DE PRIORIZACIÓN DE SEÑALES CON PRIVILEGIO PARA BTC
        ---
        En caso de señales simultáneas, prioriza BTC sobre todas las demás
        Orden de prioridad: BTC > ETH > BNB > XRP

        Lógica:
        - Si hay señal BUY de BTC, se ejecuta inmediatamente
        - Otras señales BUY se evalúan según disponibilidad de balance
        - Señales SELL no se priorizan (se ejecutan todas)
        """

        if not signals:
            return signals

        try:
            print(f"🏆 Aplicando priorización de señales...")

            # Orden de prioridad definido
            PRIORITY_ORDER = ['XRPUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT']

            # Separar señales BUY y SELL
            buy_signals = {symbol: data for symbol, data in signals.items() if data['signal'] == 'BUY'}
            sell_signals = {symbol: data for symbol, data in signals.items() if data['signal'] == 'SELL'}

            print(f"   📊 Señales BUY: {list(buy_signals.keys())}")
            print(f"   📊 Señales SELL: {list(sell_signals.keys())}")

            # ✅ SELL: Todas las señales SELL se procesan (no hay priorización)
            prioritized_signals = sell_signals.copy()

            # ✅ BUY: Aplicar priorización estricta
            if buy_signals:
                # Ordenar según prioridad
                sorted_buy_symbols = []
                for priority_symbol in PRIORITY_ORDER:
                    if priority_symbol in buy_signals:
                        sorted_buy_symbols.append(priority_symbol)

                # Agregar cualquier símbolo no contemplado al final
                for symbol in buy_signals:
                    if symbol not in sorted_buy_symbols:
                        sorted_buy_symbols.append(symbol)

                print(f"   🏆 Orden de prioridad BUY: {sorted_buy_symbols}")

                # ✅ PRIVILEGIO ABSOLUTO PARA BTC
                if 'BTCUSDT' in buy_signals:
                    btc_signal = buy_signals['BTCUSDT']
                    prioritized_signals['BTCUSDT'] = btc_signal
                    print(f"   👑 BTC PRIVILEGIADO: Señal BUY de BTCUSDT tiene prioridad absoluta")

                    # Verificar si hay balance suficiente para otros después de BTC
                    btc_confidence = btc_signal['confidence']
                    estimated_btc_size = min(18.0, max(10.0, (btc_confidence/100) * 20.0))
                    estimated_btc_value = (self.current_balance * estimated_btc_size / 100)
                    remaining_balance = self.current_balance - estimated_btc_value

                    print(f"   💰 Balance estimado después de BTC: ${remaining_balance:.2f}")

                    # Solo agregar otras señales BUY si queda balance significativo
                    min_position_value = 11.0
                    if remaining_balance >= min_position_value * 1.5:  # Buffer del 50%
                        # Agregar siguientes en orden de prioridad
                        for symbol in sorted_buy_symbols[1:]:  # Saltar BTC ya agregado
                            if symbol in buy_signals:
                                prioritized_signals[symbol] = buy_signals[symbol]
                                print(f"   ✅ Agregado {symbol} (prioridad {PRIORITY_ORDER.index(symbol) + 1})")
                    else:
                        print(f"   ⏸️ Otras señales BUY pausadas - Balance insuficiente después de BTC")

                else:
                    # No hay BTC, procesar en orden de prioridad normal
                    for symbol in sorted_buy_symbols:
                        prioritized_signals[symbol] = buy_signals[symbol]
                        print(f"   ✅ Agregado {symbol} (prioridad {PRIORITY_ORDER.index(symbol) + 1 if symbol in PRIORITY_ORDER else 'N/A'})")

            # ✅ ACTUALIZAR pending_signals para el sistema de diversificación
            self.pending_signals = {symbol: data for symbol, data in prioritized_signals.items() if data['signal'] == 'BUY'}

            # Mostrar resultado final
            if len(prioritized_signals) != len(signals):
                removed_signals = set(signals.keys()) - set(prioritized_signals.keys())
                print(f"   ⏸️ Señales pausadas por priorización: {list(removed_signals)}")

            print(f"   ✅ Señales finales priorizadas: {list(prioritized_signals.keys())}")

            return prioritized_signals

        except Exception as e:
            print(f"❌ Error en priorización de señales: {e}")
            # En caso de error, devolver señales originales
            return signals

    async def _analyze_market_context(self, prices: Dict[str, float]) -> Dict:
        """
        🌍 ANÁLISIS ROBUSTO DE CONTEXTO DE MERCADO

        Sistema mejorado que considera múltiples pares y factores de mercado
        para determinar el régimen actual de forma precisa.

        Returns:
            Dict con régimen, score, confianza y factores de riesgo
        """
        try:
            print("🔍 Analizando contexto de mercado robusto...")

            # Obtener datos históricos para análisis robusto
            market_data = {}
            for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']:
                try:
                    # Obtener datos de 5 minutos para análisis detallado
                    url = f"https://api.binance.com/api/v3/klines"
                    params = {
                        'symbol': symbol,
                        'interval': '5m',
                        'limit': 200  # 200 * 5m = ~16.7 horas de datos
                    }

                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                klines = await response.json()
                                market_data[symbol] = [float(k[4]) for k in klines]  # Precios de cierre
                except Exception as e:
                    print(f"  ⚠️ Error obteniendo datos para {symbol}: {e}")
                    market_data[symbol] = [prices.get(symbol, 0)] * 200  # Fallback

            # Usar el sistema robusto de detección de régimen
            regime, confidence = await self._detect_market_regime_robust(market_data)

            # Calcular métricas adicionales para compatibilidad
            fear_factor = 0.5  # Neutral por defecto
            correlation_strength = 0.5

            try:
                # Calcular volatilidad del mercado
                if 'BTCUSDT' in market_data and len(market_data['BTCUSDT']) > 24:
                    import numpy as np
                    btc_returns = np.diff(market_data['BTCUSDT']) / market_data['BTCUSDT'][:-1]
                    btc_volatility = np.std(btc_returns)
                    fear_factor = min(1, max(0, btc_volatility * 1000))  # Escalar a [0,1]

                # Calcular correlación entre pares principales
                if len(market_data) >= 2:
                    btc_returns = np.diff(market_data['BTCUSDT']) / market_data['BTCUSDT'][:-1]
                    eth_returns = np.diff(market_data['ETHUSDT']) / market_data['ETHUSDT'][:-1]

                    if len(btc_returns) > 10 and len(eth_returns) > 10:
                        min_len = min(len(btc_returns), len(eth_returns))
                        corr = np.corrcoef(btc_returns[:min_len], eth_returns[:min_len])[0, 1]
                        correlation_strength = max(0, min(1, (corr + 1) / 2))  # Normalizar a [0,1]
            except Exception as e:
                print(f"  ⚠️ Error calculando métricas adicionales: {e}")

            # Mapear régimen a score para compatibilidad
            if regime == 'BULLISH':
                score = 0.3 + (confidence - 0.5) * 0.4  # Score positivo
            elif regime == 'BEARISH':
                score = -0.3 - (confidence - 0.5) * 0.4  # Score negativo
            else:  # NEUTRAL
                score = 0.0

            # Nivel de volatilidad
            if fear_factor > 0.7:
                volatility_level = 'HIGH'
            elif fear_factor < 0.3:
                volatility_level = 'LOW'
            else:
                volatility_level = 'MEDIUM'

            context = {
                'regime': regime,
                'score': score,
                'confidence': confidence,
                'market_fear_factor': fear_factor,
                'trend_strength': abs(score),
                'volatility_level': volatility_level,
                'correlation_strength': correlation_strength,
                'altcoin_strength': 0.0,  # Calculado internamente por el sistema robusto
                'btc_trend_score': score,
                'components': {
                    'regime_system': 'ROBUST_MULTI_PAIR',
                    'pairs_analyzed': list(market_data.keys()),
                    'confidence': confidence,
                    'volatility': fear_factor
                }
            }

            print(f"  🎯 RÉGIMEN DETECTADO: {regime} (Confianza: {confidence:.2f})")
            print(f"  📊 Score compuesto: {score:.3f}")
            print(f"  😨 Factor miedo: {fear_factor:.3f}")
            print(f"  🔗 Correlación: {correlation_strength:.3f}")
            print(f"  ⚡ Volatilidad: {volatility_level}")

            return context

        except Exception as e:
            print(f"❌ Error en análisis robusto de contexto: {e}")
            # Contexto neutral por defecto
            return {
                'regime': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.5,
                'market_fear_factor': 0.5,
                'trend_strength': 0.0,
                'volatility_level': 'MEDIUM',
                'correlation_strength': 0.5,
                'altcoin_strength': 0.0,
                'btc_trend_score': 0.0,
                'components': {'error': str(e)}
            }

    def _apply_market_context_filter(self, signal: str, confidence: float, market_context: Dict, symbol: str) -> tuple:
        """
        🛡️ FILTRO DE CONTEXTO DE MERCADO INTEGRAL CON OPTIMIZACIONES
        ---
        Aplica filtros adaptativos basados en:
        1. Régimen de mercado (Bullish/Bearish/Neutral)
        2. Volatilidad y miedo del mercado
        3. Características específicas por activo
        4. Correlaciones y momentum

        Returns:
            tuple: (señal_filtrada, razón_del_filtro)
        """

        try:
            # Extraer información del contexto
            regime = market_context.get('regime', 'NEUTRAL')
            market_confidence = market_context.get('confidence', 0.0)
            market_score = market_context.get('score', 0.0)
            fear_factor = market_context.get('market_fear_factor', 0.5)
            volatility = market_context.get('volatility_level', 'MEDIUM')

            # Por defecto, no filtrar
            filter_reason = f"Sin filtro aplicado - {regime} con confianza {market_confidence:.1%}"

            # 🔴 FILTROS BEARISH - Restricciones de seguridad en mercado bajista (OPTIMIZADO INTEGRAL)
            if regime == 'BEARISH' and market_confidence > 0.7:
                if signal == 'BUY':
                    # ✅ UMBRALES DIFERENCIADOS POR ACTIVO EN BEARISH
                    required_confidence = {
                        'BTCUSDT': 80,   # BTC líder - umbral moderado
                        'ETHUSDT': 80,   # ETH principal altcoin - umbral medio
                        'BNBUSDT': 80,   # BNB exchange token - umbral más bajo
                        'XRPUSDT': 80    # XRP altcoin establecida - umbral medio-alto
                    }.get(symbol, 85)  # Otros activos: 88%

                    if confidence >= required_confidence:
                        filter_reason = f"{symbol.replace('USDT', '')} permitido en BEARISH por alta confianza ({confidence:.1f}% > {required_confidence}%)"
                    else:
                        signal = 'HOLD'
                        filter_reason = f"Mercado BEARISH fuerte (score: {market_score:.2f}) - {symbol} BUY requiere >{required_confidence}% confianza"

                elif signal == 'SELL':
                    # ✅ CORRECCIÓN: SELL en bearish requiere confianza razonable, no baja
                    min_sell_conf = float(os.getenv('MIN_SELL_CONFIDENCE_THRESHOLD', '0.75')) * 100
                    if confidence < min_sell_conf:  # Usar configuración dinámica
                        signal = 'HOLD'
                        filter_reason = f"SELL en BEARISH requiere >{min_sell_conf:.0f}% confianza (actual: {confidence:.1f}%)"
                    else:
                        filter_reason = f"SELL favorecido en mercado BEARISH con confianza {confidence:.1f}%"

            # 🟢 FILTROS BULLISH - Aprovechar momentum alcista (RELAJADOS)
            elif regime == 'BULLISH' and market_confidence > 0.7:
                if signal == 'BUY':
                    # ✅ RELAJADO: En bullish fuerte, ser más agresivo con BUY
                    if market_confidence > 0.9:  # Mercado BULLISH muy confiable
                        min_buy_confidence = 55  # Reducir a 55% para mercado muy bullish
                    else:
                        min_buy_confidence = 60  # Reducir a 62% para mercado bullish normal

                    if confidence < min_buy_confidence:
                        signal = 'HOLD'
                        filter_reason = f"BUY en BULLISH requiere >{min_buy_confidence}% confianza"
                    else:
                        filter_reason = f"BUY favorecido en mercado BULLISH (conf: {market_confidence:.1%})"

                elif signal == 'SELL':
                    # ✅ RELAJADO: En bullish extremo, relajar SELL para tomar ganancias
                    if market_confidence > 0.98:  # Mercado BULLISH muy confiable
                        min_sell_confidence = 75  # Reducir a 75% para mercado muy bullish
                    else:
                        min_sell_confidence = 80  # Reducir a 80% para mercado bullish normal

                    if confidence < min_sell_confidence:
                        signal = 'HOLD'
                        filter_reason = f"Mercado BULLISH (score: {market_score:.2f}) - SELL requiere >{min_sell_confidence}% confianza (actual: {confidence:.1f}%)"
                    else:
                        filter_reason = f"SELL permitido en BULLISH para tomar ganancias ({confidence:.1f}%)"

            # 🟡 FILTROS DE VOLATILIDAD - Ajustar según volatilidad del mercado (RELAJADOS EN BULLISH)
            if volatility == 'HIGH' and fear_factor > 0.8:
                # ✅ RELAJADO: En mercado BULLISH, la volatilidad puede ser oportunidad
                if regime == 'BULLISH' and market_confidence > 0.9:
                    # En mercado muy bullish, relajar filtros de volatilidad
                    volatility_thresholds = {
                        'BTCUSDT': 65,   # BTC: relajado de 78% a 65%
                        'ETHUSDT': 62,   # ETH: relajado de 75% a 62%
                        'BNBUSDT': 60,   # BNB: relajado de 72% a 60%
                        'XRPUSDT': 63    # XRP: relajado de 76% a 63%
                    }
                    vol_adjustment = "RELAJADO_BULLISH"
                else:
                    # Umbrales normales para otros contextos
                    volatility_thresholds = {
                        'BTCUSDT': 70,   # BTC más estable - umbral menor
                        'ETHUSDT': 70,   # ETH volátil - umbral medio
                        'BNBUSDT': 70,   # BNB exchange - umbral menor
                        'XRPUSDT': 70    # XRP moderadamente volátil - umbral medio
                    }
                    vol_adjustment = "NORMAL"

                if signal == 'BUY':
                    required_vol_confidence = volatility_thresholds.get(symbol, 80)
                    if confidence < required_vol_confidence:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad ({vol_adjustment}) - {symbol} BUY requiere >{required_vol_confidence}% confianza"
                elif signal == 'SELL':
                    # ✅ RELAJADO: En bullish extremo, relajar SELL en volatilidad
                    if regime == 'BULLISH' and market_confidence > 0.9:
                        min_sell_vol_conf = 80  # Relajar a 70% en mercado muy bullish
                    else:
                        min_sell_vol_conf = float(os.getenv('MIN_SELL_CONFIDENCE_THRESHOLD', '0.75')) * 100

                    if confidence < min_sell_vol_conf:
                        signal = 'HOLD'
                        filter_reason = f"Alta volatilidad ({vol_adjustment}) - SELL requiere >{min_sell_vol_conf:.0f}% confianza (actual: {confidence:.1f}%)"

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

            elif symbol in ['ETHUSDT', 'BNBUSDT', 'XRPUSDT']:
                # ✅ ALTCOINS OPTIMIZADO - Umbrales diferenciados por underperformance
                altcoin_strength = market_context.get('altcoin_strength', 0)
                if signal == 'BUY' and altcoin_strength < -0.2:  # Altcoins underperforming
                    # Umbrales específicos para cada altcoin
                    altcoin_thresholds = {
                        'ETHUSDT': 73,   # ETH principal altcoin - umbral medio
                        'BNBUSDT': 72,   # BNB exchange token - umbral menor
                        'XRPUSDT': 74    # XRP altcoin establecida - umbral medio-alto
                    }

                    required_alt_confidence = altcoin_thresholds.get(symbol, 75)

                    if confidence >= required_alt_confidence:
                        filter_reason = f"{symbol.replace('USDT', '')} permitido por alta confianza ({confidence:.1f}% > {required_alt_confidence}%) pese a underperformance"
                    else:
                        signal = 'HOLD'
                        filter_reason = f"Altcoins underperforming vs BTC - {symbol} BUY requiere >{required_alt_confidence}% confianza"

            return signal, filter_reason

        except Exception as e:
            print(f"⚠️ Error en filtro de contexto para {symbol}: {e}")
            return signal, f"Error en filtro: {e}"

    def _apply_signal_stability_filter(self, symbol: str, signal: str, confidence: float, current_price: float, market_context: Optional[Dict] = None) -> tuple:
        """
        🛡️ FILTRO DE ESTABILIDAD Y COOLDOWN PARA SEÑALES
        ---
        Previene cambios frecuentes de señal mediante:
        1. Sistema de cooldown por símbolo
        2. Confirmación de señales para ETH
        3. Protección contra ruido del modelo
        4. Validación de consistencia temporal

        Returns:
            tuple: (señal_filtrada, razón_del_filtro)
        """

        try:
            current_time = datetime.now()

            # Inicializar historial del símbolo si no existe
            if symbol not in self.signal_history:
                self.signal_history[symbol] = {
                    'last_signal': None,
                    'last_signal_time': None,
                    'signal_count': 0,
                    'consecutive_same_signal': 0
                }

            history = self.signal_history[symbol]
            cooldown_minutes = self.signal_cooldown.get(symbol, 10)

            # ✅ VERIFICAR COOLDOWN GENERAL
            if history['last_signal_time']:
                time_since_last = (current_time - history['last_signal_time']).total_seconds() / 60

                # Si estamos en cooldown y la señal cambió
                if time_since_last < cooldown_minutes and history['last_signal'] != signal:
                    # ✅ EXCEPCIÓN: Permitir si es señal HOLD (más conservador)
                    if signal != 'HOLD':
                        return 'HOLD', f"Cooldown activo: {time_since_last:.1f}min < {cooldown_minutes}min desde última señal {history['last_signal']}"

            # ✅ PROTECCIÓN ESPECÍFICA PARA ETH
            if symbol == 'ETHUSDT':
                eth_protection = self.eth_position_protection

                # Verificar si hay posiciones existentes de ETH
                existing_positions = self._get_positions_for_symbol(symbol)

                if existing_positions and signal == 'SELL':
                    # ✅ PROTECCIÓN: Tiempo mínimo de retención para posiciones ETH
                    for position in existing_positions:
                        # Obtener el tiempo de creación de la posición (approximation)
                        if hasattr(position, 'entry_time'):
                            position_age_minutes = (current_time - position.entry_time).total_seconds() / 60
                        else:
                            # Si no tenemos entry_time, usar última acción conocida
                            last_action_time = self.last_position_action.get(symbol, current_time - timedelta(minutes=eth_protection['min_hold_time_minutes']))
                            position_age_minutes = (current_time - last_action_time).total_seconds() / 60

                        if position_age_minutes < eth_protection['min_hold_time_minutes']:
                            # ✅ EXCEPCIÓN: Permitir SELL solo con confianza EXTREMA (>90%) y pérdida significativa
                            if confidence >= 90.0 and position.pnl_percent < -3.0:
                                filter_reason = f"ETH SELL permitido por confianza extrema ({confidence:.1f}%) y pérdida > 3%"
                                break
                            else:
                                return 'HOLD', f"ETH protegido: posición muy reciente ({position_age_minutes:.1f}min < {eth_protection['min_hold_time_minutes']}min)"

                # ✅ SISTEMA DE CONFIRMACIÓN PARA ETH
                # Requerir múltiples señales consecutivas iguales
                if history['last_signal'] == signal:
                    history['consecutive_same_signal'] += 1
                else:
                    history['consecutive_same_signal'] = 1

                required_confirmations = eth_protection['signal_confirmation_required']

                # Solo aplicar confirmación para señales de SELL en ETH
                if signal == 'SELL' and history['consecutive_same_signal'] < required_confirmations:
                    return 'HOLD', f"ETH SELL requiere {required_confirmations} confirmaciones consecutivas (actual: {history['consecutive_same_signal']})"

            # ✅ FILTRO DE CONFIANZA AUMENTADA PARA CAMBIOS DE SEÑAL (RELAJADO EN BULLISH)
            if history['last_signal'] and history['last_signal'] != signal:
                # ✅ RELAJADO: En mercado BULLISH muy confiable, relajar cambios de señal
                # Verificar si el mercado es muy bullish usando el contexto recibido
                market_is_very_bullish = market_context and \
                                       market_context.get('regime') == 'BULLISH' and \
                                       market_context.get('confidence', 0) > 0.9

                if market_is_very_bullish:
                    # Umbrales relajados para mercado muy bullish
                    min_confidence_for_change = {
                        'ETHUSDT': 65.0,  # ETH: relajado de 78% a 65%
                        'BTCUSDT': 62.0,  # BTC: relajado de 75% a 62%
                        'BNBUSDT': 60.0,  # BNB: relajado de 72% a 60%
                        'XRPUSDT': 63.0   # XRP: relajado de 75% a 63%
                    }.get(symbol, 65.0)
                    signal_context = "RELAJADO_BULLISH"
                else:
                    # Umbrales normales para otros contextos
                    min_confidence_for_change = {
                        'ETHUSDT': 78.0,  # ETH requiere 78% para cambiar señal
                        'BTCUSDT': 75.0,  # BTC requiere 75%
                        'BNBUSDT': 72.0,  # BNB requiere 72%
                        'XRPUSDT': 75.0   # XRP requiere 75%
                    }.get(symbol, 75.0)
                    signal_context = "NORMAL"

                if confidence < min_confidence_for_change:
                    return 'HOLD', f"Cambio de señal {history['last_signal']}→{signal} requiere >{min_confidence_for_change:.0f}% confianza (actual: {confidence:.1f}%) [{signal_context}]"

            # ✅ ACTUALIZAR HISTORIAL
            history['last_signal'] = signal
            history['last_signal_time'] = current_time
            history['signal_count'] += 1

            # Actualizar timestamp de última acción para el símbolo
            self.last_position_action[symbol] = current_time

            return signal, f"Señal estable: {signal} con {confidence:.1f}% confianza"

        except Exception as e:
            print(f"⚠️ Error en filtro de estabilidad para {symbol}: {e}")
            return signal, f"Error en filtro estabilidad: {e}"

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

        # ✅ NUEVO: Limpiar señales pendientes después de procesar
        if symbol in self.pending_signals:
            del self.pending_signals[symbol]
            print(f"  🧹 Señal pendiente limpiada para {symbol}")

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
        adjusted_position_usd = None
        try:
            # ✅ CORRECCIÓN: Usar el tamaño ajustado de la posición
            adjusted_position_usd = await self._check_portfolio_diversification_before_trade(symbol, signal_data)
            print(f"    ✅ PASO 1: Diversificación OK para {symbol}. Tamaño ajustado: ${adjusted_position_usd:.2f}")
        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e):
                print(f"    ❌ PASO 1: DIVERSIFICACIÓN BLOQUEÓ: {symbol}: {str(e)}")
                await self._send_discord_notification(f"❌ **DIVERSIFICACIÓN BLOQUEÓ**: {symbol}: {str(e)}")
                return  # Salir sin ejecutar el trade
            else:
                print(f"    ⚠️ PASO 1: Error verificando diversificación para {symbol}: {e}")
                # Continuar con el trade si es un error técnico
                adjusted_position_usd = None # Usar tamaño por defecto

        # Verificar límites de riesgo
        print(f"    🛡️ PASO 2: Verificando límites de riesgo para {symbol}...")
        can_trade = False
        risk_reason = ""
        if self.risk_manager:
            print(f"    🛡️ PASO 2: Risk manager disponible, ejecutando check_risk_limits_before_trade...")
            can_trade, risk_reason = await self.risk_manager.check_risk_limits_before_trade(symbol, signal, confidence)
            print(f"    🛡️ PASO 2: Risk Manager resultado: can_trade={can_trade}, reason='{risk_reason}'")
        else:
            print(f"    ❌ PASO 2: Risk manager no disponible")
            await self._send_discord_notification(f"❌ **RISK MANAGER NO DISPONIBLE**: {symbol}")
            return

        if not can_trade:
            print(f"    ❌ PASO 2: RISK MANAGER BLOQUEÓ: {symbol}: {risk_reason}")
            await self._send_discord_notification(f"❌ **RISK MANAGER BLOQUEÓ**: {symbol}: {risk_reason}")
            return

        # Abrir nueva posición
        print(f"    💰 PASO 3: Intentando abrir posición para {symbol}...")
        position = None
        if self.risk_manager:
            print(f"    💰 PASO 3: Risk manager disponible, ejecutando open_position...")
            # ✅ CORRECCIÓN CRÍTICA: Pasar el tamaño ajustado por diversificación al risk manager
            if adjusted_position_usd is not None:
                # Usar método especial que respeta el tamaño de diversificación
                position = await self._execute_position_with_diversification_size(
                    symbol, signal, confidence, current_price, adjusted_position_usd
                )
            else:
                # Usar método normal del risk manager
                position = await self.risk_manager.open_position(
                    symbol, signal, confidence, current_price
                )
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

            # ✅ CRÍTICO: Actualizar inmediatamente el registry del portfolio manager
            print(f"    🔄 PASO 4.1: Actualizando registry del portfolio manager...")

            # Crear posición compatible para el registry
            registry_position = PortfolioManagerPosition(
                symbol=position.symbol,
                side=position.side,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                market_value=position.quantity * position.current_price,
                unrealized_pnl_usd=0.0,  # Inicial
                unrealized_pnl_percent=0.0,  # Inicial
                entry_time=position.entry_time,
                duration_minutes=0,
                order_id=position.order_id,
                batch_id=position.order_id
            )

            # Inicializar stops para la nueva posición
            registry_position = self.portfolio_manager.initialize_position_stops(registry_position)

            # Agregar al registry inmediatamente
            self.portfolio_manager.position_registry[position.order_id] = registry_position
            print(f"    ✅ PASO 4.1: Posición agregada al registry con ID: {position.order_id}")

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

        # Si la señal es de venta, evaluar cada posición individualmente (NO cerrar automáticamente)
        if signal == 'SELL':
            print(f"📊 Señal de VENTA para {symbol} con {confidence:.1f}% confianza. Evaluando {len(existing_positions)} posición(es).")

            # ✅ CORRECCIÓN CRÍTICA: Evaluar cada posición según su rentabilidad actual
            for position in existing_positions:
                should_close = False
                close_reason = ""

                # ✅ PROTECCIÓN ESPECÍFICA PARA ETH - Criterios MÁS ESTRICTOS
                if symbol == 'ETHUSDT':
                    # ETH requiere criterios mucho más estrictos para cerrar posiciones
                    current_time = datetime.now()

                    # Verificar tiempo mínimo de retención
                    if hasattr(position, 'entry_time'):
                        position_age_minutes = (current_time - position.entry_time).total_seconds() / 60
                    else:
                        # Usar timestamp de última acción como aproximación
                        last_action_time = self.last_position_action.get(symbol, current_time - timedelta(minutes=30))
                        position_age_minutes = (current_time - last_action_time).total_seconds() / 60

                    min_hold_time = self.eth_position_protection['min_hold_time_minutes']

                    # CRITERIOS ESPECÍFICOS PARA ETH:
                    if confidence >= 90.0:  # Confianza EXTREMA (90%+)
                        if position.pnl_percent < -4.0:  # Pérdida significativa >4%
                            should_close = True
                            close_reason = "ETH_SELL_EXTREME_CONF_BIG_LOSS"
                        elif position.pnl_percent > 5.0 and position_age_minutes > min_hold_time:  # Ganancia >5% y tiempo suficiente
                            should_close = True
                            close_reason = "ETH_SELL_EXTREME_CONF_BIG_PROFIT"
                    elif confidence >= 85.0 and position_age_minutes > min_hold_time:  # Confianza muy alta y tiempo mínimo
                        if position.pnl_percent < -3.0:  # Pérdida >3%
                            should_close = True
                            close_reason = "ETH_SELL_HIGH_CONF_LOSS_PROTECTION"
                        elif position.pnl_percent > 6.0:  # Ganancia muy alta >6%
                            should_close = True
                            close_reason = "ETH_SELL_HIGH_CONF_PROFIT_TAKING"

                    if not should_close:
                        print(f"  🛡️ ETH PROTEGIDO: Manteniendo posición {position.order_id}")
                        print(f"      📊 PnL: {position.pnl_percent:.1f}%, Edad: {position_age_minutes:.1f}min, Conf: {confidence:.1f}%")
                        print(f"      ✅ ETH requiere criterios más estrictos para cierre de posición")

                # ✅ CRITERIOS ORIGINALES PARA OTROS SÍMBOLOS
                else:
                    # Criterios menos estrictos para BTC, BNB, XRP
                    if confidence >= 75.0:  # Confianza alta
                        if position.pnl_percent > 2.0:  # Si está en ganancia > 2%
                            should_close = True
                            close_reason = "SIGNAL_SELL_HIGH_CONF_PROFIT"
                        elif position.pnl_percent < -1.5:  # O pérdida > 1.5%
                            should_close = True
                            close_reason = "SIGNAL_SELL_HIGH_CONF_LOSS"
                    elif confidence >= 85.0:  # Confianza muy alta
                        should_close = True  # Cerrar independientemente del PnL
                        close_reason = "SIGNAL_SELL_VERY_HIGH_CONF"

                if should_close:
                    print(f"  🔥 Cerrando posición {position.order_id}: PnL {position.pnl_percent:.1f}% - {close_reason}")
                    await self._close_position(position.order_id, close_reason)
                else:
                    print(f"  ⏸️ Manteniendo posición {position.order_id}: PnL {position.pnl_percent:.1f}% - Confianza SELL insuficiente")

        # ✅ CORRECCIÓN: Lógica de reversión más estricta (solo para confianza extrema)
        # ETH requiere umbrales aún más altos para reversión
        reversal_threshold = 90.0 if symbol == 'ETHUSDT' else float(os.getenv('SIGNAL_REVERSAL_THRESHOLD', '0.90'))  # ETH: 90%, otros: config
        if confidence >= reversal_threshold:
            print(f"🔄 Evaluando reversión de señal con confianza extrema: {confidence:.1f}%")
            for position in existing_positions:
                if (position.side == 'BUY' and signal == 'SELL') or (position.side == 'SELL' and signal == 'BUY'):
                    # ✅ PROTECCIÓN ADICIONAL PARA ETH: Solo reversión si pérdida significativa
                    if symbol == 'ETHUSDT':
                        if position.pnl_percent <= -2.0:  # Solo si pérdida > 2% para ETH
                            print(f"  🔄 Reversión ETH ejecutada para {position.order_id}: PnL {position.pnl_percent:.1f}%")
                            await self._close_position(position.order_id, "ETH_SIGNAL_REVERSAL")
                        else:
                            print(f"  🛡️ Reversión ETH omitida para {position.order_id}: PnL no justifica reversión ({position.pnl_percent:.1f}%)")
                    else:
                        # ✅ LÓGICA ORIGINAL PARA OTROS SÍMBOLOS: Solo si no está en buena ganancia (>3%)
                        if position.pnl_percent <= 3.0:
                            print(f"  🔄 Reversión ejecutada para {position.order_id}: PnL {position.pnl_percent:.1f}%")
                            await self._close_position(position.order_id, "SIGNAL_REVERSAL")
                        else:
                            print(f"  ⏸️ Reversión omitida para {position.order_id}: PnL muy positivo {position.pnl_percent:.1f}%")

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

        # ✅ NUEVO: Actualizar timestamp de última acción para protección ETH
        if symbol == 'ETHUSDT':
            self.eth_position_protection['last_close_time'] = datetime.now()
            print(f"🛡️ ETH: Timestamp de cierre actualizado para protección")

        # Actualizar registro de última acción por símbolo
        self.last_position_action[symbol] = datetime.now()

        # Log y notificación
        color = "🟢" if pnl_usd > 0 else "🔴"
        await self.database.log_event(
            'INFO', 'TRADING',
            f'Posición cerrada: {symbol} - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})',
            symbol
        )

        # ✅ MEJORADO: Usar Smart Discord Notifier para cierres (consistencia)
        if hasattr(self, 'discord_notifier'):
            trade_notification_data = {
                'symbol': symbol,
                'side': position.side,
                'value_usd': position.quantity * position.entry_price,
                'price': current_price,
                'confidence': 0.0,  # No aplicable para cierres
                'pnl_percent': pnl_percent,
                'pnl_usd': pnl_usd,
                'reason': reason
            }
            await self.discord_notifier.send_trade_notification(trade_notification_data)
        else:
            # Fallback al sistema básico
            await self._send_discord_notification(f"{color} **POSICIÓN CERRADA**\n"
                                                 f"📊 {symbol}: {position.side}\n"
                                                 f"📈 PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})\n"
                                                 f"🔄 Razón: {reason}")

        print(f"📉 Posición cerrada: {symbol} (ID de orden: {order_id}) - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})")

    async def _check_portfolio_diversification_before_trade(self, symbol: str, signal_data: Dict) -> Optional[float]:
        """
        🎯 SISTEMA DE DIVERSIFICACIÓN INTELIGENTE CON LÍMITES POR PAR
        ---
        Implementa límites específicos de exposición por activo y priorización de BTC

        Límites por par:
        - BTC: máximo 50% del portafolio
        - ETH: máximo 20% del portafolio
        - BNB: máximo 15% del portafolio
        - XRP: máximo 15% del portafolio

        Priorización: BTC > ETH > BNB > XRP en señales simultáneas

        Returns:
            Optional[float]: El tamaño ajustado de la posición en USD si es válida,
                             o lanza una excepción si se bloquea.
        """

        try:
            print(f"🎯 Verificación de diversificación inteligente para {symbol}")

            # ✅ CRÍTICO: Actualizar balance antes de verificar diversificación
            await self.update_balance_from_binance()

            # Obtener snapshot actual del portafolio
            if self.portfolio_manager is None:
                raise Exception("Portfolio Manager no inicializado")
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()

            # ✅ CONFIGURACIÓN: Límites específicos por par
            SYMBOL_LIMITS = {
                'BTCUSDT': 10.0,  # BTC máximo 35% del portafolio (reducido de 50%)
                'ETHUSDT': 25.0,  # ETH máximo 15% del portafolio (reducido de 20%)
                'BNBUSDT': 30.0,  # BNB máximo 25% del portafolio (aumentado de 15%)
                'XRPUSDT': 30.0   # XRP máximo 15% del portafolio (sin cambios)
            }

            # ✅ PRIORIZACIÓN: Orden de preferencia para señales simultáneas
            PRIORITY_ORDER = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']

            # ✅ DEBUG: Información del balance
            print(f"🔍 DEBUG DIVERSIFICACIÓN INTELIGENTE:")
            print(f"   💰 Balance actual: ${self.current_balance:.2f}")
            print(f"   📊 Posiciones activas: {len(snapshot.active_positions)}")

            # Calcular exposición actual por símbolo
            symbol_exposures = {}
            total_invested_value = 0.0  # Total invertido originalmente
            total_current_value = 0.0   # Valor actual de mercado

            for pos in snapshot.active_positions:
                symbol_key = pos.symbol
                if symbol_key not in symbol_exposures:
                    symbol_exposures[symbol_key] = {
                        'positions': [],
                        'total_invested': 0.0,
                        'current_value': 0.0,
                        'count': 0
                    }

                symbol_exposures[symbol_key]['positions'].append(pos)
                symbol_exposures[symbol_key]['count'] += 1

                # Valor invertido originalmente (entry_price * quantity)
                invested_value = pos.quantity * pos.entry_price
                symbol_exposures[symbol_key]['total_invested'] += invested_value
                total_invested_value += invested_value

                # Valor actual de mercado (market_value ya calculado)
                current_market_value = pos.market_value
                symbol_exposures[symbol_key]['current_value'] += current_market_value
                total_current_value += current_market_value

                print(f"   📍 {pos.symbol}: {pos.quantity:.6f} x ${pos.entry_price:.4f} = ${invested_value:.2f} (actual: ${current_market_value:.2f})")

            # ✅ CORRECCIÓN CRÍTICA: Total del portafolio = balance disponible + valor actual de mercado
            total_portfolio_value = self.current_balance + total_current_value

            # ✅ DEBUG: Mostrar cálculos correctos
            print(f"🔍 CÁLCULO DE PORTAFOLIO CORREGIDO:")
            print(f"   💰 Balance disponible: ${self.current_balance:.2f}")
            print(f"   📊 Valor actual invertido: ${total_current_value:.2f}")
            print(f"   💼 Total portafolio: ${total_portfolio_value:.2f}")
            print(f"   📈 Total originalmente invertido: ${total_invested_value:.2f}")

            # ✅ VALIDACIÓN 1: Máximo 3 posiciones por símbolo
            current_positions_count = symbol_exposures.get(symbol, {}).get('count', 0)
            if current_positions_count >= 3:
                print(f"🚫 DIVERSIFICACIÓN: Máximo 3 posiciones por símbolo alcanzado para {symbol}")
                print(f"   📊 Posiciones actuales en {symbol}: {current_positions_count}/3")
                raise Exception(f"Trade bloqueado por diversificación: Máximo 3 posiciones por símbolo en {symbol}")

            # ✅ CÁLCULO: Tamaño de nueva posición propuesta
            confidence = signal_data['confidence']

            # Tamaño base según confianza (entre 10% y 18% del balance disponible)
            base_size_percent = min(18.0, max(10.0, (confidence/100) * 20.0))
            proposed_position_usd = (self.current_balance * base_size_percent / 100)

            # ✅ VALIDACIÓN 2: Límites específicos por símbolo
            symbol_limit = SYMBOL_LIMITS.get(symbol, 10.0)  # Default 10% para otros

            # ✅ CORRECCIÓN CRÍTICA: Usar valor actual de mercado para exposición
            current_symbol_market_value = symbol_exposures.get(symbol, {}).get('current_value', 0.0)
            new_total_symbol_value = current_symbol_market_value + proposed_position_usd
            new_symbol_exposure_percent = (new_total_symbol_value / (total_portfolio_value + proposed_position_usd)) * 100 if total_portfolio_value > 0 else 0

            # Exposición actual sin la nueva posición
            current_symbol_exposure_percent = (current_symbol_market_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0

            # ✅ VALIDACIÓN CRÍTICA: Verificar si ya excede el límite SIN nueva posición
            if current_symbol_exposure_percent > symbol_limit:
                print(f"🚫 DIVERSIFICACIÓN: {symbol} YA EXCEDE el límite de exposición")
                print(f"   📊 Exposición actual: {current_symbol_exposure_percent:.1f}% > {symbol_limit}%")
                print(f"   💰 Valor actual en {symbol}: ${current_symbol_market_value:.2f}")
                print(f"   💼 Total portafolio: ${total_portfolio_value:.2f}")
                print(f"   🚫 NO SE PERMITE nueva posición hasta reducir exposición")
                raise Exception(f"Trade bloqueado por diversificación: {symbol} ya excede límite de {symbol_limit}% (actual: {current_symbol_exposure_percent:.1f}%)")

            if new_symbol_exposure_percent > symbol_limit:
                # Calcular el máximo permitido para este símbolo
                max_allowed_total_value = (total_portfolio_value + proposed_position_usd) * symbol_limit / 100
                max_new_position = max_allowed_total_value - current_symbol_market_value

                if max_new_position <= 11.0:  # Mínimo Binance
                    print(f"🚫 DIVERSIFICACIÓN: Nueva posición excedería límite para {symbol}")
                    print(f"   📊 Exposición actual: {current_symbol_exposure_percent:.1f}%")
                    print(f"   📊 Nueva exposición: {new_symbol_exposure_percent:.1f}% > {symbol_limit}%")
                    print(f"   💰 Máximo permitido: ${max_new_position:.2f} (mínimo: $11)")
                    raise Exception(f"Trade bloqueado por diversificación: {symbol} excedería límite de {symbol_limit}%")
                else:
                    # Ajustar el tamaño de la posición al máximo permitido
                    proposed_position_usd = max_new_position
                    print(f"⚖️ AJUSTE DE DIVERSIFICACIÓN: {symbol} ajustado a ${proposed_position_usd:.2f}")

            # ✅ VALIDACIÓN 3: Priorización en señales simultáneas
            # Verificar si hay otras señales pendientes con mayor prioridad
            if hasattr(self, 'pending_signals') and self.pending_signals:
                current_priority = PRIORITY_ORDER.index(symbol) if symbol in PRIORITY_ORDER else len(PRIORITY_ORDER)

                for pending_symbol in self.pending_signals:
                    if pending_symbol in PRIORITY_ORDER:
                        pending_priority = PRIORITY_ORDER.index(pending_symbol)
                        if pending_priority < current_priority:
                            print(f"⏸️ PRIORIZACIÓN: {symbol} pausado, {pending_symbol} tiene mayor prioridad")
                            print(f"   📊 Orden de prioridad: {' > '.join(PRIORITY_ORDER)}")
                            raise Exception(f"Trade pausado por priorización: {pending_symbol} tiene mayor prioridad que {symbol}")

            # ✅ VALIDACIÓN 4: Exposición total del portafolio
            total_exposure_percent = ((total_portfolio_value - self.current_balance) / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
            max_total_exposure = 85.0  # Máximo 85% del portafolio invertido

            if confidence >= 80.0:
                max_total_exposure = 90.0  # Permitir hasta 90% con alta confianza

            new_total_exposure = total_exposure_percent + (proposed_position_usd / total_portfolio_value) * 100

            if new_total_exposure > max_total_exposure:
                print(f"🚫 DIVERSIFICACIÓN: Exposición total muy alta")
                print(f"   📊 Exposición actual: {total_exposure_percent:.1f}%")
                print(f"   📊 Nueva exposición: {new_total_exposure:.1f}% > {max_total_exposure:.0f}%")
                raise Exception(f"Trade bloqueado por diversificación: Exposición total > {max_total_exposure:.0f}%")

            # ✅ INFORMACIÓN: Mostrar análisis detallado
            print(f"✅ DIVERSIFICACIÓN: Trade aprobado para {symbol}")
            print(f"   🎯 Límite específico: {symbol_limit}% (BTC:35%, ETH:25%, BNB:25%, XRP:15%)")
            print(f"   📊 Exposición actual en {symbol}: {current_symbol_exposure_percent:.1f}%")
            print(f"   📊 Nueva exposición en {symbol}: {new_symbol_exposure_percent:.1f}%/{symbol_limit}%")
            print(f"   💰 Tamaño propuesto: ${proposed_position_usd:.2f} ({base_size_percent:.1f}%)")
            print(f"   📊 Posiciones en {symbol}: {current_positions_count}/3")
            print(f"   🔥 Confianza: {confidence:.1f}%")
            print(f"   🏆 Prioridad: {PRIORITY_ORDER.index(symbol) + 1 if symbol in PRIORITY_ORDER else 'N/A'}")

            # ✅ RESUMEN: Estado del portafolio (CORREGIDO)
            if symbol_exposures:
                print(f"📊 ESTADO ACTUAL DEL PORTAFOLIO:")
                for sym in PRIORITY_ORDER:
                    if sym in symbol_exposures:
                        exposure = symbol_exposures[sym]
                        # ✅ CORRECCIÓN: Usar valor actual de mercado para exposición
                        current_market_value = exposure['current_value']
                        exposure_percent = (current_market_value / total_portfolio_value) * 100
                        limit = SYMBOL_LIMITS[sym]
                        status = "✅" if exposure_percent <= limit * 0.8 else "⚠️" if exposure_percent <= limit else "🔴"
                        print(f"   {status} {sym}: {exposure['count']} pos, {exposure_percent:.1f}%/{limit}% (${current_market_value:.2f} actual, ${exposure['total_invested']:.2f} invertido)")

            # ✅ DEVOLVER TAMAÑO AJUSTADO
            return proposed_position_usd

        except Exception as e:
            if "Trade bloqueado por diversificación" in str(e) or "Trade pausado por priorización" in str(e):
                # Bloqueos legítimos de diversificación
                print(f"🚫 {str(e)}")
                if self.database:
                    await self.database.log_event('WARNING', 'DIVERSIFICATION', str(e), symbol)

                # Notificación Discord informativa
                if self.discord_notifier:
                    await self.discord_notifier.send_system_notification(
                        f"⚖️ **DIVERSIFICACIÓN INTELIGENTE**\n"
                        f"📊 {symbol}: {signal_data['signal']}\n"
                        f"💡 {str(e).replace('Trade bloqueado por diversificación: ', '').replace('Trade pausado por priorización: ', '')}\n"
                        f"🎯 Confianza: {signal_data['confidence']:.1%}\n"
                        f"🏆 Límites: BTC≤35%, ETH≤25%, BNB≤25%, XRP≤15%"
                    )

                raise  # Re-lanzar bloqueos legítimos
            else:
                # Errores técnicos no deben bloquear trades
                print(f"⚠️ Error técnico en diversificación (ignorado): {e}")
                print(f"✅ Continuando con el trade para {symbol}")
                return None # Devolver None para que se use el tamaño por defecto

    async def _heartbeat_monitor(self):
        """💓 Monitor de latido del sistema"""
        while self.status == TradingManagerStatus.RUNNING:
            try:
                # Verificar conectividad cada 5 minutos
                await asyncio.sleep(180)

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
                stop_loss_threshold = float(os.getenv('STOP_LOSS_PERCENT', '1.4'))
                if pnl_percent <= -stop_loss_threshold:
                    return True, f"STOP_LOSS_TRADICIONAL (-{abs(pnl_percent):.2f}%)"

                # Take Profit tradicional
                take_profit_threshold = float(os.getenv('TAKE_PROFIT_PERCENT', '4.0'))
                if pnl_percent >= take_profit_threshold:
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
                self.binance_config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            params['signature'] = signature

            # Headers de autenticación
            headers = {
                'X-MBX-APIKEY': self.binance_config.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            print(f"📡 Ejecutando orden de cierre: {close_side} {params['quantity']} {position.symbol}")

            # Ejecutar orden POST /api/v3/order
            async with aiohttp.ClientSession() as session:
                url = f"{self.binance_config.base_url}/api/v3/order"

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
                url = f"{self.binance_config.base_url}/api/v3/exchangeInfo"
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
            trades_today = await self._get_total_trades_today()
            win_rate = await self._calculate_win_rate()

            # ✅ CORRECCIÓN: Usar el snapshot del portfolio manager para obtener datos consistentes y reales
            if self.portfolio_manager:
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                total_exposure = snapshot.total_balance_usd - snapshot.free_usdt
                total_balance = snapshot.total_balance_usd
                # El PnL de la sesión es el PnL no realizado del snapshot
                self.session_pnl = snapshot.total_unrealized_pnl
            else:
                # Fallback por si el portfolio manager no está listo (menos preciso)
                total_exposure = 0
                for position in self.active_positions.values():
                    if hasattr(position, 'current_price') and position.current_price > 0:
                        total_exposure += position.quantity * position.current_price
                    else:
                        total_exposure += position.quantity * position.entry_price
                total_balance = self.current_balance + total_exposure

            # ✅ CORRECCIÓN: Usar el balance total real para el cálculo de exposición
            exposure_percent = (total_exposure / total_balance) * 100 if total_balance > 0 else 0

            # Actualizar balance actual en risk manager con el valor correcto
            if self.risk_manager:
                await self.risk_manager.update_balance(total_balance)

            metrics_data = {
                'timestamp': datetime.now(),
                'total_balance': total_balance,
                'daily_pnl': self.session_pnl,
                'total_pnl': self.session_pnl,  # Para sesión actual
                'daily_return_percent': (self.session_pnl / total_balance) * 100 if total_balance > 0 else 0,
                'total_return_percent': (self.session_pnl / total_balance) * 100 if total_balance > 0 else 0,
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

            if self.database:
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
            'environment': self.binance_config.environment,
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

    async def _detect_market_regime_robust(self, market_data: Dict[str, List[float]]) -> Tuple[str, float]:
        """
        🔍 DETECCIÓN ROBUSTA DE RÉGIMEN DE MERCADO - VERSIÓN CORREGIDA
        Sistema mejorado más sensible a mercados bajistas graduales

        CORRECCIÓN: Ajustado para detectar mejor mercados bajistas en los últimos días
        """
        try:
            print(f"🔍 Detectando régimen de mercado robusto (versión corregida)...")

            # 1. ANÁLISIS MULTI-PAR
            pair_regimes = {}
            total_signals = 0
            bullish_signals = 0
            bearish_signals = 0

            for symbol, prices in market_data.items():
                if len(prices) < 50:  # Necesitamos datos suficientes
                    continue

                # Convertir a pandas para análisis técnico
                df = pd.DataFrame({'close': prices})

                # === INDICADORES POR PAR ===

                # 1. Momentum multitimeframe (UMBRALES REDUCIDOS)
                df['momentum_1h'] = df['close'].pct_change(12)   # 12 * 5m = 1h
                df['momentum_4h'] = df['close'].pct_change(48)   # 48 * 5m = 4h
                df['momentum_12h'] = df['close'].pct_change(144) # 144 * 5m = 12h
                df['momentum_24h'] = df['close'].pct_change(288) # 288 * 5m = 24h (NUEVO)

                # 2. Medias móviles
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['ema_20'] = df['close'].ewm(span=20).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()

                # 3. Trend strength
                df['ma_trend'] = (df['close'] - df['sma_20']) / df['sma_20']
                df['ma_direction'] = df['sma_20'] > df['sma_50']
                df['ema_trend'] = (df['close'] - df['ema_20']) / df['ema_20']

                # 4. Volatilidad relativa
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(20).std()
                df['vol_percentile'] = df['volatility'].rolling(100).rank(pct=True)

                # 5. RSI para momentum
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))

                # 6. NUEVO: Análisis de tendencia reciente (últimos 2 días)
                df['recent_trend_2d'] = df['close'].pct_change(576)  # 576 * 5m = 48h
                df['recent_slope'] = df['close'].rolling(144).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 144 else 0, raw=True)

                # === CLASIFICACIÓN POR PAR (MEJORADA) ===
                latest = df.iloc[-1]

                # Contar señales bullish y bearish
                bullish_count = 0
                bearish_count = 0

                # ✅ CORREGIDO: Umbrales de momentum MÁS SENSIBLES
                # Momentum 4h (reducido de 2% a 1%)
                if not pd.isna(latest['momentum_4h']):
                    if latest['momentum_4h'] > 0.01:  # +1% (era +2%)
                        bullish_count += 2
                    elif latest['momentum_4h'] < -0.01:  # -1% (era -2%)
                        bearish_count += 2

                # Momentum 12h (reducido de 5% a 2.5%)
                if not pd.isna(latest['momentum_12h']):
                    if latest['momentum_12h'] > 0.025:  # +2.5% (era +5%)
                        bullish_count += 3
                    elif latest['momentum_12h'] < -0.025:  # -2.5% (era -5%)
                        bearish_count += 3

                # ✅ NUEVO: Momentum 24h para detectar tendencias de 1-2 días
                if not pd.isna(latest['momentum_24h']):
                    if latest['momentum_24h'] > 0.03:  # +3% en 24h
                        bullish_count += 4
                    elif latest['momentum_24h'] < -0.03:  # -3% en 24h (CRÍTICO)
                        bearish_count += 4

                # ✅ NUEVO: Trend reciente de 2 días (MUY IMPORTANTE)
                if not pd.isna(latest['recent_trend_2d']):
                    if latest['recent_trend_2d'] > 0.04:  # +4% en 2 días
                        bullish_count += 3
                    elif latest['recent_trend_2d'] < -0.04:  # -4% en 2 días
                        bearish_count += 5  # PESO EXTRA para tendencias bajistas de 2 días

                # Señales de MA (umbrales reducidos)
                if not pd.isna(latest['ma_trend']):
                    if latest['ma_trend'] > 0.005 and latest['ma_direction']:  # +0.5% (era +1%)
                        bullish_count += 2
                    elif latest['ma_trend'] < -0.005 and not latest['ma_direction']:  # -0.5% (era -1%)
                        bearish_count += 2

                # ✅ NUEVO: EMA trend adicional
                if not pd.isna(latest['ema_trend']):
                    if latest['ema_trend'] < -0.01:  # -1% respecto a EMA
                        bearish_count += 2

                # RSI con contexto de momentum
                if not pd.isna(latest['rsi']):
                    if latest['rsi'] > 70 and not pd.isna(latest['momentum_1h']) and latest['momentum_1h'] > 0:
                        bullish_count += 1
                    elif latest['rsi'] < 30 and not pd.isna(latest['momentum_1h']) and latest['momentum_1h'] < 0:
                        bearish_count += 1
                    # ✅ NUEVO: RSI en zona de venta pero no extrema
                    elif 45 < latest['rsi'] < 55 and not pd.isna(latest['momentum_4h']) and latest['momentum_4h'] < -0.005:
                        bearish_count += 1  # Momentum bajista con RSI neutral

                # ✅ NUEVO: Pendiente reciente negativa
                if not pd.isna(latest['recent_slope']) and latest['recent_slope'] < -0.5:
                    bearish_count += 2

                # ✅ CLASIFICACIÓN MEJORADA: Más sensible a bearish
                if bearish_count >= bullish_count:  # Cambio: era > bullish_count + 1
                    pair_regime = 'BEARISH'
                    bearish_signals += bearish_count
                elif bullish_count > bearish_count + 1:  # Mantener umbral para bullish
                    pair_regime = 'BULLISH'
                    bullish_signals += bullish_count
                else:
                    pair_regime = 'NEUTRAL'

                pair_regimes[symbol] = {
                    'regime': pair_regime,
                    'bullish_signals': bullish_count,
                    'bearish_signals': bearish_count,
                    'momentum_4h': latest['momentum_4h'] if not pd.isna(latest['momentum_4h']) else 0,
                    'momentum_12h': latest['momentum_12h'] if not pd.isna(latest['momentum_12h']) else 0,
                    'momentum_24h': latest['momentum_24h'] if not pd.isna(latest['momentum_24h']) else 0,
                    'recent_trend_2d': latest['recent_trend_2d'] if not pd.isna(latest['recent_trend_2d']) else 0,
                    'ma_trend': latest['ma_trend'] if not pd.isna(latest['ma_trend']) else 0,
                    'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 50
                }

                total_signals += bullish_count + bearish_count

                print(f"   📊 {symbol}: {pair_regime} (Bull: {bullish_count}, Bear: {bearish_count})")
                # ✅ CORREGIDO: Mostrar valores reales en lugar de NaN
                mom_4h_display = latest['momentum_4h'] if not pd.isna(latest['momentum_4h']) else 0.0
                mom_24h_display = latest['momentum_24h'] if not pd.isna(latest['momentum_24h']) else 0.0
                trend_2d_display = latest['recent_trend_2d'] if not pd.isna(latest['recent_trend_2d']) else 0.0
                print(f"      📈 Mom4h: {mom_4h_display:.3f} | Mom24h: {mom_24h_display:.3f} | Trend2d: {trend_2d_display:.3f}")

            # 2. ANÁLISIS AGREGADO DEL MERCADO

            # Contar regímenes por par
            regime_votes = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
            for data in pair_regimes.values():
                regime_votes[data['regime']] += 1

            total_pairs = len(pair_regimes)
            if total_pairs == 0:
                return 'NEUTRAL', 0.5

            # 3. MÉTRICAS AGREGADAS DE CONFIANZA

            # Ratio de señales
            if total_signals > 0:
                bullish_ratio = bullish_signals / total_signals
                bearish_ratio = bearish_signals / total_signals
            else:
                bullish_ratio = bearish_ratio = 0

            # Consenso entre pares
            max_votes = max(regime_votes.values())
            consensus_strength = max_votes / total_pairs

            # 4. ✅ CLASIFICACIÓN FINAL CORREGIDA

            # ✅ CORRECCIÓN CRÍTICA: Si hay unanimidad BEARISH (4 votos), debe ser BEARISH
            if regime_votes['BEARISH'] == total_pairs and regime_votes['BEARISH'] > 0:
                final_regime = 'BEARISH'
                confidence = 0.95  # Alta confianza en unanimidad
                print(f"   🔴 UNANIMIDAD BEARISH: {regime_votes['BEARISH']}/{total_pairs} pares → BEARISH forzado")

            # Umbrales REDUCIDOS para mayor sensibilidad (mantener lógica original mejorada)
            elif (regime_votes['BEARISH'] >= regime_votes['BULLISH'] and
                  bearish_ratio > 0.3 and  # Reducido aún más de 0.45 a 0.3
                  consensus_strength > 0.3):  # Reducido de 0.4 a 0.3
                final_regime = 'BEARISH'
                confidence = min(0.95, 0.6 + (bearish_ratio - 0.3) * 0.8 + (consensus_strength - 0.3) * 0.6)

            elif (regime_votes['BULLISH'] > regime_votes['BEARISH'] + 1 and
                  bullish_ratio > 0.55 and  # Mantener umbral más alto para bullish
                  consensus_strength > 0.5):  # Mantener umbral más alto para bullish
                final_regime = 'BULLISH'
                confidence = min(0.95, 0.5 + (bullish_ratio - 0.5) + (consensus_strength - 0.5))

            else:
                final_regime = 'NEUTRAL'
                confidence = 0.5 + (1 - consensus_strength) * 0.3  # Mayor incertidumbre

            # 5. ✅ OVERRIDE PARA MERCADOS CLARAMENTE BAJISTAS
            # Si la mayoría de pares muestran tendencia bajista de 2 días, forzar BEARISH
            valid_trends = [data['recent_trend_2d'] for data in pair_regimes.values() if data['recent_trend_2d'] != 0]
            if valid_trends:
                avg_trend_2d = np.mean(valid_trends)
                if avg_trend_2d < -0.03 and final_regime == 'NEUTRAL':  # -3% promedio en 2 días
                    final_regime = 'BEARISH'
                    confidence = 0.75
                    print(f"   🔴 OVERRIDE: Tendencia 2d promedio {avg_trend_2d:.3f} < -3% → BEARISH forzado")
            else:
                avg_trend_2d = 0.0

            # 6. LOGGING DETALLADO
            print(f"   📊 Votos por régimen: {regime_votes}")
            print(f"   📊 Ratio señales: Bull {bullish_ratio:.2f}, Bear {bearish_ratio:.2f}")
            print(f"   📊 Consenso: {consensus_strength:.2f}")
            print(f"   📊 Tendencia 2d promedio: {avg_trend_2d:.3f}")
            print(f"   🎯 RÉGIMEN FINAL: {final_regime} (Confianza: {confidence:.2f})")

            # Mostrar detalles por par
            for symbol, data in pair_regimes.items():
                print(f"     {symbol}: {data['regime']} | Mom4h: {data['momentum_4h']:.3f} | "
                      f"Mom24h: {data['momentum_24h']:.3f} | Trend2d: {data['recent_trend_2d']:.3f} | "
                      f"MA: {data['ma_trend']:.3f} | RSI: {data['rsi']:.1f}")

            return final_regime, confidence

        except Exception as e:
            print(f"❌ Error en detección de régimen: {e}")
            return 'NEUTRAL', 0.5

    async def _execute_position_with_diversification_size(self, symbol: str, side: str, confidence: float,
                                                        current_price: float, position_size_usd: float) -> Optional:
        """💰 Ejecutar posición con tamaño específico calculado por diversificación"""
        try:
            print(f"🎯 EJECUTANDO POSICIÓN CON TAMAÑO DE DIVERSIFICACIÓN:")
            print(f"   📊 {symbol}: {side}")
            print(f"   💰 Tamaño USD: ${position_size_usd:.2f}")
            print(f"   💲 Precio: ${current_price:.4f}")

            # Calcular cantidad basada en el tamaño USD especificado por diversificación
            quantity = position_size_usd / current_price
            print(f"   🔢 Cantidad calculada: {quantity:.8f}")

            # Usar el método interno del risk manager para ejecutar la orden real
            order_result = await self.risk_manager._execute_real_order(symbol, side, quantity)

            if not order_result or 'orderId' not in order_result:
                print(f"❌ Falló la ejecución de la orden real para {symbol}. No se abre posición.")
                return None

            real_entry_price = float(order_result.get('fills', [{}])[0].get('price', current_price))
            real_quantity = float(order_result.get('executedQty', quantity))
            order_id = str(order_result['orderId'])

            print(f"🎉 Orden real ejecutada para {symbol}: ID {order_id}")
            print(f"   - Precio Real: ${real_entry_price:.4f}, Cantidad Real: {real_quantity:.6f}")

            # Crear posición usando la misma estructura que AdvancedRiskManager
            from advanced_risk_manager import Position
            from datetime import timezone

            position = Position(
                symbol=symbol,
                side=side,
                quantity=real_quantity,
                entry_price=real_entry_price,
                current_price=real_entry_price,
                entry_time=datetime.now(timezone.utc),
                stop_loss=real_entry_price * (1 - self.risk_manager.limits.stop_loss_percent / 100),
                take_profit=real_entry_price * (1 + self.risk_manager.limits.take_profit_percent / 100),
                trade_id=order_id,
                order_id=order_id,
                is_active=True
            )

            # Registrar la posición en el risk manager
            self.risk_manager.active_positions[order_id] = position

            # Actualizar balance del risk manager
            await self.risk_manager.update_balance(self.risk_manager.current_balance - (real_quantity * real_entry_price))

            print(f"✅ POSICIÓN CREADA CON TAMAÑO DE DIVERSIFICACIÓN: {symbol} - Order ID: {order_id}")
            print(f"   💰 Valor real invertido: ${real_quantity * real_entry_price:.2f}")

            return position

        except Exception as e:
            print(f"❌ Error ejecutando posición con tamaño de diversificación para {symbol}: {e}")
            return None

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
