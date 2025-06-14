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
import numpy as np
import random

# Importar nuestros módulos de risk y database
from advanced_risk_manager import AdvancedRiskManager, Position, RiskLimits
from trading_database import TradingDatabase

# Importar el módulo de Smart Discord Notifier
from smart_discord_notifier import SmartDiscordNotifier

# ✅ NUEVO: Importar Professional Portfolio Manager
from professional_portfolio_manager import ProfessionalPortfolioManager

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

class TradingManager:
    """🚀 Trading Manager Profesional Simplificado"""
    
    def __init__(self):
        """🚀 Inicializar Trading Manager"""
        print("🚀 Simple Professional Trading Manager inicializado")
        
        # Configuración básica
        self.config = self._load_config()
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.check_interval = 60  # 1 minuto
        
        # Estado del sistema
        self.status = TradingManagerStatus.STOPPED
        self.database = None
        self.risk_manager = None
        self.client = None
        
        # ✅ CRÍTICO: Cliente Binance para datos reales
        self.binance_client = None
        
        # ✅ NUEVO: Professional Portfolio Manager
        self.portfolio_manager = None
        
        # Balance y trading - ✅ CORREGIDO: Inicializar en 0, obtener de Binance
        self.current_balance = 0.0  # Se actualizará desde Binance
        self.session_pnl = 0.0
        self.trade_count = 0
        self.active_positions = {}
        self.account_info = None
        
        # ✅ NUEVO: Portfolio tracking
        self.last_portfolio_snapshot = None
        self.last_tcn_report_time = None
        
        # Smart Discord Notifier
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
        
        # Métricas
        self.metrics = {
            'total_checks': 0,
            'successful_checks': 0,
            'error_count': 0,
            'active_positions': 0,
            'session_pnl': 0.0
        }
        
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
        
        # Métricas en tiempo real
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
            'tcn_reports_sent': 0
        }
        
        # ✅ NUEVO: Modelos TCN production
        self.tcn_models = {}
        self.tcn_models_active = False
    
    def _load_config(self) -> BinanceConfig:
        """⚙️ Cargar configuración desde variables de entorno"""
        return BinanceConfig(
            api_key=os.getenv('BINANCE_API_KEY'),
            secret_key=os.getenv('BINANCE_SECRET_KEY'),
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
    
    async def get_account_info(self) -> AccountInfo:
        """💰 Obtener información completa de la cuenta de Binance"""
        try:
            timestamp = int(time.time() * 1000)
            params = f"timestamp={timestamp}&recvWindow=60000"
            signature = self._generate_signature(params)
            
            headers = {
                'X-MBX-APIKEY': self.config.api_key
            }
            
            url = f"{self.config.base_url}/api/v3/account"
            full_params = f"{params}&signature={signature}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}?{full_params}", headers=headers) as response:
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
            
            # 2. ✅ CRÍTICO: Inicializar cliente Binance para datos reales
            await self._initialize_binance_client()
            
            # 3. Obtener balance inicial de Binance - ✅ NUEVO
            print("💰 Obteniendo balance de Binance...")
            await self.update_balance_from_binance()
            if self.current_balance == 0:
                print("⚠️ No se pudo obtener balance de Binance, usando valor por defecto")
                self.current_balance = 102.0  # Fallback
            
            # 4. ✅ NUEVO: Inicializar Professional Portfolio Manager
            print("💼 Inicializando Professional Portfolio Manager...")
            self.portfolio_manager = ProfessionalPortfolioManager(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                base_url=self.config.base_url
            )
            print("✅ Portfolio Manager inicializado")
            
            # 5. Inicializar Risk Manager
            await self._initialize_risk_manager()
            
            # 6. ✅ NUEVO: Inicializar modelos TCN production
            await self._initialize_tcn_models()
            
            # 7. Verificar conectividad
            await self._verify_connectivity()
            
            # 8. Configurar monitoreo
            await self._setup_monitoring()
            
            self.start_time = time.time()
            self.last_heartbeat = datetime.now()
            self.status = TradingManagerStatus.RUNNING
            
            # Log inicial
            await self.database.log_event('INFO', 'SYSTEM', 'Simple Trading Manager inicializado correctamente')
            
            print("✅ Simple Professional Trading Manager iniciado correctamente")
            
        except Exception as e:
            self.status = TradingManagerStatus.ERROR
            print(f"❌ Error inicializando Trading Manager: {e}")
            if self.database:
                await self.database.log_event('ERROR', 'SYSTEM', f'Error inicializando: {e}')
            raise
    
    async def _initialize_database(self):
        """🗄️ Inicializar sistema de base de datos"""
        print("🗄️ Inicializando base de datos...")
        self.database = TradingDatabase()
        
        # Limpiar datos antiguos si es necesario
        await self.database.cleanup_old_data(days_to_keep=90)
        
        print("✅ Base de datos lista")
    
    async def _initialize_binance_client(self):
        """🔗 Inicializar cliente Binance para datos reales de mercado"""
        print("🔗 Inicializando cliente Binance...")
        
        try:
            from binance.client import Client
            
            # Verificar que tenemos credenciales
            if not self.config.api_key or not self.config.secret_key:
                print("⚠️ Sin credenciales API, usando cliente público")
                # Cliente público para datos de mercado (sin API keys)
                self.binance_client = Client()
            else:
                # Cliente autenticado
                is_testnet = 'testnet' in self.config.base_url.lower()
                print(f"   🔑 Usando credenciales {'(Testnet)' if is_testnet else '(Producción)'}")
                
                self.binance_client = Client(
                    api_key=self.config.api_key,
                    api_secret=self.config.secret_key,
                    testnet=is_testnet
                )
            
            # Verificar conectividad con un ping
            try:
                server_time = self.binance_client.get_server_time()
                print(f"   ✅ Conectado a Binance API")
                print(f"   🕐 Tiempo servidor: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            except Exception as ping_error:
                print(f"   ⚠️ Warning: {ping_error}")
            
            print("✅ Cliente Binance inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error inicializando cliente Binance: {e}")
            # Usar cliente dummy que falle gracefully
            self.binance_client = None
            raise
    
    async def _initialize_risk_manager(self):
        """🛡️ Inicializar Risk Manager"""
        self.logger.info("🛡️ Inicializando Advanced Risk Manager...")
        self.risk_manager = AdvancedRiskManager(self.config, self.database, self.logger)
        await self.risk_manager.initialize()
        self.logger.info("✅ Risk Manager configurado y listo.")
    
    async def _initialize_tcn_models(self):
        """🤖 Cargar los modelos TCN de producción"""
        self.logger.info("🤖 Cargando modelos TCN de producción...")
        
        # Cargar modelos para cada par de trading
        trading_pairs = self.risk_manager.get_trading_pairs()
        
        for pair in trading_pairs:
            model_path = f"models/tcn_final_{pair.lower()}.h5"
            
            if os.path.exists(model_path):
                try:
                    from tensorflow.keras.models import load_model
                    
                    # Si tus modelos usan capas personalizadas, agrégalas aquí.
                    # from tcn import TCN
                    # custom_objects = {'TCN': TCN}
                    custom_objects = {} # Por defecto, sin objetos personalizados.

                    model = load_model(model_path, custom_objects=custom_objects)
                    self.tcn_models[pair] = model
                    self.logger.info(f"✅ Modelo {model_path} cargado exitosamente para {pair}.")
                except Exception as e:
                    self.logger.critical(f" MCRITICAL: No se pudo cargar el modelo {model_path} para {pair}. Error: {e}")
                    self.tcn_models[pair] = None
            else:
                self.logger.warning(f"⚠️ No se encontró el modelo en la ruta: {model_path}. El bot no operará con ML para el par {pair}.")
                self.tcn_models[pair] = None

        if any(self.tcn_models.values()):
            self.tcn_models_active = True
            self.logger.info("✅ Modelos TCN activados. El bot operará con predicciones de ML.")
        else:
            self.tcn_models_active = False
            self.logger.error("🚨 No se cargó ningún modelo TCN. El bot no puede operar con ML.")
    
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
                """Loop de monitoreo de posiciones con trailing stops"""
                while self.status == TradingManagerStatus.RUNNING:
                    try:
                        await self._position_monitor()
                        await asyncio.sleep(30)  # Cada 30 segundos
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
        """🎯 Ejecutar loop principal de trading"""
        print("🎯 Iniciando loop principal de trading...")
        
        while self.status == TradingManagerStatus.RUNNING:
            try:
                loop_start_time = datetime.now()
                
                # Verificar si está pausado
                if self.pause_trading:
                    await self._handle_pause_state()
                    await asyncio.sleep(10)
                    continue
                
                # ✅ NUEVO: Generar reporte TCN cada 5 minutos
                await self._generate_tcn_report_if_needed()
                
                # ✅ MEJORADO: Mostrar información profesional en tiempo real
                await self._display_professional_info()
                
                # 1. Actualizar balance cada 5 minutos
                time_since_balance_update = None
                if self.last_balance_update:
                    time_since_balance_update = (datetime.now() - self.last_balance_update).total_seconds()
                
                if not self.last_balance_update or time_since_balance_update > 300:  # 5 minutos
                    print("🔄 Actualizando balance desde Binance...")
                    await self.update_balance_from_binance()
                
                # 2. Obtener precios actuales
                prices = await self._get_current_prices()
                self.current_prices = prices
                
                # ✅ NUEVO: Actualizar PnL de posiciones existentes
                await self._update_positions_pnl(prices)
                
                # 3. Generar señales simples (ejemplo)
                signals = await self._generate_simple_signals(prices)
                
                # 4. Procesar cada señal
                for symbol, signal_data in signals.items():
                    await self._process_signal(symbol, signal_data)
                
                # 5. Actualizar métricas
                await self._update_metrics()
                
                # 6. Guardar estado en DB
                await self._save_periodic_metrics()
                
                # ✅ NUEVO: Mostrar resumen cada ciclo
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                print(f"⏱️ Ciclo completado en {loop_duration:.1f}s")
                
                # 7. Esperar siguiente ciclo
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                await self._handle_error(e)
                await asyncio.sleep(30)  # Pausa en caso de error
    
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
                
                # Obtener snapshot del portafolio
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                self.last_portfolio_snapshot = snapshot
                self.metrics['portfolio_snapshots'] += 1
                
                # Generar reporte TCN
                tcn_report = self.portfolio_manager.format_tcn_style_report(snapshot)
                
                # Mostrar en consola
                print("\n" + "="*80)
                print("🎯 REPORTE TCN PROFESSIONAL")
                print("="*80)
                print(tcn_report)
                print("="*80)
                
                # Enviar a Discord si está configurado
                if hasattr(self, 'discord_notifier'):
                    await self._send_tcn_discord_notification(tcn_report)
                    self.metrics['tcn_reports_sent'] += 1
                
                self.last_tcn_report_time = now
                
        except Exception as e:
            print(f"❌ Error generando reporte TCN: {e}")
    
    async def _display_professional_info(self):
        """📺 Mostrar información profesional mejorada"""
        try:
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
                                print(f"      #{i}: {pos.size:.6f} @ ${pos.entry_price:,.2f} ({pos_pnl_sign}{pos.unrealized_pnl_percent:.1f}%) {duration_str}")
                
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
                await self.database.log_event('ERROR', 'MARKET_DATA', f'Error precio {symbol}: {e}', symbol)
        
        self.last_check_time = datetime.now()
        self.metrics['api_calls_count'] += len(self.symbols)
        
        return prices
    
    async def _generate_simple_signals(self, prices: Dict[str, float]) -> Dict:
        """🤖 Generar señales usando modelos TCN - SOLO BUY para Binance Spot"""
        signals = {}
        
        # Verificar si tenemos modelos TCN cargados
        if not self.tcn_models_active or not self.tcn_models:
            print("⚠️ Modelos TCN no disponibles, usando señales básicas")
            return await self._generate_fallback_signals(prices)
        
        for symbol, price in prices.items():
            try:
                # Verificar si tenemos USDT para comprar
                if self.current_balance < self.risk_manager.limits.min_position_value_usdt:
                    continue  # No hay suficiente USDT para comprar
                
                # Usar modelo TCN si está disponible
                if symbol in self.tcn_models and self.tcn_models[symbol] is not None:
                    tcn_signal = await self._get_tcn_prediction(symbol, price)
                    
                    if tcn_signal and tcn_signal['action'] == 'BUY':
                        signals[symbol] = {
                            'signal': 'BUY',  # ✅ Solo BUY permitido en Spot
                            'confidence': tcn_signal['confidence'],
                            'current_price': price,
                            'timestamp': datetime.now(),
                            'reason': 'tcn_model_prediction',
                            'available_usdt': self.current_balance,
                            'tcn_details': tcn_signal
                        }
                        
                        print(f"🤖 {symbol}: BUY TCN ({tcn_signal['confidence']:.1%}) @ ${price:.4f}")
                else:
                    # Fallback básico si no hay modelo para este símbolo
                    basic_signal = await self._generate_basic_signal(symbol, price)
                    if basic_signal:
                        signals[symbol] = basic_signal
                
            except Exception as e:
                print(f"❌ Error generando señal TCN {symbol}: {e}")
                # Intentar señal básica como fallback
                try:
                    basic_signal = await self._generate_basic_signal(symbol, price)
                    if basic_signal:
                        signals[symbol] = basic_signal
                except Exception as e2:
                    print(f"❌ Error en fallback {symbol}: {e2}")
        
        return signals
    
    async def _process_signal(self, symbol: str, signal_data: Dict):
        """⚡ Procesar una señal individual"""
        
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        
        # Skip si es HOLD
        if signal == 'HOLD':
            return
        
        # Verificar si ya tenemos posición en este símbolo
        if symbol in self.active_positions:
            await self._manage_existing_position(symbol, signal_data)
        else:
            await self._consider_new_position(symbol, signal_data)
    
    async def _consider_new_position(self, symbol: str, signal_data: Dict):
        """📈 Considerar nueva posición"""
        
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        
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
                'strategy': 'SIMPLE_SIGNALS',
                'is_active': True,
                'metadata': {
                    'signal_reason': signal_data.get('reason'),
                    'signal_time': signal_data['timestamp'].isoformat()
                }
            }
            
            trade_id = await self.database.save_trade(trade_data)
            position.trade_id = trade_id
            
            self.trade_count += 1
            
            # Log del trade
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
            else:
                await self._send_discord_notification(f"🟢 **NUEVA POSICIÓN**\n"
                                                     f"📊 {symbol}: {signal}\n"
                                                     f"💰 Precio: ${current_price:.4f}\n"
                                                     f"🎯 Confianza: {confidence:.1%}\n"
                                                     f"📈 Cantidad: {position.quantity:.6f}")
    
    async def _manage_existing_position(self, symbol: str, signal_data: Dict):
        """🔄 Gestionar posición existente"""
        
        position = self.active_positions[symbol]
        current_price = signal_data['current_price']
        
        # Actualizar precio actual
        position.current_price = current_price
        
        # Calcular PnL actual
        if position.side == 'BUY':
            position.pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            position.pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
        
        position.pnl_usd = (position.pnl_percent / 100) * (position.quantity * position.entry_price)
        
        # Si la señal cambió drásticamente, considerar cierre
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        
        if ((position.side == 'BUY' and signal == 'SELL') or 
            (position.side == 'SELL' and signal == 'BUY')) and confidence > 0.85:
            
            await self._close_position(symbol, "SIGNAL_REVERSAL")
    
    async def _close_position(self, symbol: str, reason: str):
        """📉 Cerrar posición específica"""
        
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        current_price = await self.get_current_price(symbol)
        
        # Calcular PnL final
        if position.side == 'BUY':
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
        
        pnl_usd = (pnl_percent / 100) * (position.quantity * position.entry_price)
        
        # Actualizar estadísticas
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
        
        # Remover de posiciones activas
        del self.active_positions[symbol]
        
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
        
        print(f"📉 Posición cerrada: {symbol} - PnL: {pnl_percent:.2f}% (${pnl_usd:.2f})")
    
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
        """🔍 Monitoreo continuo de posiciones y gestión de riesgo"""
        try:
            # 1. Obtener posiciones actuales
            snapshot = await self.portfolio_manager.get_portfolio_snapshot()
            
            if not snapshot.active_positions:
                print("   📊 Sin posiciones activas para monitorear")
                return
            
            print(f"🔍 Monitoreando {len(snapshot.active_positions)} posición(es)...")
            
            # 2. Actualizar precios para cada posición
            symbols_to_update = list(set([pos.symbol for pos in snapshot.active_positions]))
            current_prices = await self.portfolio_manager.update_all_prices(symbols_to_update)
            
            positions_to_close = []
            trailing_updates = []
            
            # 3. ✅ NUEVO: Procesar cada posición individualmente con trailing stop
            for position in snapshot.active_positions:
                try:
                    current_price = current_prices.get(position.symbol, position.current_price)
                    
                    # Actualizar precio actual en la posición
                    position.current_price = current_price
                    
                    # 🔄 Recalcular PnL con precio actual
                    if position.side == 'BUY':
                        entry_value = position.size * position.entry_price
                        current_value = position.size * current_price
                        position.unrealized_pnl_usd = current_value - entry_value
                        position.unrealized_pnl_percent = (position.unrealized_pnl_usd / entry_value) * 100 if entry_value > 0 else 0.0
                        position.market_value = current_value
                    
                    # ✅ NUEVO: Aplicar trailing stop profesional
                    updated_position, stop_triggered, trigger_reason = self.portfolio_manager.update_trailing_stop_professional(
                        position, current_price
                    )
                    
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
        """🚀 Cerrar múltiples posiciones en lote"""
        try:
            print(f"🔥 Iniciando cierre de {len(positions_and_reasons)} posición(es)...")
            
            for position, reason in positions_and_reasons:
                try:
                    # Simular cierre de posición (en producción, aquí iría la orden real de venta)
                    print(f"🛑 CERRANDO POSICIÓN {position.symbol} Pos #{position.order_id}:")
                    print(f"   📍 Entrada: ${position.entry_price:.4f}")
                    print(f"   💰 Actual: ${position.current_price:.4f}")
                    print(f"   📊 PnL: {position.unrealized_pnl_percent:.2f}% (${position.unrealized_pnl_usd:.2f})")
                    print(f"   🏷️ Razón: {reason}")
                    
                    # ✅ HABILITADO: Ejecutar orden real de venta
                    order_result = await self._execute_sell_order(position)
                    
                    if order_result and order_result.get('status') == 'FILLED':
                        print(f"   ✅ ORDEN EJECUTADA EXITOSAMENTE:")
                        print(f"   🆔 Order ID: {order_result.get('orderId')}")
                        print(f"   📊 Status: {order_result.get('status')}")
                        print(f"   💰 Ejecutado: {order_result.get('executedQty')} {position.symbol.replace('USDT', '')}")
                        print(f"   💵 Valor total: ${float(order_result.get('cummulativeQuoteQty', 0)):.2f}")
                        
                        # Remover posición de activas después de venta exitosa
                        if position.symbol in self.active_positions:
                            del self.active_positions[position.symbol]
                            print(f"   🗑️ Posición {position.symbol} removida de activas")
                    else:
                        print(f"   ❌ ERROR en ejecución de orden: {order_result}")
                        continue
                    
                    # Logging de la operación
                    await self.database.log_event(
                        'TRADE', 
                        'POSITION_CLOSED', 
                        f"{position.symbol}: {reason} - PnL: {position.unrealized_pnl_percent:.2f}% - OrderID: {order_result.get('orderId', 'N/A')}"
                    )
                    
                    # Actualizar métricas
                    self.metrics['total_trades'] += 1
                    if position.unrealized_pnl_usd > 0:
                        self.metrics['profitable_trades'] += 1
                    
                    print(f"✅ Posición {position.symbol} cerrada exitosamente")
                    
                except Exception as e:
                    print(f"❌ Error cerrando {position.symbol}: {e}")
                    continue
            
            print(f"🎯 Proceso de cierre completado")
            
        except Exception as e:
            print(f"❌ Error en cierre de posiciones en lote: {e}")
    
    async def _daily_loss_exceeds_limit(self, max_daily_loss_percent: float = 10.0) -> bool:
        """🚨 Verificar si se ha excedido la pérdida máxima diaria"""
        try:
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
        for symbol in list(self.active_positions.keys()):
            await self._close_position(symbol, "EMERGENCY_STOP")
        
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
            for symbol in list(self.active_positions.keys()):
                await self._close_position(symbol, "SYSTEM_SHUTDOWN")
        
        # Guardar métricas finales
        await self._save_periodic_metrics()
        
        # Log final
        await self.database.log_event('INFO', 'SYSTEM', 'Sistema apagado correctamente')
        
        print("✅ Sistema apagado correctamente")

    async def _get_tcn_prediction(self, symbol: str, current_price: float) -> Dict:
        """🤖 Obtener predicción del modelo TCN usando datos reales de mercado"""
        try:
            if symbol not in self.tcn_models or self.tcn_models[symbol] is None:
                return None
                
            model_info = self.tcn_models[symbol]
            model = model_info['model']
            
            # === USAR DATOS REALES DE MERCADO ===
            if not hasattr(self, 'market_data_provider'):
                # Inicializar proveedor de datos reales
                from real_market_data_provider import RealMarketDataProvider
                self.market_data_provider = RealMarketDataProvider(self.binance_client)
            
            # Obtener features reales de mercado
            real_features = await self.market_data_provider.get_real_market_features(symbol)
            
            if real_features is None:
                print(f"⚠️ No se pudieron obtener datos reales para {symbol}, usando fallback")
                return self._get_fallback_prediction(symbol, current_price)
            
            # Verificar que tenemos el shape correcto
            input_shape = model_info['input_shape']
            expected_shape = (input_shape[1], input_shape[2])  # (timesteps, features)
            
            if real_features.shape != expected_shape:
                print(f"❌ Shape de features incorrecto para {symbol}: {real_features.shape} != {expected_shape}")
                return self._get_fallback_prediction(symbol, current_price)
            
            # Normalizar features
            normalized_features = self.market_data_provider.normalize_features(real_features)
            
            # Preparar para predicción (agregar dimensión batch)
            model_input = np.expand_dims(normalized_features, axis=0)  # Shape: (1, timesteps, features)
            
            # Hacer predicción con el modelo TCN
            prediction = model.predict(model_input, verbose=0)
            raw_prediction = prediction[0]  # Remover dimensión batch
            
            # Interpretar predicción
            predicted_class = np.argmax(raw_prediction)
            confidence = float(raw_prediction[predicted_class])
            
            class_names = ['SELL', 'HOLD', 'BUY']
            action = class_names[predicted_class]
            
            # Crear respuesta
            result = {
                'action': action,
                'confidence': confidence,
                'price': current_price,
                'source': 'tcn_model_prediction',
                'model_used': symbol,
                'raw_prediction': [f"{x:.3f}" for x in raw_prediction],
                'using_real_data': True
            }
            
            print(f"🤖 {symbol}: {action} TCN ({confidence*100:.1f}%) @ ${current_price:.4f}")
            print(f"   ✅ USANDO DATOS REALES DE MERCADO")
            print(f"   🤖 TCN usado: {symbol}")
            print(f"   🎯 Raw prediction: {result['raw_prediction']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en predicción TCN {symbol}: {e}")
            return self._get_fallback_prediction(symbol, current_price)
    
    def _get_fallback_prediction(self, symbol: str, current_price: float) -> Dict:
        """🔄 Predicción fallback cuando TCN no está disponible"""
        try:
            # Usar datos dummy para fallback
            import numpy as np
            
            model_info = self.tcn_models.get(symbol)
            if model_info is None:
                return None
            
            model = model_info['model']
            input_shape = model_info['input_shape']
            
            # Crear datos dummy con el shape correcto
            timesteps = input_shape[1]
            features = input_shape[2]
            dummy_data = np.random.normal(0, 0.1, (1, timesteps, features)).astype(np.float32)
            
            # Predicción con datos dummy
            prediction = model.predict(dummy_data, verbose=0)
            raw_prediction = prediction[0]
            
            predicted_class = np.argmax(raw_prediction)
            confidence = float(raw_prediction[predicted_class])
            
            class_names = ['SELL', 'HOLD', 'BUY']
            action = class_names[predicted_class]
            
            result = {
                'action': action,
                'confidence': confidence,
                'price': current_price,
                'source': 'tcn_model_prediction',
                'model_used': f"{symbol}_fallback",
                'raw_prediction': [f"{x:.3f}" for x in raw_prediction],
                'using_real_data': False
            }
            
            print(f"🤖 {symbol}: {action} TCN ({confidence*100:.1f}%) @ ${current_price:.4f}")
            print(f"   ⚠️ USANDO DATOS DUMMY (fallback)")
            
            return result
            
        except Exception as e:
            print(f"❌ Error en fallback TCN {symbol}: {e}")
            return None
    
    async def _generate_fallback_signals(self, prices: Dict[str, float]) -> Dict:
        """🔄 Señales básicas cuando TCN no está disponible"""
        signals = {}
        
        for symbol, price in prices.items():
            try:
                basic_signal = await self._generate_basic_signal(symbol, price)
                if basic_signal:
                    signals[symbol] = basic_signal
            except Exception as e:
                print(f"❌ Error en señal fallback {symbol}: {e}")
        
        return signals
    
    async def _generate_basic_signal(self, symbol: str, price: float) -> Dict:
        """📊 Generar señal básica (fallback sin TCN)"""
        try:
            # Verificar si tenemos USDT suficiente
            if self.current_balance < self.risk_manager.limits.min_position_value_usdt:
                return None
            
            # Lógica básica simple (mejorable)
            
            # Probabilidad más conservadora sin TCN
            should_buy = random.random() < 0.3  # 30% chance
            confidence = random.uniform(0.6, 0.8)  # Menor confianza sin TCN
            
            if should_buy and confidence > 0.7:
                return {
                    'signal': 'BUY',
                    'confidence': confidence,
                    'current_price': price,
                    'timestamp': datetime.now(),
                    'reason': 'basic_fallback_signal',
                    'available_usdt': self.current_balance
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error señal básica {symbol}: {e}")
            return None

    async def _execute_sell_order(self, position):
        """🚀 Ejecutar orden de venta real en Binance"""
        try:
            # Validaciones previas
            if not position or not hasattr(position, 'symbol'):
                print(f"❌ Posición inválida para venta")
                return None
            
            # Obtener información actual del activo
            asset = position.symbol.replace('USDT', '')
            balances = await self.portfolio_manager.get_account_balances()
            
            if asset not in balances:
                print(f"❌ No hay balance para {asset}")
                return None
            
            # Cantidad disponible para vender
            available_quantity = balances[asset]['free']
            
            # Usar la cantidad menor entre la posición y el balance disponible
            sell_quantity = min(position.size, available_quantity)
            
            if sell_quantity <= 0:
                print(f"❌ Cantidad inválida para vender: {sell_quantity}")
                return None
            
            # Obtener precio actual de mercado
            current_price = await self.get_current_price(position.symbol)
            if not current_price or current_price <= 0:
                print(f"❌ No se pudo obtener precio actual para {position.symbol}")
                return None
            
            # Formatear cantidad según las especificaciones del símbolo
            # Binance requiere cantidades específicas según el asset
            sell_quantity = self._format_quantity_for_symbol(position.symbol, sell_quantity)
            
            print(f"🚀 EJECUTANDO ORDEN DE VENTA:")
            print(f"   📈 Símbolo: {position.symbol}")
            print(f"   📊 Cantidad: {sell_quantity}")
            print(f"   💰 Precio estimado: ${current_price:.4f}")
            print(f"   📍 Razón: Trailing Stop / Stop Loss")
            
            # Preparar parámetros de la orden
            order_params = {
                'symbol': position.symbol,
                'side': 'SELL',
                'type': 'MARKET',  # Orden de mercado para ejecución rápida
                'quantity': sell_quantity,
                'recvWindow': 5000,  # Ventana de recepción de 5 segundos
                'timestamp': int(time.time() * 1000)
            }
            
            # Agregar signature para autenticación
            query_string = "&".join([f"{k}={v}" for k, v in order_params.items()])
            signature = self._generate_signature(query_string)
            order_params['signature'] = signature
            
            # Ejecutar orden real en Binance
            order_result = await self._make_authenticated_binance_request('order', order_params, method='POST')
            
            if order_result:
                print(f"✅ ORDEN EJECUTADA EXITOSAMENTE:")
                print(f"   🆔 Order ID: {order_result.get('orderId')}")
                print(f"   📊 Status: {order_result.get('status')}")
                print(f"   💰 Ejecutado: {order_result.get('executedQty')} {asset}")
                print(f"   💵 Valor total: ${float(order_result.get('cummulativeQuoteQty', 0)):.2f}")
                
                # Enviar notificación Discord si está configurado
                await self._send_discord_notification(
                    f"🛑 TRAILING STOP EJECUTADO\n"
                    f"📈 {position.symbol}: Vendido {sell_quantity} {asset}\n"
                    f"💰 PnL: {position.unrealized_pnl_percent:.2f}%\n"
                    f"🆔 Order: {order_result.get('orderId')}"
                )
                
                return order_result
            else:
                print(f"❌ Error: No se recibió respuesta de la orden")
                return None
                
        except Exception as e:
            print(f"❌ ERROR ejecutando orden de venta para {position.symbol}: {e}")
            
            # Log detallado del error
            await self.database.log_event(
                'ERROR', 
                'ORDER_EXECUTION', 
                f"Error vendiendo {position.symbol}: {str(e)}"
            )
            
            return None
    
    def _format_quantity_for_symbol(self, symbol: str, quantity: float) -> float:
        """📏 Formatear cantidad según especificaciones del símbolo"""
        try:
            # Configuraciones típicas para diferentes activos
            quantity_precision = {
                'BTCUSDT': 5,   # 5 decimales para BTC
                'ETHUSDT': 4,   # 4 decimales para ETH  
                'BNBUSDT': 2,   # 2 decimales para BNB
                'ADAUSDT': 0,   # Sin decimales para ADA
                'default': 3    # 3 decimales por defecto
            }
            
            precision = quantity_precision.get(symbol, quantity_precision['default'])
            formatted_quantity = round(quantity, precision)
            
            # Validar cantidad mínima (típicamente $10-20 en Binance)
            if formatted_quantity <= 0.0001:
                return 0.0
                
            return formatted_quantity
            
        except Exception as e:
            print(f"❌ Error formateando cantidad para {symbol}: {e}")
            return quantity
    
    async def _make_authenticated_binance_request(self, endpoint: str, params: dict, method: str = 'GET'):
        """🔐 Hacer petición autenticada a Binance API"""
        try:
            import aiohttp
            import time
            
            # Headers de la petición
            headers = {
                'X-MBX-APIKEY': self.binance_config.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # URL completa
            url = f"{self.binance_config.base_url}/api/v3/{endpoint}"
            
            # Hacer petición HTTP
            async with aiohttp.ClientSession() as session:
                if method == 'GET':
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            error_text = await response.text()
                            print(f"❌ Error API {response.status}: {error_text}")
                            return None
                            
                elif method == 'POST':
                    async with session.post(url, data=params, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            error_text = await response.text()
                            print(f"❌ Error API {response.status}: {error_text}")
                            return None
            
        except Exception as e:
            print(f"❌ Error en petición Binance: {e}")
            return None

async def main():
    """🎯 Función principal para testing directo"""
    print("🧪 Modo de prueba - Simple Professional Trading Manager")
    
    manager = TradingManager()
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