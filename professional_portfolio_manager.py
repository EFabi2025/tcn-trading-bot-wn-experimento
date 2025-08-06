#!/usr/bin/env python3
"""
💼 PROFESSIONAL PORTFOLIO MANAGER
Sistema avanzado para gestión de portafolio con datos reales de Binance
Replica y mejora el formato del bot TCN anterior
VERSIÓN CORREGIDA: Múltiples posiciones por par con precios de entrada reales
"""

import asyncio
import aiohttp
import time
import hmac
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from config.trading_config import get_risk_params

load_dotenv()

@dataclass
class Position:
    """📈 Posición individual en el portafolio"""
    symbol: str
    side: str  # BUY o SELL
    quantity: float  # Cantidad del activo (unificado con AdvancedRiskManager)
    entry_price: float
    current_price: float
    market_value: float  # Valor actual en USDT
    unrealized_pnl_usd: float
    unrealized_pnl_percent: float
    entry_time: datetime
    duration_minutes: int
    order_id: Optional[str] = None  # ID de la orden original
    batch_id: Optional[str] = None  # Para agrupar órdenes relacionadas

    # ✅ NUEVO: Sistema de Trailing Stop Profesional
    trailing_stop_active: bool = False
    trailing_stop_price: Optional[float] = None
    trailing_stop_percent: float = 1.4  # Default 1.4% - Configurado desde .env
    highest_price_since_entry: Optional[float] = None  # Para tracking del máximo
    lowest_price_since_entry: Optional[float] = None   # Para shorts
    trailing_activation_threshold: float = 0.4 # Activar trailing después de +0.4% ganancia
    last_trailing_update: Optional[datetime] = None
    trailing_movements: int = 0  # Contador de movimientos del trailing

    # Stop Loss y Take Profit tradicionales
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_percent: float = 1.4  # Default 1.4% - Configurado desde .env
    take_profit_percent: float = 4.0  # Default 4.0% - Configurado desde .env

@dataclass
class Asset:
    """🪙 Activo individual en el portafolio"""
    symbol: str
    free: float
    locked: float
    total: float
    usd_value: float
    percentage_of_portfolio: float

@dataclass
class TradeOrder:
    """📋 Orden de trading individual"""
    order_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    executed_qty: float
    cumulative_quote_qty: float
    time: datetime
    status: str

@dataclass
class PortfolioSnapshot:
    """📊 Snapshot completo del portafolio"""
    timestamp: datetime
    total_balance_usd: float
    free_usdt: float
    total_unrealized_pnl: float
    total_unrealized_pnl_percent: float
    active_positions: List[Position]
    all_assets: List[Asset]
    position_count: int
    max_positions: int
    total_trades_today: int

class ProfessionalPortfolioManager:
    """🏛️ Gestor profesional de portafolio con TCN y trailing stops avanzados"""

    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://testnet.binance.vision"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

        # ✅ NUEVO: Registry persistente de posiciones
        self.position_registry: Dict[str, Position] = {}  # order_id -> Position
        self.last_orders_hash: Optional[str] = None  # Para detectar cambios en órdenes

        # Configuración
        self.max_positions = int(os.getenv('MAX_CONCURRENT_POSITIONS', '3'))  # ✅ Lee del .env
        self.min_position_value = 5.0  # Mínimo $5 USD por posición
        self.days_to_lookback = 30  # Días hacia atrás para historial

        # Cache de precios
        self.price_cache = {}
        self.last_price_update = {}

        # Cache de trailing stops - ✅ MEJORADO: Más robusto
        self.trailing_cache_file = "trailing_stops_cache.json"
        self.trailing_cache = self._load_trailing_cache()

        # Timestamps
        self.last_snapshot_time = None

        print(f"✅ ProfessionalPortfolioManager inicializado")
        print(f"   📊 Max posiciones: {self.max_positions}")
        print(f"   💰 Valor mínimo por posición: ${self.min_position_value}")
        print(f"   📅 Días de historial: {self.days_to_lookback}")
        print(f"   🗂️ Registry de posiciones: Inicializado")

        # Configuración de timeouts y límites
        self.request_timeout = 10
        self.max_retries = 3
        self.rate_limit_delay = 0.1

        # Métricas
        self.api_calls_count = 0
        self.last_api_call = None
        self.error_count = 0

        # ✅ NUEVO: Cache de órdenes para tracking de posiciones
        self.orders_cache = {}
        self.last_orders_update = None

        self.trailing_state_cache_file = Path("trailing_states.json")
        self.trailing_state_cache = self._load_trailing_cache()

    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma HMAC SHA256 para Binance"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _make_authenticated_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """🔗 Realizar petición autenticada a Binance"""
        if params is None:
            params = {}

        # Añadir timestamp y recvWindow
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 10000  # Aumentar la ventana a 10 segundos

        # Crear query string
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])

        # Generar firma
        signature = self._generate_signature(query_string)
        query_string += f"&signature={signature}"

        # Headers
        headers = {
            'X-MBX-APIKEY': self.api_key
        }

        # Realizar petición
        url = f"{self.base_url}/api/v3/{endpoint}?{query_string}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                self.api_calls_count += 1

                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Error API Binance: {response.status} - {error_text}")

    async def get_current_price(self, symbol: str) -> float:
        """💲 Obtener precio actual de un símbolo"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol}
                async with session.get(url, params=params) as response:
                    self.api_calls_count += 1
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        self.price_cache[symbol] = price
                        return price
        except Exception as e:
            print(f"❌ Error obteniendo precio {symbol}: {e}")
        return 0.0

    async def update_all_prices(self, symbols: List[str]) -> Dict[str, float]:
        """💲 Actualizar precios de múltiples símbolos con validación de frescura"""
        try:
            current_time = datetime.now()
            price_dict = {}
            symbols_to_fetch = []

            # Verificar precios cacheados primero
            for symbol in symbols:
                last_update = self.last_price_update.get(symbol)
                cached_price = self.price_cache.get(symbol)

                if (last_update and cached_price and
                    (current_time - last_update).total_seconds() < 30):
                    # Usar precio cacheado si es reciente (< 30 segundos)
                    price_dict[symbol] = cached_price
                else:
                    symbols_to_fetch.append(symbol)

            # Obtener precios frescos para símbolos que lo necesiten
            if symbols_to_fetch:
                tasks = [self.get_current_price(symbol) for symbol in symbols_to_fetch]
                fresh_prices = await asyncio.gather(*tasks)

                for symbol, price in zip(symbols_to_fetch, fresh_prices):
                    if price > 0:
                        price_dict[symbol] = price
                        self.price_cache[symbol] = price
                        self.last_price_update[symbol] = current_time
                    elif symbol in self.price_cache:
                        # Fallback al precio cacheado si falla la API
                        price_dict[symbol] = self.price_cache[symbol]
                        print(f"⚠️ {symbol}: Usando precio cacheado por error API")

            return price_dict

        except Exception as e:
            print(f"❌ Error actualizando precios: {e}")
            # Fallback a precios cacheados disponibles
            return {symbol: self.price_cache.get(symbol, 0) for symbol in symbols
                   if self.price_cache.get(symbol, 0) > 0}

    async def get_account_balances(self) -> Dict[str, Dict]:
        """💰 Obtener balances de la cuenta"""
        try:
            data = await self._make_authenticated_request("account")

            balances = {}
            for balance in data.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked

                if total > 0:  # Solo activos con balance > 0
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }

            return balances

        except Exception as e:
            print(f"❌ Error obteniendo balances: {e}")
            return {}

    async def get_order_history(self, symbol: Optional[str] = None, days_back: Optional[int] = None) -> List[TradeOrder]:
        """📋 Obtener historial de órdenes ejecutadas"""
        try:
            if days_back is None:
                days_back = self.days_to_lookback

            # Calcular timestamp de inicio (días hacia atrás)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

            orders = []

            if symbol:
                # Obtener órdenes para un símbolo específico
                params = {
                    'symbol': symbol,
                    'startTime': start_time,
                    'limit': 1000  # Máximo por request
                }

                data = await self._make_authenticated_request("allOrders", params)

                for order in data:
                    if order['status'] == 'FILLED':  # Solo órdenes ejecutadas
                        trade_order = TradeOrder(
                            order_id=str(order['orderId']),
                            symbol=order['symbol'],
                            side=order['side'],
                            quantity=float(order['origQty']),
                            price=float(order['price']) if order['price'] != '0.00000000' else float(order['cummulativeQuoteQty']) / float(order['executedQty']),
                            executed_qty=float(order['executedQty']),
                            cumulative_quote_qty=float(order['cummulativeQuoteQty']),
                            time=datetime.fromtimestamp(order['time'] / 1000),
                            status=order['status']
                        )
                        orders.append(trade_order)
            else:
                # Obtener órdenes para todos los símbolos activos
                balances = await self.get_account_balances()

                for asset in balances.keys():
                    if asset != 'USDT':
                        try:
                            symbol_orders = await self.get_order_history(f"{asset}USDT", days_back)
                            orders.extend(symbol_orders)
                        except Exception as e:
                            print(f"⚠️ Error obteniendo órdenes para {asset}USDT: {e}")
                            continue

            return sorted(orders, key=lambda x: x.time, reverse=True)

        except Exception as e:
            print(f"❌ Error obteniendo historial de órdenes: {e}")
            return []

    def group_orders_into_positions(self, orders: List[TradeOrder], current_balances: Dict[str, Dict]) -> List[Position]:
        """🔄 Agrupar órdenes en posiciones individuales usando FIFO"""
        try:
            positions = []

            # Agrupar órdenes por símbolo
            orders_by_symbol = {}
            for order in orders:
                if order.symbol not in orders_by_symbol:
                    orders_by_symbol[order.symbol] = []
                orders_by_symbol[order.symbol].append(order)

            # Procesar cada símbolo
            for symbol, symbol_orders in orders_by_symbol.items():
                # Ordenar órdenes por tiempo (más antiguas primero)
                symbol_orders.sort(key=lambda x: x.time)

                # Obtener balance actual del activo
                asset = symbol.replace('USDT', '')
                current_balance = current_balances.get(asset, {}).get('total', 0.0)

                if current_balance <= 0:
                    continue  # No hay balance actual, skip

                # Algoritmo FIFO para determinar posiciones actuales
                remaining_balance = current_balance
                buy_orders = [order for order in symbol_orders if order.side == 'BUY']
                sell_orders = [order for order in symbol_orders if order.side == 'SELL']

                # Primero, restar todas las ventas del balance inicial acumulado
                total_bought = sum(order.executed_qty for order in buy_orders)
                total_sold = sum(order.executed_qty for order in sell_orders)

                # Si el balance actual es menor que el total comprado menos vendido,
                # significa que algunas posiciones fueron cerradas

                # Crear posiciones basadas en órdenes de compra que aún están "abiertas"
                current_position_qty = remaining_balance

                # Procesar órdenes de compra desde la más reciente (LIFO para mostrar mejor info)
                for buy_order in reversed(buy_orders):
                    if current_position_qty <= 0:
                        break

                    # Determinar cuánta cantidad de esta orden aún está en posición
                    qty_from_this_order = min(buy_order.executed_qty, current_position_qty)

                    if qty_from_this_order > 0:
                        # Crear posición para esta parte
                        current_price = self.price_cache.get(symbol, buy_order.price)
                        market_value = qty_from_this_order * current_price

                        # Calcular PnL
                        entry_value = qty_from_this_order * buy_order.price
                        pnl_usd = market_value - entry_value
                        pnl_percent = (pnl_usd / entry_value) * 100 if entry_value > 0 else 0

                        # Calcular duración
                        duration_minutes = int((datetime.now() - buy_order.time).total_seconds() / 60)

                        new_position = Position(
                            symbol=symbol,
                            side='BUY',
                            quantity=qty_from_this_order,
                            entry_price=buy_order.price,
                            current_price=current_price,
                            market_value=market_value,
                            unrealized_pnl_usd=pnl_usd,
                            unrealized_pnl_percent=pnl_percent,
                            entry_time=buy_order.time,
                            duration_minutes=duration_minutes,
                            order_id=f"pos_{buy_order.order_id}",  # ✅ CORREGIDO: ID estable basado en order_id original
                            batch_id=buy_order.order_id
                        )

                        # ✅ NUEVO: Inicializar stops para nueva posición
                        new_position = self.initialize_position_stops(new_position)
                        positions.append(new_position)
                        current_position_qty -= qty_from_this_order

            return positions

        except Exception as e:
            print(f"❌ Error agrupando órdenes en posiciones: {e}")
            return []

    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """📊 Obtener snapshot completo del portafolio - ✅ MEJORADO: Con persistencia de posiciones"""
        try:
            print("📊 Obteniendo snapshot del portafolio...")

            # 1. Obtener balances
            balances = await self.get_account_balances()
            if not balances:
                raise Exception("No se pudieron obtener balances")

            # 2. Identificar símbolos para obtener precios
            symbols_needed = []
            for asset in balances.keys():
                if asset != 'USDT':
                    symbols_needed.append(f"{asset}USDT")

            # 3. Obtener precios actuales
            if symbols_needed:
                prices = await self.update_all_prices(symbols_needed)
            else:
                prices = {}

            # 4. ✅ NUEVO: Obtener historial de órdenes
            print("📋 Obteniendo historial de órdenes...")
            all_orders = await self.get_order_history(days_back=self.days_to_lookback)
            print(f"   📄 Encontradas {len(all_orders)} órdenes ejecutadas")

            # 5. ✅ MEJORADO: Sincronizar registry con órdenes (solo si hay cambios)
            orders_hash = self._calculate_orders_hash(all_orders)
            if orders_hash != self.last_orders_hash:
                print("🔄 Detectados cambios en órdenes, sincronizando registry...")
                self.sync_positions_with_orders(all_orders, balances)
                self.last_orders_hash = orders_hash
            else:
                print("✅ Sin cambios en órdenes, usando registry existente")

            # 6. ✅ NUEVO: Actualizar precios y PnL de posiciones existentes
            await self.update_existing_positions_prices(prices)

            # 7. Calcular valor de cada activo
            all_assets = []
            total_portfolio_value = 0.0
            free_usdt = balances.get('USDT', {}).get('free', 0.0)

            for asset, balance_info in balances.items():
                if balance_info['total'] > 0:
                    if asset == 'USDT':
                        usd_value = balance_info['total']
                    else:
                        symbol = f"{asset}USDT"
                        price = prices.get(symbol, 0.0)
                        usd_value = balance_info['total'] * price if price > 0 else 0.0

                    total_portfolio_value += usd_value

                    asset_obj = Asset(
                        symbol=asset,
                        free=balance_info['free'],
                        locked=balance_info['locked'],
                        total=balance_info['total'],
                        usd_value=usd_value,
                        percentage_of_portfolio=0.0  # Se calculará después
                    )
                    all_assets.append(asset_obj)

            # 8. Calcular porcentajes
            for asset in all_assets:
                asset.percentage_of_portfolio = (asset.usd_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0

            # 9. ✅ MEJORADO: Usar posiciones del registry (con trailing stops preservados)
            active_positions = [pos for pos in self.position_registry.values()
                             if pos.market_value >= self.min_position_value]

            # 10. Calcular PnL total
            total_unrealized_pnl = sum(pos.unrealized_pnl_usd for pos in active_positions)

            # 11. Crear snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_balance_usd=total_portfolio_value,
                free_usdt=free_usdt,
                total_unrealized_pnl=total_unrealized_pnl,
                total_unrealized_pnl_percent=(total_unrealized_pnl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0,
                active_positions=active_positions,
                all_assets=all_assets,
                position_count=len(active_positions),
                max_positions=self.max_positions,
                total_trades_today=len([o for o in all_orders if o.time.date() == datetime.now().date()])
            )

            self.last_snapshot_time = datetime.now()
            print(f"✅ Snapshot obtenido: {len(all_assets)} activos, {len(active_positions)} posiciones del registry")

            return snapshot

        except Exception as e:
            print(f"❌ Error obteniendo snapshot: {e}")
            raise

    def format_tcn_style_report(self, snapshot: PortfolioSnapshot) -> str:
        """🎨 Formatear reporte estilo TCN para Discord"""
        try:
            now = snapshot.timestamp

            # Header
            report = f"**🚀 TCN SIGNALS - {now.strftime('%H:%M:%S')}**\n"
            report += f"📊 **Recomendaciones del Modelo Profesional**\n\n"

            # Posiciones activas - ✅ MEJORADO: Mostrar posiciones individuales
            if snapshot.active_positions:
                report += f"**📈 POSICIONES ACTIVAS ({len(snapshot.active_positions)})**\n"

                # Agrupar por símbolo para mejor visualización
                positions_by_symbol = {}
                for pos in snapshot.active_positions:
                    if pos.symbol not in positions_by_symbol:
                        positions_by_symbol[pos.symbol] = []
                    positions_by_symbol[pos.symbol].append(pos)

                for symbol, positions in positions_by_symbol.items():
                    if len(positions) == 1:
                        # Una sola posición para este símbolo
                        pos = positions[0]
                        pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                        pnl_color = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"

                        report += f"**{pos.symbol}: {pos.side}**\n"
                        report += f"└ ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} "
                        report += f"({pnl_sign}{pos.unrealized_pnl_percent:.2f}% = ${pnl_sign}{pos.unrealized_pnl_usd:.2f}) {pnl_color}\n"
                        report += f"   💰 Cantidad: {pos.quantity:.6f} | 🕐 {pos.duration_minutes}min"

                        # ✅ NUEVO: Mostrar estado del trailing stop
                        if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                            report += f" | 📈 Trail: ${pos.trailing_stop_price:.2f}"

                        report += "\n\n"
                    else:
                        # Múltiples posiciones para este símbolo
                        report += f"**{symbol}: MÚLTIPLES POSICIONES ({len(positions)})**\n"

                        total_pnl = sum(p.unrealized_pnl_usd for p in positions)
                        total_pnl_sign = "+" if total_pnl >= 0 else ""
                        total_pnl_color = "🟢" if total_pnl >= 0 else "🔴"

                        for i, pos in enumerate(positions, 1):
                            pnl_sign = "+" if pos.unrealized_pnl_usd >= 0 else ""
                            pnl_color = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"

                            report += f"├─ **Pos #{i}:** ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} "
                            report += f"({pnl_sign}{pos.unrealized_pnl_percent:.2f}% = ${pnl_sign}{pos.unrealized_pnl_usd:.2f}) {pnl_color}\n"
                            report += f"│  💰 {pos.quantity:.6f} | 🕐 {pos.duration_minutes}min"

                            # ✅ NUEVO: Estado trailing stop por posición
                            if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active:
                                report += f" | 📈 Trail: ${pos.trailing_stop_price:.2f}"

                            report += "\n"

                        report += f"└─ **TOTAL:** ${total_pnl_sign}{total_pnl:.2f} {total_pnl_color}\n\n"
            else:
                report += "**📈 POSICIONES ACTIVAS (0)**\n"
                report += "└ Sin posiciones activas\n\n"

            # Resumen rápido
            report += "**⚡ RESUMEN RÁPIDO**\n"
            report += f"💰 **USDT Libre:** ${snapshot.free_usdt:,.2f}\n"

            pnl_sign = "+" if snapshot.total_unrealized_pnl >= 0 else ""
            pnl_emoji = "📈" if snapshot.total_unrealized_pnl >= 0 else "📉"
            report += f"{pnl_emoji} **P&L No Realizado:** ${pnl_sign}{snapshot.total_unrealized_pnl:.2f}\n"

            report += f"🎯 **Posiciones:** {snapshot.position_count}/{snapshot.max_positions}\n"
            report += f"📊 **Trades Totales:** {snapshot.total_trades_today}\n\n"

            # Detalle del portafolio
            report += "**💼 DETALLE DEL PORTAFOLIO**\n"

            # Ordenar activos por valor USD (mayor a menor)
            sorted_assets = sorted(snapshot.all_assets, key=lambda x: x.usd_value, reverse=True)

            for asset in sorted_assets:
                if asset.usd_value >= 0.01:  # Solo mostrar activos con valor > $0.01
                    if asset.symbol == 'USDT':
                        report += f"💵 **{asset.symbol}:** ${asset.total:,.2f}\n"
                    else:
                        report += f"🪙 **{asset.symbol}:** {asset.total:.6f} (${asset.usd_value:,.2f})\n"

            report += f"\n💎 **VALOR TOTAL: ${snapshot.total_balance_usd:,.2f}**\n"

            # Footer
            report += f"\n🔄 *Actualización cada 3 min • {now.strftime('%d/%m/%y, %H:%M')}*"

            return report

        except Exception as e:
            print(f"❌ Error formateando reporte: {e}")
            return f"❌ Error generando reporte: {e}"

    def format_compact_report(self, snapshot: PortfolioSnapshot) -> str:
        """📱 Formatear reporte compacto para notificaciones"""
        try:
            total_value = snapshot.total_balance_usd
            pnl = snapshot.total_unrealized_pnl
            positions = len(snapshot.active_positions)

            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            return (f"💼 Portfolio: ${total_value:,.2f} | "
                   f"{pnl_emoji} PnL: ${pnl_sign}{pnl:.2f} | "
                   f"🎯 Pos: {positions}/{snapshot.max_positions}")

        except Exception as e:
            return f"❌ Error: {e}"

    # ✅ NUEVO: Sistema de Trailing Stop Profesional

    def _save_trailing_state(self, position: Position):
        """💾 Guardar estado del trailing stop en cache"""
        try:
            if position.order_id:
                self.trailing_cache[position.order_id] = {
                    'trailing_stop_active': position.trailing_stop_active,
                    'trailing_stop_price': position.trailing_stop_price,
                    'highest_price_since_entry': position.highest_price_since_entry,
                    'lowest_price_since_entry': position.lowest_price_since_entry,
                    'trailing_movements': position.trailing_movements,
                    'last_trailing_update': position.last_trailing_update.isoformat() if position.last_trailing_update else None,
                    'symbol': position.symbol,  # Para debugging
                    'entry_price': position.entry_price  # Para validación
                }

                # ✅ PERSISTENCIA: Guardar inmediatamente en archivo
                self._save_trailing_cache()

                # ✅ NUEVO: Logging detallado para debugging
                if position.trailing_stop_active:
                    protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100 if position.trailing_stop_price else 0
                    print(f"💾 TRAILING GUARDADO {position.symbol} Pos #{position.order_id}:")
                    print(f"   📈 Estado: ACTIVO ${position.trailing_stop_price:.4f} (+{protection:.2f}%)")
                    print(f"   🏔️ Máximo: ${position.highest_price_since_entry:.4f}")
                    print(f"   📊 Movimientos: {position.trailing_movements}")
                else:
                    print(f"💾 TRAILING GUARDADO {position.symbol} Pos #{position.order_id}: INACTIVO")
            else:
                print(f"⚠️ No se puede guardar trailing para {position.symbol}: Sin order_id")

        except Exception as e:
            print(f"❌ Error guardando estado trailing para {position.symbol}: {e}")

    def _restore_trailing_state(self, position: Position) -> Position:
        """🔄 Restaurar estado del trailing stop desde cache"""
        try:
            if position.order_id and position.order_id in self.trailing_cache:
                cached_state = self.trailing_cache[position.order_id]

                # Restaurar estado
                position.trailing_stop_active = cached_state.get('trailing_stop_active', False)
                position.trailing_stop_price = cached_state.get('trailing_stop_price', None)
                position.highest_price_since_entry = cached_state.get('highest_price_since_entry', position.entry_price)
                position.lowest_price_since_entry = cached_state.get('lowest_price_since_entry', position.entry_price)
                position.trailing_movements = cached_state.get('trailing_movements', 0)
                # Restaurar timestamp
                last_update_str = cached_state.get('last_trailing_update')
                if last_update_str:
                    try:
                        from datetime import datetime
                        position.last_trailing_update = datetime.fromisoformat(last_update_str)
                    except:
                        position.last_trailing_update = datetime.now()
                else:
                    position.last_trailing_update = None

                # ✅ NUEVO: Logging detallado para debugging
                if position.trailing_stop_active:
                    protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100 if position.trailing_stop_price else 0
                    print(f"🔄 TRAILING RESTAURADO {position.symbol} Pos #{position.order_id}:")
                    print(f"   📈 Estado: ACTIVO ${position.trailing_stop_price:.4f} (+{protection:.2f}%)")
                    print(f"   🏔️ Máximo histórico: ${position.highest_price_since_entry:.4f}")
                    print(f"   📊 Movimientos: {position.trailing_movements}")
                else:
                    print(f"🔄 TRAILING RESTAURADO {position.symbol} Pos #{position.order_id}: INACTIVO")

                return position
            else:
                # ✅ NUEVO: Logging cuando no hay estado previo
                if position.order_id:
                    print(f"🆕 NUEVA POSICIÓN {position.symbol} Pos #{position.order_id}: Sin estado trailing previo")
                else:
                    print(f"⚠️ POSICIÓN SIN ID {position.symbol}: No se puede restaurar trailing")

        except Exception as e:
            print(f"❌ Error restaurando estado trailing para {position.symbol}: {e}")

        return position

    def initialize_position_stops(self, position: Position) -> Position:
        """🛡️ Inicializar Stop Loss, Take Profit y Trailing Stop para una posición"""
        try:
            # ✅ CONFIGURACIÓN CENTRALIZADA DESDE .ENV
            import os
            stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', '1.4'))
            take_profit_percent = float(os.getenv('TAKE_PROFIT_PERCENT', '4.0'))
            trailing_stop_percent = float(os.getenv('TRAILING_STOP_PERCENT', '1.4'))
            trailing_activation_threshold = float(os.getenv('TRAILING_ACTIVATION_THRESHOLD', '0.4'))

            # Actualizar los valores de la posición con la configuración
            position.stop_loss_percent = stop_loss_percent
            position.take_profit_percent = take_profit_percent
            position.trailing_stop_percent = trailing_stop_percent
            position.trailing_activation_threshold = trailing_activation_threshold

            # ✅ PRIMERO: Intentar restaurar estado previo del trailing stop
            position = self._restore_trailing_state(position)

            # Solo inicializar si no hay estado previo
            if not hasattr(position, 'trailing_stop_active') or position.trailing_stop_active is None:
                # Configurar Stop Loss y Take Profit tradicionales
                if position.side == 'BUY':
                    position.stop_loss_price = position.entry_price * (1 - position.stop_loss_percent / 100)
                    position.take_profit_price = position.entry_price * (1 + position.take_profit_percent / 100)
                    position.highest_price_since_entry = position.entry_price
                    position.lowest_price_since_entry = None
                else:  # SELL (para futuros)
                    position.stop_loss_price = position.entry_price * (1 + position.stop_loss_percent / 100)
                    position.take_profit_price = position.entry_price * (1 - position.take_profit_percent / 100)
                    position.lowest_price_since_entry = position.entry_price
                    position.highest_price_since_entry = None

                # Trailing stop inicialmente inactivo (solo si es nueva posición)
                position.trailing_stop_active = False
                position.trailing_stop_price = None
                position.last_trailing_update = datetime.now()
                position.trailing_movements = 0

                print(f"🛡️ Stops inicializados para {position.symbol} Pos #{position.order_id}:")
                print(f"   📍 Entrada: ${position.entry_price:.4f}")
                print(f"   🛑 Stop Loss: ${position.stop_loss_price:.4f} (-{position.stop_loss_percent}%)")
                print(f"   🎯 Take Profit: ${position.take_profit_price:.4f} (+{position.take_profit_percent}%)")
                print(f"   📈 Trailing: INACTIVO (activar en +{position.trailing_activation_threshold}%)")
                print(f"   💰 Protección mínima: +0.9% (cubre comisiones Binance)")
            else:
                # Posición con estado previo restaurado
                if position.trailing_stop_active:
                    protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100 if position.trailing_stop_price else 0
                    print(f"🔄 Estado trailing restaurado para {position.symbol} Pos #{position.order_id}:")
                    print(f"   📈 Trailing: ACTIVO ${position.trailing_stop_price:.4f} (+{protection:.2f}%)")
                    print(f"   🏔️ Máximo histórico: ${position.highest_price_since_entry:.4f}")
                    print(f"   📊 Movimientos: {position.trailing_movements}")

            return position

        except Exception as e:
            print(f"❌ Error inicializando stops para {position.symbol}: {e}")
            return position

    def update_trailing_stop_professional(self, position: Position, current_price: float) -> Tuple[Position, bool, str]:
        """
        📈 Sistema profesional de Trailing Stop por posición individual.
        Lógica simplificada y robusta para mayor fiabilidad.
        """
        try:
            stop_triggered = False
            trigger_reason = ""

            # ✅ VALIDACIÓN: Verificar que el precio sea válido
            if current_price <= 0:
                print(f"⚠️ Precio inválido para {position.symbol}: ${current_price:.4f} - Saltando trailing stop")
                return position, False, ""

            # ✅ VALIDACIÓN: Verificar que el precio no sea demasiado diferente del último conocido
            if hasattr(position, 'current_price') and position.current_price > 0:
                price_change_percent = abs((current_price - position.current_price) / position.current_price) * 100
                if price_change_percent > 10:  # Cambio > 10% podría ser error
                    print(f"⚠️ Cambio de precio sospechoso para {position.symbol}: {price_change_percent:.2f}% - Verificando...")
                    # Usar el precio más conservador para trailing stops
                    if position.side == 'BUY':
                        current_price = min(current_price, position.current_price)
                    else:
                        current_price = max(current_price, position.current_price)

            # ✅ CONFIGURACIÓN DESDE .ENV
            risk_params = get_risk_params()
            activation_pnl_percent = risk_params.trailing_activation_threshold
            trailing_percent = position.trailing_stop_percent

            if position.side == 'BUY':
                # --- LÓGICA PARA POSICIONES LONG ---

                # 1. ✅ CRÍTICO: Actualizar el precio más alto desde la entrada SIEMPRE
                if position.highest_price_since_entry is None or current_price > position.highest_price_since_entry:
                    old_highest = position.highest_price_since_entry
                    position.highest_price_since_entry = current_price
                    old_highest_str = f"${old_highest:.4f}" if old_highest is not None else "N/A"
                    print(f"🏔️ NUEVO MÁXIMO {position.symbol}: {old_highest_str} → ${current_price:.4f}")

                # 2. Calcular PnL actual
                current_pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100

                # 3. Activar el trailing stop si se alcanza el umbral de ganancia
                if not position.trailing_stop_active and current_pnl_percent >= activation_pnl_percent:
                    position.trailing_stop_active = True

                    # ✅ CÁLCULO INTELIGENTE MEJORADO: Protección proporcional
                    current_gain_percent = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100

                    # ✅ PROTECCIÓN PROPORCIONAL INTELIGENTE (80% de la ganancia actual)
                    if current_gain_percent >= 0.6:
                        # Proteger el 80% de la ganancia actual
                        min_profit_protection = current_gain_percent * 0.9
                    else:
                        # Protección mínima base configurable desde .env
                        risk_params = get_risk_params()
                        min_profit_protection = risk_params.min_profit_protection

                    min_trailing_price = position.entry_price * (1 + min_profit_protection / 100)

                    # Calcular trailing stop desde el máximo histórico
                    trailing_from_peak = position.highest_price_since_entry * (1 - trailing_percent / 100)

                    # Usar el mayor entre: protección progresiva o trailing desde pico
                    position.trailing_stop_price = max(trailing_from_peak, min_trailing_price)

                    position.last_trailing_update = datetime.now()
                    self._save_trailing_state(position)

                    protection_percent = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                    print(f"📈 TRAILING STOP ACTIVADO para {position.symbol} Pos #{position.order_id}:")
                    print(f"   📍 Precio entrada: ${position.entry_price:.4f}")
                    print(f"   💰 Precio actual: ${current_price:.4f}")
                    print(f"   🏔️ Precio máximo: ${position.highest_price_since_entry:.4f}")
                    print(f"   🎯 Ganancia actual: +{current_pnl_percent:.2f}% (Umbral: {activation_pnl_percent}%)")
                    print(f"   🚀 Stop inicial en: ${position.trailing_stop_price:.4f} (+{protection_percent:.2f}%)")

                # 4. ✅ CORREGIDO: Actualizar el trailing stop si ya está activo
                elif position.trailing_stop_active:
                    # ✅ CÁLCULO INTELIGENTE MEJORADO: Protección proporcional
                    current_gain_percent = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100

                    # ✅ PROTECCIÓN PROPORCIONAL INTELIGENTE (80% de la ganancia actual)
                    if current_gain_percent >= 0.6:
                        # Proteger el 80% de la ganancia actual
                        min_profit_protection = current_gain_percent * 0.90
                    else:
                        # Protección mínima base configurable desde .env
                        risk_params = get_risk_params()
                        min_profit_protection = risk_params.min_profit_protection

                    min_trailing_price = position.entry_price * (1 + min_profit_protection / 100)

                    # Calcular trailing stop desde el máximo histórico
                    trailing_from_peak = position.highest_price_since_entry * (1 - trailing_percent / 100)

                    # Usar el mayor entre: protección progresiva o trailing desde pico
                    new_trailing_price = max(trailing_from_peak, min_trailing_price)

                    # ✅ MOVER el stop solo si el nuevo precio es más alto que el anterior
                    if position.trailing_stop_price is None or new_trailing_price > position.trailing_stop_price:
                        old_price = position.trailing_stop_price
                        position.trailing_stop_price = new_trailing_price
                        position.last_trailing_update = datetime.now()
                        position.trailing_movements += 1
                        self._save_trailing_state(position)

                        profit_protection_percent = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                        print(f"📈 TRAILING STOP MOVIDO para {position.symbol} Pos #{position.order_id}:")
                        print(f"   📍 Precio entrada: ${position.entry_price:.4f}")
                        print(f"   💰 Precio actual: ${current_price:.4f}")
                        print(f"   🏔️ Precio máximo: ${position.highest_price_since_entry:.4f}")
                        print(f"   🔄 Stop: ${old_price:.4f} → ${new_trailing_price:.4f}")
                        print(f"   🛡️ Protegiendo ganancia de: +{profit_protection_percent:.2f}%")
                        print(f"   📊 Protección proporcional: +{min_profit_protection:.1f}% (80% de +{current_gain_percent:.2f}%)")
                    else:
                        # ✅ NUEVO: Log detallado cuando el trailing no se mueve
                        if position.trailing_stop_price is not None:
                            current_protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                            print(f"📊 TRAILING STOP MANTIENE {position.symbol}: ${position.trailing_stop_price:.4f} (+{current_protection:.2f}%)")
                            print(f"   💡 Calculado: ${new_trailing_price:.4f} | Desde pico: ${trailing_from_peak:.4f} | Mín: ${min_trailing_price:.4f}")

                # 5. Verificar si el precio actual ha caído por debajo del trailing stop
                if position.trailing_stop_active and position.trailing_stop_price is not None and current_price <= position.trailing_stop_price:
                    stop_triggered = True
                    trigger_reason = "TRAILING_STOP"
                    final_pnl = ((current_price - position.entry_price) / position.entry_price) * 100

                    print(f"🛑 TRAILING STOP EJECUTADO para {position.symbol} Pos #{position.order_id}:")
                    print(f"   📉 Precio actual: ${current_price:.4f} <= Stop: ${position.trailing_stop_price:.4f}")
                    print(f"   💰 PnL final estimado: {final_pnl:.2f}%")

                    # Limpiar estado del cache
                    if position.order_id in self.trailing_cache:
                        del self.trailing_cache[position.order_id]

            elif position.side == 'SELL':
                # --- LÓGICA PARA POSICIONES SHORT (MEJORADA) ---

                # 1. ✅ CRÍTICO: Actualizar el precio más bajo desde la entrada SIEMPRE
                if position.lowest_price_since_entry is None or current_price < position.lowest_price_since_entry:
                    old_lowest = position.lowest_price_since_entry
                    position.lowest_price_since_entry = current_price
                    old_lowest_str = f"${old_lowest:.4f}" if old_lowest is not None else "N/A"
                    print(f"📉 NUEVO MÍNIMO {position.symbol}: {old_lowest_str} → ${current_price:.4f}")

                # 2. Calcular PnL actual
                current_pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100

                # 3. Activar el trailing stop si se alcanza el umbral de ganancia
                if not position.trailing_stop_active and current_pnl_percent >= activation_pnl_percent:
                    position.trailing_stop_active = True

                    # ✅ CÁLCULO INTELIGENTE MEJORADO: Protección proporcional para SHORTS
                    current_gain_percent = ((position.entry_price - position.lowest_price_since_entry) / position.entry_price) * 100

                    # ✅ PROTECCIÓN PROPORCIONAL INTELIGENTE (80% de la ganancia actual)
                    if current_gain_percent >= 0.6:
                        # Proteger el 80% de la ganancia actual
                        min_profit_protection = current_gain_percent * 0.9
                    else:
                        # Protección mínima base configurable desde .env
                        risk_params = get_risk_params()
                        min_profit_protection = risk_params.min_profit_protection

                    min_trailing_price = position.entry_price * (1 - min_profit_protection / 100)

                    # Calcular trailing stop desde el mínimo histórico
                    trailing_from_peak = position.lowest_price_since_entry * (1 + trailing_percent / 100)

                    # Usar el menor entre: protección progresiva o trailing desde pico
                    position.trailing_stop_price = min(trailing_from_peak, min_trailing_price)

                    position.last_trailing_update = datetime.now()
                    self._save_trailing_state(position)

                    protection_percent = ((position.entry_price - position.trailing_stop_price) / position.entry_price) * 100
                    print(f"📈 TRAILING STOP (SHORT) ACTIVADO para {position.symbol} Pos #{position.order_id}:")
                    print(f"   📍 Precio entrada: ${position.entry_price:.4f}")
                    print(f"   💰 Precio actual: ${current_price:.4f}")
                    print(f"   📉 Precio mínimo: ${position.lowest_price_since_entry:.4f}")
                    print(f"   🎯 Ganancia actual: +{current_pnl_percent:.2f}% (Umbral: {activation_pnl_percent}%)")
                    print(f"   🚀 Stop inicial en: ${position.trailing_stop_price:.4f} (+{protection_percent:.2f}%)")

                # 4. ✅ CORREGIDO: Actualizar el trailing stop si ya está activo
                elif position.trailing_stop_active:
                    # ✅ CÁLCULO INTELIGENTE MEJORADO: Protección proporcional para SHORTS
                    current_gain_percent = ((position.entry_price - position.lowest_price_since_entry) / position.entry_price) * 100

                    # ✅ PROTECCIÓN PROPORCIONAL INTELIGENTE (80% de la ganancia actual)
                    if current_gain_percent >= 0.6:
                        # Proteger el 80% de la ganancia actual
                        min_profit_protection = current_gain_percent * 0.90
                    else:
                        # Protección mínima base configurable desde .env
                        risk_params = get_risk_params()
                        min_profit_protection = risk_params.min_profit_protection

                    min_trailing_price = position.entry_price * (1 - min_profit_protection / 100)

                    # Calcular trailing stop desde el mínimo histórico
                    trailing_from_peak = position.lowest_price_since_entry * (1 + trailing_percent / 100)

                    # Usar el menor entre: protección progresiva o trailing desde pico
                    new_trailing_price = min(trailing_from_peak, min_trailing_price)

                    # ✅ MOVER el stop solo si el nuevo precio es más bajo que el anterior
                    if position.trailing_stop_price is None or new_trailing_price < position.trailing_stop_price:
                        old_price = position.trailing_stop_price
                        position.trailing_stop_price = new_trailing_price
                        position.last_trailing_update = datetime.now()
                        position.trailing_movements += 1
                        self._save_trailing_state(position)

                        profit_protection_percent = ((position.entry_price - position.trailing_stop_price) / position.entry_price) * 100
                        print(f"📈 TRAILING STOP (SHORT) MOVIDO para {position.symbol} Pos #{position.order_id}:")
                        print(f"   📍 Precio entrada: ${position.entry_price:.4f}")
                        print(f"   💰 Precio actual: ${current_price:.4f}")
                        print(f"   📉 Precio mínimo: ${position.lowest_price_since_entry:.4f}")
                        print(f"   🔄 Stop: ${old_price:.4f} → ${new_trailing_price:.4f}")
                        print(f"   🛡️ Protegiendo ganancia de: +{profit_protection_percent:.2f}%")
                        print(f"   📊 Protección proporcional: +{min_profit_protection:.1f}% (80% de +{current_gain_percent:.2f}%)")
                    else:
                        # ✅ NUEVO: Log detallado cuando el trailing no se mueve
                        if position.trailing_stop_price is not None:
                            current_protection = ((position.entry_price - position.trailing_stop_price) / position.entry_price) * 100
                            print(f"📊 TRAILING STOP (SHORT) MANTIENE {position.symbol}: ${position.trailing_stop_price:.4f} (+{current_protection:.2f}%)")
                            print(f"   💡 Calculado: ${new_trailing_price:.4f} | Desde pico: ${trailing_from_peak:.4f} | Mín: ${min_trailing_price:.4f}")

                # 5. Verificar si el precio actual ha subido por encima del trailing stop
                if position.trailing_stop_active and position.trailing_stop_price is not None and current_price >= position.trailing_stop_price:
                    stop_triggered = True
                    trigger_reason = "TRAILING_STOP"
                    final_pnl = ((position.entry_price - current_price) / position.entry_price) * 100

                    print(f"🛑 TRAILING STOP (SHORT) EJECUTADO para {position.symbol} Pos #{position.order_id}:")
                    print(f"   📈 Precio actual: ${current_price:.4f} >= Stop: ${position.trailing_stop_price:.4f}")
                    print(f"   💰 PnL final estimado: {final_pnl:.2f}%")

                    # Limpiar estado del cache
                    if position.order_id in self.trailing_cache:
                        del self.trailing_cache[position.order_id]

            # Verificar stop loss tradicional solo si el trailing no está activo
            if not position.trailing_stop_active:
                if position.side == 'BUY' and position.stop_loss_price and current_price <= position.stop_loss_price:
                    stop_triggered = True
                    trigger_reason = "STOP_LOSS"
                    print(f"🛑 STOP LOSS TRADICIONAL para {position.symbol}")
                elif position.side == 'SELL' and position.stop_loss_price and current_price >= position.stop_loss_price:
                    stop_triggered = True
                    trigger_reason = "STOP_LOSS"
                    print(f"🛑 STOP LOSS TRADICIONAL (SHORT) para {position.symbol}")

            return position, stop_triggered, trigger_reason

        except Exception as e:
            print(f"❌ Error en trailing stop para {position.symbol}: {e}")
            return position, False, ""

    def get_atr_based_trailing_distance(self, symbol: str, periods: int = 14) -> float:
        """📊 Calcular distancia de trailing basada en ATR (Average True Range)"""
        try:
            # Esta sería una implementación más avanzada usando ATR
            # Por ahora, usar porcentajes adaptativos según el activo

            atr_multipliers = {
                'BTC': 1.5,    # Menos volátil, trailing más cercano
                'ETH': 2.0,    # Volatilidad media
                'BNB': 2.5,    # Más volátil, trailing más amplio
                'ADA': 3.0,    # Altcoin más volátil
                'default': 2.0
            }

            # Extraer el asset del símbolo
            asset = symbol.replace('USDT', '').replace('BUSD', '')
            multiplier = atr_multipliers.get(asset, atr_multipliers['default'])

            # Retornar porcentaje adaptativo
            base_percent = 2.0
            adaptive_percent = base_percent * multiplier

            return min(adaptive_percent, 5.0)  # Máximo 5%

        except Exception as e:
            print(f"❌ Error calculando ATR para {symbol}: {e}")
            return 2.0  # Default fallback

    def debug_trailing_cache(self):
        """🔍 Mostrar estado actual del cache de trailing stops para debugging"""
        try:
            print(f"\n🔍 DEBUG TRAILING CACHE ({len(self.trailing_cache)} entradas):")

            if not self.trailing_cache:
                print("   📭 Cache vacío - No hay trailing stops guardados")
                return

            for order_id, state in self.trailing_cache.items():
                active = state.get('trailing_stop_active', False)
                price = state.get('trailing_stop_price', 0)
                movements = state.get('trailing_movements', 0)

                status = "ACTIVO" if active else "INACTIVO"
                print(f"   📋 {order_id}: {status}")
                if active:
                    print(f"      💰 Precio: ${price:.4f}")
                    print(f"      📊 Movimientos: {movements}")

        except Exception as e:
            print(f"❌ Error en debug trailing cache: {e}")

    def generate_trailing_stop_report(self, positions: List[Position]) -> str:
        """📊 Generar reporte detallado de trailing stops"""
        try:
            if not positions:
                return "📈 No hay posiciones con trailing stop activo"

            report = "📈 **TRAILING STOPS ACTIVOS**\n"

            active_trailing = [pos for pos in positions if hasattr(pos, 'trailing_stop_active') and pos.trailing_stop_active]

            if not active_trailing:
                return "📈 No hay trailing stops activos"

            for pos in active_trailing:
                current_protection = 0.0
                if pos.trailing_stop_price and pos.entry_price:
                    if pos.side == 'BUY':
                        current_protection = ((pos.trailing_stop_price - pos.entry_price) / pos.entry_price) * 100
                    else:
                        current_protection = ((pos.entry_price - pos.trailing_stop_price) / pos.entry_price) * 100

                max_profit = 0.0
                if pos.side == 'BUY' and pos.highest_price_since_entry:
                    max_profit = ((pos.highest_price_since_entry - pos.entry_price) / pos.entry_price) * 100
                elif pos.side == 'SELL' and pos.lowest_price_since_entry:
                    max_profit = ((pos.entry_price - pos.lowest_price_since_entry) / pos.entry_price) * 100

                report += f"\n🎯 **{pos.symbol} Pos #{pos.order_id}**\n"
                report += f"├─ Entrada: ${pos.entry_price:.4f}\n"
                report += f"├─ Actual: ${pos.current_price:.4f}\n"
                report += f"├─ Trailing: ${pos.trailing_stop_price:.4f}\n"
                report += f"├─ Protegiendo: +{current_protection:.2f}%\n"
                report += f"├─ Máximo: +{max_profit:.2f}%\n"
                report += f"└─ Movimientos: {pos.trailing_movements}\n"

            return report

        except Exception as e:
            print(f"❌ Error generando reporte trailing: {e}")
            return "❌ Error en reporte trailing stops"

    def _load_trailing_cache(self):
        """💾 Cargar estado del trailing stop desde archivo"""
        try:
            if os.path.exists(self.trailing_cache_file):
                with open(self.trailing_cache_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"❌ Error cargando trailing cache: {e}")
            return {}

    def _save_trailing_cache(self):
        """💾 Guardar estado del trailing stop en archivo"""
        try:
            with open(self.trailing_cache_file, 'w') as f:
                json.dump(self.trailing_cache, f)
        except Exception as e:
            print(f"❌ Error guardando trailing cache: {e}")

    def _calculate_orders_hash(self, orders: List[TradeOrder]) -> str:
        """🔢 Calcular hash de órdenes para detectar cambios"""
        try:
            # Crear string único basado en órdenes
            orders_str = ""
            for order in sorted(orders, key=lambda x: x.order_id):
                orders_str += f"{order.order_id}_{order.executed_qty}_{order.time.isoformat()}"

            import hashlib
            return hashlib.md5(orders_str.encode()).hexdigest()
        except Exception as e:
            print(f"❌ Error calculando hash de órdenes: {e}")
            return ""

    def sync_positions_with_orders(self, orders: List[TradeOrder], balances: Dict[str, Dict]):
        """🔄 Sincronizar registry de posiciones con órdenes (solo cambios)"""
        try:
            print("🔄 Sincronizando posiciones con órdenes...")

            # 1. Crear posiciones nuevas basadas en órdenes
            new_positions = self.group_orders_into_positions(orders, balances)

            # 2. Crear diccionario de nuevas posiciones por order_id
            new_positions_dict = {pos.order_id: pos for pos in new_positions}

            # 3. Identificar posiciones que ya no existen (vendidas completamente)
            positions_to_remove = []
            for order_id in self.position_registry.keys():
                if order_id not in new_positions_dict:
                    positions_to_remove.append(order_id)
                    print(f"🗑️ Posición eliminada: {order_id} (vendida completamente)")

            # 4. Eliminar posiciones que ya no existen
            for order_id in positions_to_remove:
                del self.position_registry[order_id]
                # También limpiar cache de trailing
                if order_id in self.trailing_cache:
                    del self.trailing_cache[order_id]

            # 5. Agregar/actualizar posiciones
            for order_id, new_position in new_positions_dict.items():
                if order_id in self.position_registry:
                    # Posición existente: preservar trailing stops, actualizar datos básicos
                    existing_position = self.position_registry[order_id]

                    # Preservar estado de trailing stops
                    new_position.trailing_stop_active = existing_position.trailing_stop_active
                    new_position.trailing_stop_price = existing_position.trailing_stop_price
                    new_position.trailing_stop_percent = existing_position.trailing_stop_percent
                    new_position.highest_price_since_entry = existing_position.highest_price_since_entry
                    new_position.lowest_price_since_entry = existing_position.lowest_price_since_entry
                    new_position.trailing_activation_threshold = existing_position.trailing_activation_threshold
                    new_position.last_trailing_update = existing_position.last_trailing_update
                    new_position.trailing_movements = existing_position.trailing_movements

                    # Preservar stops tradicionales
                    new_position.stop_loss_price = existing_position.stop_loss_price
                    new_position.take_profit_price = existing_position.take_profit_price
                    new_position.stop_loss_percent = existing_position.stop_loss_percent
                    new_position.take_profit_percent = existing_position.take_profit_percent

                    print(f"🔄 Posición actualizada: {order_id} (trailing preservado: {new_position.trailing_stop_active})")
                else:
                    # Posición nueva: inicializar stops
                    new_position = self.initialize_position_stops(new_position)
                    print(f"🆕 Nueva posición: {order_id}")

                # Actualizar registry
                self.position_registry[order_id] = new_position

            print(f"✅ Registry sincronizado: {len(self.position_registry)} posiciones activas")

        except Exception as e:
            print(f"❌ Error sincronizando posiciones: {e}")

    async def update_existing_positions_prices(self, prices: Dict[str, float]):
        """💰 Actualizar precios y PnL de posiciones existentes en el registry"""
        try:
            for order_id, position in self.position_registry.items():
                # Obtener precio actual
                current_price = prices.get(position.symbol, position.current_price)

                # Actualizar precio y valores
                position.current_price = current_price
                position.market_value = position.quantity * current_price

                # Recalcular PnL
                entry_value = position.quantity * position.entry_price
                position.unrealized_pnl_usd = position.market_value - entry_value
                position.unrealized_pnl_percent = (position.unrealized_pnl_usd / entry_value) * 100 if entry_value > 0 else 0

                # Actualizar duración (corregir timezone awareness)
                current_time = datetime.now()
                if position.entry_time.tzinfo is not None:
                    # Si entry_time tiene timezone, usar UTC para current_time
                    from datetime import timezone
                    current_time = datetime.now(timezone.utc)
                    if position.entry_time.tzinfo != timezone.utc:
                        # Convertir entry_time a UTC si tiene otro timezone
                        entry_time_utc = position.entry_time.astimezone(timezone.utc)
                    else:
                        entry_time_utc = position.entry_time
                else:
                    # Si entry_time es naive, usar current_time naive
                    entry_time_utc = position.entry_time

                position.duration_minutes = int((current_time - entry_time_utc).total_seconds() / 60)

        except Exception as e:
            print(f"❌ Error actualizando precios de posiciones: {e}")

async def test_portfolio_manager():
    """🧪 Probar Portfolio Manager"""
    print("🧪 TESTING PORTFOLIO MANAGER")
    print("=" * 50)

    try:
        # Configuración
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision')

        if not api_key or not secret_key:
            print("❌ Faltan credenciales de Binance")
            return

        # Crear manager
        portfolio_manager = ProfessionalPortfolioManager(api_key, secret_key, base_url)

        # Obtener snapshot
        print("📊 Obteniendo snapshot del portafolio...")
        snapshot = await portfolio_manager.get_portfolio_snapshot()

        # Generar reporte TCN
        print("\n" + "="*60)
        tcn_report = portfolio_manager.format_tcn_style_report(snapshot)
        print(tcn_report)
        print("="*60)

        # Reporte compacto
        compact_report = portfolio_manager.format_compact_report(snapshot)
        print(f"\n📱 Compacto: {compact_report}")

        print(f"\n✅ Test completado - {portfolio_manager.api_calls_count} API calls realizadas")

    except Exception as e:
        print(f"❌ Error en test: {e}")

if __name__ == "__main__":
    asyncio.run(test_portfolio_manager())
