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

load_dotenv()

@dataclass
class Position:
    """📈 Posición individual en el portafolio"""
    symbol: str
    side: str  # BUY o SELL
    size: float  # Cantidad del activo
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
    trailing_stop_percent: float = 2.0  # Default 2%
    highest_price_since_entry: Optional[float] = None  # Para tracking del máximo
    lowest_price_since_entry: Optional[float] = None   # Para shorts
    trailing_activation_threshold: float = 1.0  # Activar trailing después de +1% ganancia
    last_trailing_update: Optional[datetime] = None
    trailing_movements: int = 0  # Contador de movimientos del trailing
    
    # Stop Loss y Take Profit tradicionales
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_percent: float = 3.0  # Default 3%
    take_profit_percent: float = 6.0  # Default 6%

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
    small_positions: List[Position] = None  # ✅ NUEVO: Posiciones pequeñas filtradas

class ProfessionalPortfolioManager:
    """💼 Gestor Profesional de Portafolio"""
    
    def __init__(self, binance_config: object, trading_symbols: List[str], logger: object):
        """Inicializar el Portfolio Manager.

        Args:
            binance_config (object): Configuración de la API de Binance.
            trading_symbols (List[str]): Lista de símbolos a operar.
            logger (object): Objeto logger para registrar eventos.
        """
        self.config = binance_config
        self.trading_symbols = trading_symbols
        self.logger = logger
        self.client = None
        
        # Cache de precios para evitar llamadas innecesarias
        self.price_cache = {}
        self.last_price_update = None
        
        # ✅ NUEVO: Cache de órdenes para tracking de posiciones
        self.orders_cache = {}
        self.last_orders_update = None
        
        # Configuración
        self.max_positions = 5
        self.min_position_value = 10.0  # USD mínimo por posición
        
        # ✅ NUEVO: Configuración para historial de órdenes
        self.days_to_lookback = 30  # Días hacia atrás para buscar órdenes
        
        # Métricas
        self.api_calls_count = 0
        self.last_snapshot_time = None
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeOrder] = []
        self.balance: Dict[str, float] = {}
        self.last_snapshot: Optional[PortfolioSnapshot] = None
        self.last_update: Optional[datetime] = None
        
    async def initialize(self) -> Optional[float]:
        """
        🚀 Inicializa el Portfolio Manager, obtiene los balances iniciales de la cuenta
        y retorna el balance de USDT.
        """
        self.logger.info("💼 Obteniendo balance inicial desde Binance...")
        try:
            balances = await self.get_account_balances()
            usdt_balance = balances.get('USDT', {}).get('free', 0.0)
            
            self.logger.info(f"✅ Balance USDT inicial obtenido: ${usdt_balance:.2f}")
            return usdt_balance
            
        except Exception as e:
            self.logger.error(f"❌ No se pudo obtener el balance inicial. Error: {e}")
            return None

    def _generate_signature(self, params: str) -> str:
        """🔐 Generar firma HMAC SHA256 para Binance"""
        return hmac.new(
            self.config.BINANCE_SECRET_KEY.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _make_authenticated_request(self, endpoint: str, params: Dict = None) -> Dict:
        """🔗 Realizar petición autenticada a Binance"""
        if params is None:
            params = {}
        
        # Añadir timestamp y recvWindow
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 60000
        
        # Crear query string
        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        
        # Generar firma
        signature = self._generate_signature(query_string)
        query_string += f"&signature={signature}"
        
        # Headers
        headers = {
            'X-MBX-APIKEY': self.config.BINANCE_API_KEY
        }
        
        # Realizar petición
        url = f"{self.config.BINANCE_BASE_URL}/api/v3/{endpoint}?{query_string}"
        
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
                url = f"{self.config.BINANCE_BASE_URL}/api/v3/ticker/price"
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
        """💲 Actualizar precios de múltiples símbolos en paralelo"""
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        prices = await asyncio.gather(*tasks)
        
        price_dict = {}
        for symbol, price in zip(symbols, prices):
            if price > 0:
                price_dict[symbol] = price
        
        self.last_price_update = datetime.now()
        return price_dict
    
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
    
    async def get_order_history(self, symbol: str = None, days_back: int = None) -> List[TradeOrder]:
        """📋 Obtener historial de órdenes ejecutadas"""
        try:
            if days_back is None:
                days_back = self.days_to_lookback
            
            # Calcular timestamp de inicio (días hacia atrás)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            orders = []
            
            if symbol:
                # Validar que el símbolo no esté vacío y tenga formato válido
                if not symbol or len(symbol) < 6:
                    print(f"⚠️ Símbolo inválido: '{symbol}' - saltando")
                    return []
                
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
                # En lugar de consultar todos los activos, usamos los símbolos de trading configurados
                self.logger.info(f"📋 Consultando órdenes para los símbolos de trading: {self.trading_symbols}")
                
                for symbol_pair in self.trading_symbols:
                    try:
                        self.logger.info(f"   🔍 Consultando órdenes para {symbol_pair}...")
                        symbol_orders = await self.get_order_history(symbol_pair, days_back)
                        if symbol_orders:
                            self.logger.info(f"      ✅ Encontradas {len(symbol_orders)} órdenes para {symbol_pair}")
                            orders.extend(symbol_orders)
                        else:
                            self.logger.info(f"      📭 Sin órdenes para {symbol_pair}")
                    except Exception as e:
                        self.logger.error(f"❌ Error obteniendo historial de órdenes para {symbol_pair}: {e}")
                        # Continuar con el siguiente símbolo
                        continue
            
            self.logger.info(f"   📄 Encontradas {len(orders)} órdenes ejecutadas en total.")
            
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
                            size=qty_from_this_order,
                            entry_price=buy_order.price,
                            current_price=current_price,
                            market_value=market_value,
                            unrealized_pnl_usd=pnl_usd,
                            unrealized_pnl_percent=pnl_percent,
                            entry_time=buy_order.time,
                            duration_minutes=duration_minutes,
                            order_id=f"{len(orders)}ord_{buy_order.order_id}",  # ID único para la posición
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
        """📊 Obtener snapshot completo del portafolio"""
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
            
            # 4. ✅ NUEVO: Obtener historial de órdenes para posiciones reales
            print("📋 Obteniendo historial de órdenes...")
            all_orders = await self.get_order_history(days_back=self.days_to_lookback)
            print(f"   📄 Encontradas {len(all_orders)} órdenes ejecutadas")
            
            # 5. ✅ NUEVO: Agrupar órdenes en posiciones individuales
            print("🔄 Agrupando órdenes en posiciones...")
            individual_positions = self.group_orders_into_positions(all_orders, balances)
            print(f"   📈 Identificadas {len(individual_positions)} posiciones individuales")
            
            # 6. Calcular valor de cada activo
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
            
            # 7. Calcular porcentajes
            for asset in all_assets:
                asset.percentage_of_portfolio = (asset.usd_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0
            
            # 8. ✅ NUEVO: Filtrar posiciones por valor mínimo y separar pequeñas
            print(f"🔍 DEBUG: Analizando {len(individual_positions)} posiciones antes del filtro:")
            active_positions = []
            small_positions = []
            
            for i, pos in enumerate(individual_positions, 1):
                if pos.market_value >= self.min_position_value:
                    active_positions.append(pos)
                    status = "✅ INCLUIDA"
                else:
                    small_positions.append(pos)
                    status = "📊 PEQUEÑA"
                print(f"   {i}. {pos.symbol} Pos #{pos.order_id}: ${pos.market_value:.2f} - {status}")
            
            print(f"📊 RESULTADO: {len(active_positions)} activas, {len(small_positions)} pequeñas (mín: ${self.min_position_value})")
            
            # 9. Calcular PnL total (incluyendo posiciones pequeñas)
            all_positions = active_positions + small_positions
            total_unrealized_pnl = sum(pos.unrealized_pnl_usd for pos in all_positions)
            total_unrealized_pnl_percent = (total_unrealized_pnl / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0
            
            # 10. Crear snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_balance_usd=total_portfolio_value,
                free_usdt=free_usdt,
                total_unrealized_pnl=total_unrealized_pnl,
                total_unrealized_pnl_percent=total_unrealized_pnl_percent,
                active_positions=active_positions,
                all_assets=all_assets,
                position_count=len(active_positions),
                max_positions=self.max_positions,
                total_trades_today=len([o for o in all_orders if o.time.date() == datetime.now().date()]),
                small_positions=small_positions
            )
            
            self.last_snapshot_time = datetime.now()
            print(f"✅ Snapshot obtenido: {len(all_assets)} activos, {len(active_positions)} posiciones activas, {len(small_positions)} posiciones pequeñas")
            
            return snapshot
            
        except Exception as e:
            print(f"❌ Error obteniendo snapshot: {e}")
            raise
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """🔍 Obtener posición específica por símbolo"""
        try:
            # Obtener snapshot actual
            snapshot = await self.get_portfolio_snapshot()
            
            # Buscar posición para el símbolo
            for position in snapshot.active_positions:
                if position.symbol == symbol:
                    return position
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo posición para {symbol}: {e}")
            return None
    
    def format_tcn_style_report(
        self, 
        snapshot: PortfolioSnapshot, 
        market_regime: Optional[str] = None, 
        tcn_predictions: Optional[List[Dict]] = None
    ) -> str:
        """
        Formatea un reporte de estado completo al estilo TCN, ahora con contexto de mercado.

        Args:
            snapshot (PortfolioSnapshot): El snapshot del portafolio.
            market_regime (Optional[str]): El régimen de mercado actual.
            tcn_predictions (Optional[List[Dict]]): Lista de predicciones de los modelos.

        Returns:
            str: El reporte formateado como un string.
        """
        report_lines = [
            "**🚀 TCN SIGNALS - {time}**".format(time=snapshot.timestamp.strftime('%H:%M:%S')),
            "📊 **Recomendaciones del Modelo Profesional**",
            ""
        ]

        # --- CONTEXTO DE MERCADO (NUEVA SECCIÓN) ---
        if market_regime:
            report_lines.append("**🏛️ CONTEXTO DE MERCADO**")
            report_lines.append(f"**Régimen:** {market_regime}")
            report_lines.append("")

        # --- ESTADO DE MODELOS TCN (NUEVA SECCIÓN) ---
        if tcn_predictions:
            report_lines.append("**🤖 ESTADO DE MODELOS TCN**")
            for pred in tcn_predictions:
                signal = pred.get('signal', 'N/A')
                confidence = pred.get('confidence', 0) * 100
                price = pred.get('current_price', 0)
                
                emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                
                report_lines.append(
                    f"{emoji} **{pred['pair']}**: {signal} ({confidence:.1f}%)"
                )
                report_lines.append(f"   └ Precio: ${price:,.4f}")
            report_lines.append("")

        # --- POSICIONES ACTIVAS (Sección existente) ---
        report_lines.append(f"**📈 POSICIONES ACTIVAS ({snapshot.position_count})**")
        if not snapshot.active_positions:
            report_lines.append("   *(No hay posiciones activas)*")
        else:
            for pos in snapshot.active_positions:
                pnl_sign = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"
                report_lines.append(f"**{pos.symbol}: {pos.side}**")
                report_lines.append(
                    f"└ ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} "
                    f"({pos.unrealized_pnl_percent:+.2f}% = ${pos.unrealized_pnl_usd:+.2f}) {pnl_sign}"
                )
                report_lines.append(
                    f"   💰 Cantidad: {pos.size:.6f} | 🕐 {pos.duration_minutes}min"
                )
        
        # ✅ NUEVO: Sección de Posiciones Pequeñas
        small_positions = getattr(snapshot, 'small_positions', [])
        if small_positions:
            report_lines.append("")
            report_lines.append(f"**📊 POSICIONES PEQUEÑAS ({len(small_positions)})** *(< ${self.min_position_value})*")
            for pos in small_positions:
                pnl_sign = "🟢" if pos.unrealized_pnl_usd >= 0 else "🔴"
                report_lines.append(f"**{pos.symbol}: {pos.side}** (${pos.market_value:.2f})")
                report_lines.append(
                    f"└ ${pos.entry_price:,.2f} → ${pos.current_price:,.2f} "
                    f"({pos.unrealized_pnl_percent:+.2f}% = ${pos.unrealized_pnl_usd:+.2f}) {pnl_sign}"
                )
        
        report_lines.append("")

        # --- RESUMEN RÁPIDO (Sección existente) ---
        report_lines.append("**⚡ RESUMEN RÁPIDO**")
        report_lines.append(f"💰 **USDT Libre:** ${snapshot.free_usdt:,.2f}")
        report_lines.append(f"📉 **P&L No Realizado:** ${snapshot.total_unrealized_pnl:,.2f}")
        report_lines.append(f"🎯 **Posiciones:** {snapshot.position_count}/{snapshot.max_positions}")
        report_lines.append(f"📊 **Trades Totales:** {snapshot.total_trades_today}")
        report_lines.append("")

        # --- DETALLE DEL PORTAFOLIO (Sección existente) ---
        report_lines.append("**💼 DETALLE DEL PORTAFOLIO**")
        top_assets = sorted([a for a in snapshot.all_assets if a.symbol != 'USDT'], key=lambda x: x.usd_value, reverse=True)[:3]
        for asset in top_assets:
            report_lines.append(f"🪙 **{asset.symbol}:** {asset.total:,.6f} (${asset.usd_value:,.2f})")
        report_lines.append(f"💵 **USDT:** ${snapshot.free_usdt:,.2f}")
        report_lines.append("")
        
        report_lines.append(f"💎 **VALOR TOTAL: ${snapshot.total_balance_usd:,.2f}**")
        report_lines.append("")
        report_lines.append(f"🔄 *Actualización cada {self.config.DISCORD_REPORT_INTERVAL_SECONDS // 60} min • {snapshot.timestamp.strftime('%d/%m/%y, %H:%M')}*")

        return "\n".join(report_lines)
    
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
    
    def initialize_position_stops(self, position: Position) -> Position:
        """🛡️ Inicializar stops tradicionales y trailing para nueva posición"""
        try:
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
            
            # Trailing stop inicialmente inactivo
            position.trailing_stop_active = False
            position.trailing_stop_price = None
            position.last_trailing_update = datetime.now()
            position.trailing_movements = 0
            
            print(f"🛡️ Stops inicializados para {position.symbol} Pos #{position.order_id}:")
            print(f"   📍 Entrada: ${position.entry_price:.4f}")
            print(f"   🛑 Stop Loss: ${position.stop_loss_price:.4f} (-{position.stop_loss_percent}%)")
            print(f"   🎯 Take Profit: ${position.take_profit_price:.4f} (+{position.take_profit_percent}%)")
            print(f"   📈 Trailing: INACTIVO (activar en +{position.trailing_activation_threshold}%)")
            
            return position
            
        except Exception as e:
            print(f"❌ Error inicializando stops para {position.symbol}: {e}")
            return position
    
    def update_trailing_stop_professional(self, position: Position, current_price: float) -> Tuple[Position, bool, str]:
        """📈 Sistema profesional de Trailing Stop por posición individual"""
        try:
            stop_triggered = False
            trigger_reason = ""
            
            if position.side == 'BUY':
                # ✅ LONG POSITION LOGIC
                
                # 1. Actualizar precio máximo histórico
                if position.highest_price_since_entry is None or current_price > position.highest_price_since_entry:
                    position.highest_price_since_entry = current_price
                
                # 2. Calcular ganancia actual
                current_pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
                
                # 3. ✅ LÓGICA CORREGIDA: Verificar si debe activarse el trailing stop
                # Solo activar cuando la ganancia sea suficiente para proteger ganancias reales
                min_activation_needed = position.trailing_stop_percent + 0.5  # 2% + 0.5% buffer = 2.5%
                
                if not position.trailing_stop_active and current_pnl_percent >= min_activation_needed:
                    position.trailing_stop_active = True
                    
                    # ✅ NUEVA LÓGICA: Asegurar que el trailing inicial proteja ganancias
                    # Trailing = max(precio_actual * (1-trailing%), precio_entrada * (1+0.5%))
                    trailing_by_percent = current_price * (1 - position.trailing_stop_percent / 100)
                    trailing_min_profit = position.entry_price * (1 + 0.5 / 100)  # Mínimo +0.5% ganancia
                    
                    position.trailing_stop_price = max(trailing_by_percent, trailing_min_profit)
                    position.last_trailing_update = datetime.now()
                    
                    protected_profit = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                    
                    print(f"📈 TRAILING ACTIVADO {position.symbol} Pos #{position.order_id}:")
                    print(f"   🎯 Ganancia actual: +{current_pnl_percent:.2f}%")
                    print(f"   📈 Trailing Stop: ${position.trailing_stop_price:.4f}")
                    print(f"   🛡️ Ganancia protegida: +{protected_profit:.2f}%")
                
                # 4. Actualizar trailing stop si está activo
                elif position.trailing_stop_active:
                    # Calcular nuevo trailing basado en el máximo histórico
                    trailing_by_percent = position.highest_price_since_entry * (1 - position.trailing_stop_percent / 100)
                    
                    # Asegurar que nunca baje del precio de entrada (breakeven mínimo)
                    trailing_min_breakeven = position.entry_price * 1.001  # +0.1% mínimo
                    new_trailing_price = max(trailing_by_percent, trailing_min_breakeven)
                    
                    # Solo mover trailing stop hacia arriba (más favorable)
                    if new_trailing_price > position.trailing_stop_price:
                        old_price = position.trailing_stop_price
                        position.trailing_stop_price = new_trailing_price
                        position.last_trailing_update = datetime.now()
                        position.trailing_movements += 1
                        
                        profit_protection = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                        
                        print(f"📈 TRAILING MOVIDO {position.symbol} Pos #{position.order_id}:")
                        print(f"   🔄 ${old_price:.4f} → ${new_trailing_price:.4f}")
                        print(f"   🏔️ Máximo: ${position.highest_price_since_entry:.4f}")
                        print(f"   🛡️ Protegiendo: +{profit_protection:.2f}% ganancia")
                        print(f"   📊 Movimiento #{position.trailing_movements}")
                
                # 5. Verificar si se debe cerrar por trailing stop
                if position.trailing_stop_active and current_price <= position.trailing_stop_price:
                    stop_triggered = True
                    trigger_reason = "TRAILING_STOP"
                    
                    final_pnl = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                    max_profit = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100
                    
                    print(f"🛑 TRAILING STOP EJECUTADO {position.symbol} Pos #{position.order_id}:")
                    print(f"   📉 Precio: ${current_price:.4f} <= Trailing: ${position.trailing_stop_price:.4f}")
                    print(f"   💰 PnL Final: +{final_pnl:.2f}%")
                    print(f"   🏔️ Máximo alcanzado: +{max_profit:.2f}%")
                    print(f"   📈 Movimientos trailing: {position.trailing_movements}")
                    print(f"   🎯 NOTA: Esta es solo una ALERTA - NO se ejecuta orden automática")
                
                # 6. Verificar stop loss tradicional (solo si trailing no está activo o es menor)
                elif current_price <= position.stop_loss_price:
                    if not position.trailing_stop_active or position.stop_loss_price > position.trailing_stop_price:
                        stop_triggered = True
                        trigger_reason = "STOP_LOSS"
                        
                        loss_pnl = ((position.stop_loss_price - position.entry_price) / position.entry_price) * 100
                        print(f"🛑 STOP LOSS TRADICIONAL {position.symbol} Pos #{position.order_id}: {loss_pnl:.2f}%")
                        print(f"   🎯 NOTA: Esta es solo una ALERTA - NO se ejecuta orden automática")
            
            else:
                # ✅ SHORT POSITION LOGIC (para futuros)
                
                # Actualizar precio mínimo histórico
                if position.lowest_price_since_entry is None or current_price < position.lowest_price_since_entry:
                    position.lowest_price_since_entry = current_price
                
                # Calcular ganancia para short
                current_pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
                
                # Activar trailing solo con ganancia suficiente
                min_activation_needed = position.trailing_stop_percent + 0.5
                
                if not position.trailing_stop_active and current_pnl_percent >= min_activation_needed:
                    position.trailing_stop_active = True
                    
                    # Para shorts: trailing arriba del precio actual
                    trailing_by_percent = current_price * (1 + position.trailing_stop_percent / 100)
                    trailing_max_loss = position.entry_price * (1 - 0.5 / 100)  # Máximo -0.5% pérdida
                    
                    position.trailing_stop_price = min(trailing_by_percent, trailing_max_loss)
                    position.last_trailing_update = datetime.now()
                
                # Actualizar trailing (solo hacia abajo para shorts)
                elif position.trailing_stop_active:
                    trailing_by_percent = position.lowest_price_since_entry * (1 + position.trailing_stop_percent / 100)
                    trailing_max_breakeven = position.entry_price * 0.999  # -0.1% máximo
                    new_trailing_price = min(trailing_by_percent, trailing_max_breakeven)
                    
                    if new_trailing_price < position.trailing_stop_price:
                        position.trailing_stop_price = new_trailing_price
                        position.last_trailing_update = datetime.now()
                        position.trailing_movements += 1
                
                # Verificar cierre por trailing
                if position.trailing_stop_active and current_price >= position.trailing_stop_price:
                    stop_triggered = True
                    trigger_reason = "TRAILING_STOP"
                    print(f"   🎯 NOTA: Esta es solo una ALERTA - NO se ejecuta orden automática")
            
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