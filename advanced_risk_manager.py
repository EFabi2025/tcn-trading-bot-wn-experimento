#!/usr/bin/env python3
"""
🛡️ ADVANCED RISK MANAGER
Sistema de gestión de riesgo profesional para trading algorítmico,
ahora impulsado por una configuración centralizada.
"""

import asyncio
import time
import hmac
import hashlib
import math
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import logging

# ✅ NUEVO: Importar la configuración centralizada
from config import trading_config

@dataclass
class Position:
    """📊 Representación de una posición activa"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    pnl_percent: float = 0.0
    pnl_usd: float = 0.0

class AdvancedRiskManager:
    """🛡️ Gestor avanzado de riesgo para trading"""
    
    def __init__(self):
        """Inicializa el gestor de riesgo usando la configuración centralizada."""
        # ✅ USA CONFIGURACIÓN CENTRALIZADA
        self.config = trading_config
        self.logger = logging.getLogger(__name__)
        
        # ❌ ELIMINADO: Balances hardcodeados. Se inicializan a 0 y se esperan desde el manager.
        self.current_balance = 0.0
        self.start_balance = 0.0
        self.peak_balance = 0.0
        
        # Estado del sistema
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.active_positions = {}
        
        # Circuit breaker
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        
        # Estadísticas
        self.stats = {
            'trades': 0,
            'pnl': 0.0,
            'wins': 0,
            'losses': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    async def initialize(self, initial_balance: float):
        """
        🚀 Inicializar el risk manager con el balance real de la cuenta.
        """
        self.logger.info("🛡️ Inicializando Advanced Risk Manager con configuración centralizada...")
        
        # ✅ USA BALANCE REAL
        self.current_balance = initial_balance
        self.start_balance = initial_balance
        self.peak_balance = initial_balance
        
        if self.current_balance < 50.0:
            self.logger.warning("⚠️ Balance inicial de %.2f USDT es bajo para un trading diversificado.", self.current_balance)
        
        self.logger.info(f"💰 Balance inicial para gestión de riesgo: ${self.current_balance:.2f} USDT")
        self._log_risk_parameters()

    def _log_risk_parameters(self):
        """Muestra en los logs los parámetros de riesgo cargados."""
        self.logger.info("⚠️ Límites de riesgo configurados:")
        max_pos_val = self.current_balance * (self.config.MAX_POSITION_SIZE_PERCENT / 100)
        self.logger.info(f"   📊 Max Posición: {self.config.MAX_POSITION_SIZE_PERCENT:.1f}% (${max_pos_val:.2f})")
        self.logger.info(f"   ⚖️ Max Exposición Total: {self.config.MAX_TOTAL_EXPOSURE_PERCENT:.1f}%")
        self.logger.info(f"   🚨 Max Pérdida Diaria: {self.config.MAX_DAILY_LOSS_PERCENT:.1f}%")
        self.logger.info(f"   📉 Stop Loss: {self.config.STOP_LOSS_PERCENT:.1f}%")
        self.logger.info(f"   🎯 Take Profit: {self.config.TAKE_PROFIT_PERCENT:.1f}%")
        self.logger.info(f"   🔢 Max Posiciones Concurrentes: {self.config.MAX_CONCURRENT_POSITIONS}")
        self.logger.info(f"   💵 Mínimo por Trade (Binance): ${self.config.MIN_POSITION_VALUE_USDT} USDT")

    def calculate_position_size(self, symbol: str, confidence: float, price: float) -> float:
        """📊 Calcular tamaño de posición usando configuración centralizada"""
        
        # ✅ Usa config
        base_size_percent = self.config.MAX_POSITION_SIZE_PERCENT
        
        # Ajustar según confianza (puede ser más complejo)
        # Por ahora, usamos un enfoque directo para claridad.
        final_size_percent = base_size_percent
        
        # Limitar al máximo configurado
        final_size_percent = min(final_size_percent, self.config.MAX_POSITION_SIZE_PERCENT)
        
        # Calcular cantidad en USD
        position_value_usd = self.current_balance * (final_size_percent / 100)
        
        # ⚠️ VALIDACIÓN CRÍTICA: Verificar mínimo de Binance
        if position_value_usd < self.config.MIN_POSITION_VALUE_USDT:
            self.logger.warning(f"Posición calculada ${position_value_usd:.2f} es menor al mínimo de Binance ${self.config.MIN_POSITION_VALUE_USDT}")
            
            # Si el balance lo permite, usar el mínimo de Binance
            if self.current_balance >= self.config.MIN_POSITION_VALUE_USDT * 1.2:
                position_value_usd = self.config.MIN_POSITION_VALUE_USDT
                self.logger.info(f"🔧 Ajustando al mínimo de Binance: ${position_value_usd:.2f}")
            else:
                self.logger.error(f"❌ Balance insuficiente para cubrir el mínimo de trade de Binance.")
                return 0.0
        
        quantity = position_value_usd / price
        
        self.logger.info(f"📊 Cálculo de Tamaño para {symbol}: Valor=${position_value_usd:.2f} USD, Cantidad={quantity:.6f}")
        
        return quantity
    
    def set_stop_loss_take_profit(self, position: Position) -> Position:
        """🛑 Configurar Stop Loss y Take Profit desde la configuración"""
        
        if position.side == 'BUY':
            # ✅ Usa config
            position.stop_loss = position.entry_price * (1 - self.config.STOP_LOSS_PERCENT / 100)
            position.take_profit = position.entry_price * (1 + self.config.TAKE_PROFIT_PERCENT / 100)
        
        self.logger.info(f"🛡️ SL/TP para {position.symbol}: SL=${position.stop_loss:.4f}, TP=${position.take_profit:.4f}")
        
        return position

    async def check_risk_limits_before_trade(self, symbol: str, signal: str, confidence: float) -> Tuple[bool, str]:
        """🛡️ Verificar todos los límites de riesgo antes de abrir un nuevo trade"""
        
        # 1. Circuit breaker por pérdida diaria
        if self.circuit_breaker_active:
            remaining_time = (self.circuit_breaker_until - datetime.now()).total_seconds()
            return False, f"🔥 CIRCUIT BREAKER ACTIVO. {remaining_time // 60:.0f} min restantes."
        
        # 2. Verificar que no sea una señal de venta (para Spot)
        # Esta lógica puede ser más compleja si se manejan posiciones para cerrar.
        if signal == 'SELL':
            # Se permite la señal SELL si es para cerrar una posición existente, no para abrir una nueva.
            if symbol not in self.active_positions:
                 return False, "🚫 Venta en corto no permitida en Spot."

        # 3. Verificar número de posiciones concurrentes
        if len(self.active_positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return False, f"🔢 Límite de posiciones concurrentes ({self.config.MAX_CONCURRENT_POSITIONS}) alcanzado."
        
        # 4. Verificar exposición total
        current_exposure_usd = sum(p.quantity * p.current_price for p in self.active_positions.values())
        max_exposure_usd = self.current_balance * (self.config.MAX_TOTAL_EXPOSURE_PERCENT / 100)
        
        # Calcular tamaño potencial de la nueva posición
        potential_pos_value = self.current_balance * (self.config.MAX_POSITION_SIZE_PERCENT / 100)
        potential_total_exposure = current_exposure_usd + potential_pos_value

        if potential_total_exposure > max_exposure_usd:
            return False, f"⚖️ Exposición total ({potential_total_exposure:.2f}) excedería el límite de ${max_exposure_usd:.2f}."

        # 5. TODO: Verificar correlación de activos si se implementa.
        # if await self.check_correlation_risk(symbol):
        #    return False, f"📈 Alta correlación con posiciones existentes."

        return True, "✅ Límites de riesgo pre-trade verificados."

    async def update_balance(self, new_balance: float):
        """Actualiza el balance y recalcula el PnL diario y el drawdown."""
        self.daily_pnl += (new_balance - self.current_balance)
        self.current_balance = new_balance
        
        # Actualizar drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        
        # Verificar circuit breaker por drawdown o pérdida diaria
        daily_loss_percent = (self.daily_pnl / self.start_balance) * 100 if self.start_balance > 0 else 0

        if drawdown > (self.config.MAX_DRAWDOWN_PERCENT / 100):
            await self.activate_circuit_breaker(f"Drawdown {drawdown:.2%} excede el límite de {self.config.MAX_DRAWDOWN_PERCENT}%")

        if daily_loss_percent < -self.config.MAX_DAILY_LOSS_PERCENT:
            await self.activate_circuit_breaker(f"Pérdida diaria {daily_loss_percent:.2f}% excede el límite de -{self.config.MAX_DAILY_LOSS_PERCENT}%")

    async def activate_circuit_breaker(self, reason: str, duration_minutes: int = 60 * 24):
        """Activa el circuit breaker, deteniendo nuevos trades."""
        if not self.circuit_breaker_active:
            self.logger.critical(f"🚨🚨 CIRCUIT BREAKER ACTIVADO: {reason} 🚨🚨")
            self.circuit_breaker_active = True
            self.circuit_breaker_until = datetime.now() + timedelta(minutes=duration_minutes)
            # Aquí podrías añadir una notificación a Discord.
    
    async def open_position(self, symbol: str, signal: str, confidence: float, price: float) -> Optional[Position]:
        """Abre una nueva posición después de verificar el riesgo."""
        can_trade, reason = await self.check_risk_limits_before_trade(symbol, signal, confidence)
        if not can_trade:
            self.logger.warning(f"❌ Trade para {symbol} rechazado por Risk Manager: {reason}")
            return None

        quantity = self.calculate_position_size(symbol, confidence, price)
        if quantity == 0:
            return None
        
        position = Position(
            symbol=symbol,
            side=signal,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now()
        )
        
        position = self.set_stop_loss_take_profit(position)
        
        self.active_positions[symbol] = position
        self.stats['trades'] += 1
        self.logger.info(f"✅ Nueva posición abierta para {symbol}: {quantity:.6f} unidades.")
        return position

    async def close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[Dict]:
        """Cierra una posición y registra el resultado."""
        if symbol not in self.active_positions:
            self.logger.warning(f"Intento de cerrar posición inexistente para {symbol}.")
            return None

        position = self.active_positions.pop(symbol)
        
        pnl_usd = (exit_price - position.entry_price) * position.quantity if position.side == 'BUY' else (position.entry_price - exit_price) * position.quantity
        pnl_percent = (pnl_usd / (position.entry_price * position.quantity)) * 100
        
        # Actualizar balance y PnL
        new_balance = self.current_balance + pnl_usd
        await self.update_balance(new_balance)
        
        self.total_pnl += pnl_usd
        self.stats['pnl'] += pnl_usd
        
        if pnl_usd > 0:
            self.stats['wins'] += 1
            if pnl_usd > self.stats['largest_win']:
                self.stats['largest_win'] = pnl_usd
        else:
            self.stats['losses'] += 1
            if pnl_usd < self.stats['largest_loss']:
                self.stats['largest_loss'] = pnl_usd

        result = {
            "symbol": symbol,
            "pnl_usd": pnl_usd,
            "pnl_percent": pnl_percent,
            "exit_price": exit_price,
            "reason": reason
        }

        self.logger.info(f"Position Closed: {symbol}, PnL: ${pnl_usd:.2f} ({pnl_percent:.2f}%), Reason: {reason}")
        return result

    def get_risk_report(self) -> Dict:
        """Genera un reporte completo del estado de riesgo."""
        win_rate = (self.stats['wins'] / self.stats['trades']) * 100 if self.stats['trades'] > 0 else 0
        
        return {
            "current_balance_usd": self.current_balance,
            "total_pnl_usd": self.total_pnl,
            "daily_pnl_usd": self.daily_pnl,
            "active_positions_count": len(self.active_positions),
            "total_exposure_usd": sum(p.quantity * p.current_price for p in self.active_positions.values()),
            "peak_balance_usd": self.peak_balance,
            "current_drawdown_percent": ((self.peak_balance - self.current_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0,
            "trades_count": self.stats['trades'],
            "win_rate_percent": win_rate,
            "circuit_breaker_active": self.circuit_breaker_active
        } 