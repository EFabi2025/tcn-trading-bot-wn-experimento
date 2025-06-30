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

    def calculate_position_size(self, symbol: str, price: float, confidence: float, risk_adjustment_factor: float = 1.0) -> float:
        """
        📊 Calcular tamaño de posición usando configuración centralizada con límites inteligentes.
        
        Args:
            symbol (str): El símbolo del activo.
            price (float): El precio actual del activo.
            confidence (float): La confianza del modelo en la señal.
            risk_adjustment_factor (float, optional): Factor para ajustar el riesgo. Defaults to 1.0.

        Returns:
            float: La cantidad del activo a comprar.
        """
        # 🎯 NUEVO: Usar cálculo inteligente que considera TODOS los límites
        return self._calculate_smart_position_size(symbol, price, confidence, risk_adjustment_factor)
    
    def _calculate_smart_position_size(self, symbol: str, price: float, confidence: float, risk_adjustment_factor: float = 1.0) -> float:
        """
        🧠 Cálculo INTELIGENTE de tamaño de posición que considera TODOS los límites de riesgo.
        
        Este método calcula el tamaño óptimo considerando:
        1. Límite por posición individual (MAX_POSITION_SIZE_PERCENT)
        2. Límite de exposición total (MAX_TOTAL_EXPOSURE_PERCENT)
        3. Mínimo requerido por Binance (MIN_POSITION_VALUE_USDT)
        4. Balance disponible
        """
        try:
            # 1. CALCULAR TAMAÑO BASE (método original)
            base_size_percent = self.config.MAX_POSITION_SIZE_PERCENT
            confidence_factor = 1 + (confidence - self.config.TCN_BUY_CONFIDENCE_THRESHOLD)
            final_size_percent = min(base_size_percent * confidence_factor, self.config.MAX_POSITION_SIZE_PERCENT)
            
            base_position_value_usd = self.current_balance * (final_size_percent / 100)
            
            # 2. APLICAR FACTOR DE AJUSTE DE RIESGO
            if risk_adjustment_factor != 1.0:
                self.logger.warning(f"🏛️ Aplicando factor de ajuste de riesgo: {risk_adjustment_factor}")
                base_position_value_usd *= risk_adjustment_factor
            
            # 3. 🎯 VERIFICAR LÍMITE DE EXPOSICIÓN TOTAL
            current_exposure_usd = sum(p.quantity * p.current_price for p in self.active_positions.values())
            max_exposure_usd = self.current_balance * (self.config.MAX_TOTAL_EXPOSURE_PERCENT / 100)
            available_exposure = max_exposure_usd - current_exposure_usd
            
            self.logger.info(f"🔍 ANÁLISIS DE EXPOSICIÓN para {symbol}:")
            self.logger.info(f"   💰 Balance actual: ${self.current_balance:.2f}")
            self.logger.info(f"   📊 Exposición actual: ${current_exposure_usd:.2f}")
            self.logger.info(f"   ⚖️ Límite exposición: ${max_exposure_usd:.2f} ({self.config.MAX_TOTAL_EXPOSURE_PERCENT}%)")
            self.logger.info(f"   🆓 Exposición disponible: ${available_exposure:.2f}")
            self.logger.info(f"   🎯 Tamaño base calculado: ${base_position_value_usd:.2f}")
            
            # 4. AJUSTAR TAMAÑO SI EXCEDE EXPOSICIÓN DISPONIBLE
            if available_exposure <= 0:
                self.logger.warning(f"❌ No hay exposición disponible para nuevas posiciones")
                return 0.0
            
            # Usar el menor entre el tamaño base y la exposición disponible
            final_position_value_usd = min(base_position_value_usd, available_exposure)
            
            if final_position_value_usd < base_position_value_usd:
                self.logger.warning(f"⚖️ AJUSTE POR EXPOSICIÓN: Reduciendo de ${base_position_value_usd:.2f} a ${final_position_value_usd:.2f}")
            
            # 5. VERIFICAR MÍNIMO DE BINANCE CON AJUSTE INTELIGENTE
            effective_min_position = self._get_effective_min_position_value()
            
            if final_position_value_usd < effective_min_position:
                self.logger.warning(f"💰 Posición calculada ${final_position_value_usd:.2f} < mínimo efectivo ${effective_min_position:.2f}")
                
                # Si hay suficiente exposición disponible, usar el mínimo efectivo
                if available_exposure >= effective_min_position:
                    final_position_value_usd = effective_min_position
                    self.logger.info(f"🔧 Ajustando al mínimo efectivo: ${final_position_value_usd:.2f}")
                else:
                    # 🚀 LÓGICA MEJORADA: Verificar si el balance es muy pequeño
                    if self.current_balance <= 50:
                        # Para balances muy pequeños, permitir usar el máximo disponible
                        # pero solo si es al menos $8 (para evitar órdenes demasiado pequeñas)
                        if available_exposure >= 8.0:
                            final_position_value_usd = available_exposure
                            self.logger.warning(f"⚠️ BALANCE PEQUEÑO: Usando máximo disponible ${final_position_value_usd:.2f} (< ${effective_min_position:.2f} mínimo)")
                            self.logger.warning(f"💡 RECOMENDACIÓN: Depositar más USDT para trading óptimo")
                        else:
                            self.logger.error(f"❌ Exposición disponible ${available_exposure:.2f} insuficiente para trading seguro")
                            return 0.0
                    else:
                        # Para balances normales, mantener el mínimo estricto
                        self.logger.error(f"❌ Exposición disponible ${available_exposure:.2f} insuficiente para mínimo ${effective_min_position:.2f}")
                        return 0.0
            
            # 6. CALCULAR CANTIDAD FINAL
            quantity = final_position_value_usd / price
            
            self.logger.info(f"✅ CÁLCULO INTELIGENTE COMPLETADO para {symbol}:")
            self.logger.info(f"   💵 Valor final: ${final_position_value_usd:.2f} USD")
            self.logger.info(f"   📊 Cantidad: {quantity:.6f} {symbol.replace('USDT', '')}")
            self.logger.info(f"   💱 Precio: ${price:.2f}")
            self.logger.info(f"   🎯 Confianza: {confidence:.1%}")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"❌ Error en cálculo inteligente de posición: {e}")
            return 0.0

    def _get_effective_min_position_value(self) -> float:
        """🧠 Calcular mínimo efectivo de posición basado en balance disponible
        
        Ajusta automáticamente el MIN_POSITION_VALUE_USDT según el balance actual
        para evitar que órdenes sean rechazadas por configuración restrictiva.
        """
        try:
            # Configuración base - SIEMPRE MÍNIMO $11 para Binance
            config_min = self.config.MIN_POSITION_VALUE_USDT  # $11.0
            
            # Calcular máximo teórico por posición
            max_per_position = self.current_balance * (self.config.MAX_POSITION_SIZE_PERCENT / 100)
            
            # 🎯 LÓGICA MEJORADA: Asegurar que SIEMPRE se cumpla el mínimo de Binance
            if self.current_balance <= 50:  # Balance muy pequeño
                # Para balances muy pequeños, verificar si puede hacer al menos una orden de $11
                if max_per_position >= config_min:
                    self.logger.info(f"✅ Balance pequeño (${self.current_balance:.2f}) pero puede hacer orden mínima de ${config_min:.2f}")
                    return config_min
                else:
                    # Si ni siquiera puede hacer $11, sugerir usar todo el balance disponible
                    # pero con una advertencia clara
                    available_for_trade = self.current_balance * 0.9  # 90% del balance como máximo
                    self.logger.warning(f"⚠️ Balance ${self.current_balance:.2f} insuficiente para mínimo Binance ${config_min:.2f}")
                    self.logger.warning(f"💡 Sugerencia: Depositar al menos ${config_min * 5:.0f} USDT para trading seguro")
                    return available_for_trade
                    
            elif self.current_balance <= 100:  # Balance pequeño pero viable
                # Para balances pequeños viables, usar el mínimo estándar
                self.logger.info(f"✅ Balance pequeño (${self.current_balance:.2f}) - usando mínimo estándar ${config_min:.2f}")
                return config_min
                
            elif self.current_balance <= 500:  # Balance medio
                # Para balances medios, usar el mínimo estándar
                self.logger.info(f"✅ Balance medio (${self.current_balance:.2f}) - usando mínimo estándar ${config_min:.2f}")
                return config_min
                
            else:  # Balance grande
                # Para balances grandes, usar configuración original
                self.logger.info(f"✅ Balance grande (${self.current_balance:.2f}) - usando mínimo estándar ${config_min:.2f}")
                return config_min
            
        except Exception as e:
            self.logger.error(f"❌ Error calculando mínimo efectivo: {e}")
            return self.config.MIN_POSITION_VALUE_USDT  # Fallback a configuración original

    def set_stop_loss_take_profit(self, position: Position) -> Position:
        """🛑 Configurar Stop Loss y Take Profit desde la configuración"""
        
        if position.side == 'BUY':
            # ✅ Usa config
            position.stop_loss = position.entry_price * (1 - self.config.STOP_LOSS_PERCENT / 100)
            position.take_profit = position.entry_price * (1 + self.config.TAKE_PROFIT_PERCENT / 100)
        
        self.logger.info(f"🛡️ SL/TP para {position.symbol}: SL=${position.stop_loss:.4f}, TP=${position.take_profit:.4f}")
        
        return position

    async def check_risk_limits_before_trade(self, symbol: str, signal: str, confidence: float) -> Tuple[bool, str]:
        """🛡️ Verificar límites de riesgo básicos antes de abrir un nuevo trade
        
        NOTA: La verificación de exposición total ahora se hace en calculate_position_size()
        para permitir ajustes automáticos del tamaño.
        """
        
        # 1. Circuit breaker por pérdida diaria
        if self.circuit_breaker_active:
            remaining_time = (self.circuit_breaker_until - datetime.now()).total_seconds()
            return False, f"🔥 CIRCUIT BREAKER ACTIVO. {remaining_time // 60:.0f} min restantes."
        
        # 2. Verificar que no sea una señal de venta (para Spot)
        if signal == 'SELL':
            # Se permite la señal SELL si es para cerrar una posición existente, no para abrir una nueva.
            if symbol not in self.active_positions:
                 return False, "🚫 Venta en corto no permitida en Spot."

        # 3. Verificar número de posiciones concurrentes
        if len(self.active_positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return False, f"🔢 Límite de posiciones concurrentes ({self.config.MAX_CONCURRENT_POSITIONS}) alcanzado."
        
        # 4. ✅ REMOVIDO: Verificación de exposición total (ahora se hace en calculate_position_size)
        # Esto permite que el cálculo de tamaño se ajuste automáticamente a la exposición disponible
        
        # 5. Verificar confianza mínima
        if confidence < self.config.TCN_BUY_CONFIDENCE_THRESHOLD:
            return False, f"🎯 Confianza {confidence:.1%} < {self.config.TCN_BUY_CONFIDENCE_THRESHOLD:.1%} requerida."

        # 6. TODO: Verificar correlación de activos si se implementa.
        # if await self.check_correlation_risk(symbol):
        #    return False, f"📈 Alta correlación con posiciones existentes."

        return True, "✅ Límites de riesgo básicos verificados. Exposición se verificará en cálculo de tamaño."

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
    
    async def open_position(self, symbol: str, side: str, amount: float, price: float, confidence: float, signal_data: dict) -> Optional[Dict]:
        """Abre una nueva posición con una cantidad pre-calculada."""
        can_trade, reason = await self.check_risk_limits_before_trade(symbol, side, confidence)
        if not can_trade:
            self.logger.warning(f"❌ Trade para {symbol} rechazado por Risk Manager: {reason}")
            return {'success': False, 'reason': reason}

        if amount <= 0:
            # 🎯 MEJORADO: Intentar recalcular tamaño automáticamente
            self.logger.warning(f"⚠️ Cantidad inválida ({amount}) para {symbol}. Intentando recálculo automático...")
            
            # Recalcular usando el método inteligente
            recalculated_amount = self.calculate_position_size(symbol, price, confidence, 1.0)
            
            if recalculated_amount <= 0:
                self.logger.error(f"❌ Recálculo automático falló para {symbol}. No se puede abrir posición.")
                return {'success': False, 'reason': 'Cantidad inválida incluso después del recálculo automático'}
            
            amount = recalculated_amount
            self.logger.info(f"✅ Recálculo exitoso: Nueva cantidad {amount:.6f} para {symbol}")
        
        # Verificar valor mínimo final con lógica inteligente
        effective_min_position = self._get_effective_min_position_value()
        position_value_usd = amount * price
        
        if position_value_usd < effective_min_position:
            self.logger.warning(f"❌ Valor final ${position_value_usd:.2f} < mínimo efectivo ${effective_min_position:.2f}")
            return {'success': False, 'reason': f'Valor de posición ${position_value_usd:.2f} menor al mínimo efectivo ${effective_min_position:.2f}'}
        
        # 🎯 LOG INFORMATIVO si se usó ajuste inteligente
        config_min = self.config.MIN_POSITION_VALUE_USDT
        if effective_min_position < config_min:
            self.logger.info(f"✅ AJUSTE INTELIGENTE APLICADO: Usando mínimo ${effective_min_position:.2f} en lugar de ${config_min:.2f}")
        
        position = Position(
            symbol=symbol,
            side=side,
            quantity=amount,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now()
        )
        
        position = self.set_stop_loss_take_profit(position)
        
        self.active_positions[symbol] = position
        self.stats['trades'] += 1
        self.logger.info(f"✅ Nueva posición abierta para {symbol}: {amount:.6f} unidades (${position_value_usd:.2f} USD)")
        return {'success': True, 'position': position}

    async def close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[Dict]:
        """Cierra una posición y registra el resultado."""
        if symbol not in self.active_positions:
            self.logger.warning(f"Intento de cerrar posición inexistente para {symbol}.")
            return {
                'success': False,
                'error': f'Posición no encontrada para {symbol}',
                'symbol': symbol,
                'reason': reason,
                'message': 'La posición ya fue cerrada o nunca existió'
            }

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
            "success": True,
            "symbol": symbol,
            "pnl_usd": pnl_usd,
            "pnl_percent": pnl_percent,
            "exit_price": exit_price,
            "reason": reason,
            "profit_loss": pnl_usd  # Para compatibilidad con código existente
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