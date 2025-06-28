#!/usr/bin/env python3
"""
🎯 ADVANCED TRAILING MONITOR
Sistema avanzado de monitoreo de trailing stops con:
- Verificación directa en Binance sin caches
- Persistencia de estado entre reinicios
- Lógica de protección proporcional inteligente
- Monitor reactivo optimizado
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import trading_config

@dataclass
class Position:
    """📈 Posición con sistema avanzado de trailing stop"""
    symbol: str
    side: str  # BUY o SELL
    size: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl_usd: float
    unrealized_pnl_percent: float
    entry_time: datetime
    duration_minutes: int
    order_id: Optional[str] = None
    
    # ✅ SISTEMA DE TRAILING STOP AVANZADO
    trailing_stop_active: bool = False
    trailing_stop_price: Optional[float] = None
    trailing_stop_percent: float = 2.0  # Default 2%
    highest_price_since_entry: Optional[float] = None
    lowest_price_since_entry: Optional[float] = None
    trailing_activation_threshold: float = 1.0  # Activar en +1%
    last_trailing_update: Optional[datetime] = None
    trailing_movements: int = 0
    
    # Stop Loss y Take Profit tradicionales
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_percent: float = 3.0
    take_profit_percent: float = None  # Se asigna desde config centralizada

    def __post_init__(self):
        """✅ CENTRALIZADO: Asignar take profit desde configuración"""
        if self.take_profit_percent is None:
            self.take_profit_percent = trading_config.TAKE_PROFIT_PERCENT

class AdvancedTrailingMonitor:
    """🎯 Monitor avanzado de trailing stops con verificación directa"""
    
    def __init__(self, portfolio_manager, risk_manager, logger):
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.logger = logger
        
        # Estado del monitor
        self.running = False
        self.active_positions: List[Position] = []
        
        # ✅ CACHE PERSISTENTE para trailing stops
        self.trailing_cache = {}
        self.cache_file = 'trailing_cache.json'
        self._load_trailing_cache()
        
        # Configuración optimizada
        self.base_sleep_time = 15  # 15 segundos con posiciones
        self.empty_sleep_time = 45  # 45 segundos sin posiciones
        self.consecutive_empty_snapshots = 0
        self.max_empty_snapshots = 2

    def _load_trailing_cache(self):
        """🔄 Cargar cache de trailing stops desde archivo"""
        try:
            with open(self.cache_file, 'r') as f:
                self.trailing_cache = json.load(f)
            self.logger.info(f"✅ Cache de trailing stops cargado: {len(self.trailing_cache)} posiciones")
        except FileNotFoundError:
            self.trailing_cache = {}
            self.logger.info("📂 Archivo de cache no encontrado - iniciando cache vacío")
        except Exception as e:
            self.logger.error(f"❌ Error cargando cache: {e}")
            self.trailing_cache = {}

    def _save_trailing_state(self, position: Position):
        """💾 Guardar estado del trailing stop con persistencia"""
        if position.order_id:
            self.trailing_cache[position.order_id] = {
                'trailing_stop_active': position.trailing_stop_active,
                'trailing_stop_price': position.trailing_stop_price,
                'highest_price_since_entry': position.highest_price_since_entry,
                'lowest_price_since_entry': position.lowest_price_since_entry,
                'trailing_movements': position.trailing_movements,
                'last_trailing_update': position.last_trailing_update.isoformat() if position.last_trailing_update else None,
                'symbol': position.symbol,
                'entry_price': position.entry_price
            }
            
            # Guardar en archivo JSON de forma segura
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.trailing_cache, f, indent=2)
            except Exception as e:
                self.logger.error(f"❌ Error guardando cache: {e}")

    def update_trailing_stop_professional(self, position: Position, current_price: float) -> Tuple[Position, bool, str]:
        """📈 Sistema profesional de Trailing Stop con protección proporcional"""
        
        stop_triggered = False
        trigger_reason = ""
        
        # VALIDACIÓN: Precio válido
        if current_price <= 0:
            return position, False, ""
        
        if position.side == 'BUY':
            # --- LÓGICA PARA POSICIONES LONG ---
            
            # 1. ACTUALIZAR MÁXIMO HISTÓRICO (SIEMPRE)
            if position.highest_price_since_entry is None or current_price > position.highest_price_since_entry:
                old_max = position.highest_price_since_entry
                position.highest_price_since_entry = current_price
                
                if old_max is not None:
                    gain_from_entry = ((current_price - position.entry_price) / position.entry_price) * 100
                    self.logger.info(f"🏔️ NUEVO MÁXIMO {position.symbol}: ${current_price:.4f} (+{gain_from_entry:.2f}%)")
            
            # 2. CALCULAR PnL ACTUAL
            current_pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # 3. ACTIVAR TRAILING STOP (si alcanza umbral de ganancia)
            if not position.trailing_stop_active and current_pnl_percent >= position.trailing_activation_threshold:
                position.trailing_stop_active = True
                
                # CÁLCULO INTELIGENTE: Protección proporcional
                current_gain = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100
                
                # Proteger 80% de la ganancia actual (mínimo 0.75% para cubrir comisiones)
                if current_gain >= 2.0:
                    min_profit_protection = current_gain * 0.8  # 80% de ganancia
                else:
                    min_profit_protection = 0.75  # Mínimo para cubrir comisiones
                
                # Precio mínimo de protección
                min_trailing_price = position.entry_price * (1 + min_profit_protection / 100)
                
                # Trailing desde máximo histórico
                trailing_from_peak = position.highest_price_since_entry * (1 - position.trailing_stop_percent / 100)
                
                # Usar el MAYOR entre ambos (más conservador)
                position.trailing_stop_price = max(trailing_from_peak, min_trailing_price)
                position.last_trailing_update = datetime.now()
                
                protected_profit = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                
                self.logger.warning(f"📈 TRAILING ACTIVADO {position.symbol}:")
                self.logger.info(f"   🎯 Ganancia actual: +{current_pnl_percent:.2f}%")
                self.logger.info(f"   📈 Trailing Stop: ${position.trailing_stop_price:.4f}")
                self.logger.info(f"   🛡️ Ganancia protegida: +{protected_profit:.2f}%")
            
            # 4. MOVER TRAILING STOP (si ya está activo)
            elif position.trailing_stop_active:
                # Recalcular protección proporcional
                current_gain = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100
                
                if current_gain >= 2.0:
                    min_profit_protection = current_gain * 0.8
                else:
                    min_profit_protection = 0.75
                
                min_trailing_price = position.entry_price * (1 + min_profit_protection / 100)
                trailing_from_peak = position.highest_price_since_entry * (1 - position.trailing_stop_percent / 100)
                new_trailing_price = max(trailing_from_peak, min_trailing_price)
                
                # MOVER solo si el nuevo precio es MÁS ALTO (nunca hacia abajo)
                if new_trailing_price > position.trailing_stop_price:
                    old_price = position.trailing_stop_price
                    position.trailing_stop_price = new_trailing_price
                    position.trailing_movements += 1
                    position.last_trailing_update = datetime.now()
                    
                    protected_profit = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                    
                    self.logger.info(f"📈 TRAILING MOVIDO {position.symbol}:")
                    self.logger.info(f"   🔄 ${old_price:.4f} → ${new_trailing_price:.4f}")
                    self.logger.info(f"   🏔️ Máximo: ${position.highest_price_since_entry:.4f}")
                    self.logger.info(f"   🛡️ Protegiendo: +{protected_profit:.2f}% ganancia")
                    self.logger.info(f"   📊 Movimiento #{position.trailing_movements}")
            
            # 5. VERIFICAR EJECUCIÓN DEL TRAILING STOP
            if position.trailing_stop_active and current_price <= position.trailing_stop_price:
                stop_triggered = True
                trigger_reason = "TRAILING_STOP"
                
                final_pnl = ((position.trailing_stop_price - position.entry_price) / position.entry_price) * 100
                max_profit = ((position.highest_price_since_entry - position.entry_price) / position.entry_price) * 100
                
                self.logger.warning(f"🛑 TRAILING STOP EJECUTADO {position.symbol}:")
                self.logger.info(f"   📉 Precio: ${current_price:.4f} <= Trailing: ${position.trailing_stop_price:.4f}")
                self.logger.info(f"   💰 PnL Final: +{final_pnl:.2f}%")
                self.logger.info(f"   🏔️ Máximo alcanzado: +{max_profit:.2f}%")
                self.logger.info(f"   📈 Movimientos trailing: {position.trailing_movements}")
        
        elif position.side == 'SELL':
            # --- LÓGICA PARA POSICIONES SHORT (invertida) ---
            
            # 1. Actualizar mínimo histórico
            if position.lowest_price_since_entry is None or current_price < position.lowest_price_since_entry:
                position.lowest_price_since_entry = current_price
                gain_from_entry = ((position.entry_price - current_price) / position.entry_price) * 100
                self.logger.info(f"🏔️ NUEVO MÍNIMO {position.symbol}: ${current_price:.4f} (+{gain_from_entry:.2f}%)")
            
            # 2. PnL para short (ganancia cuando precio baja)
            current_pnl_percent = ((position.entry_price - current_price) / position.entry_price) * 100
            
            # 3. Activar trailing
            if not position.trailing_stop_active and current_pnl_percent >= position.trailing_activation_threshold:
                position.trailing_stop_active = True
                new_trailing_price = position.lowest_price_since_entry * (1 + position.trailing_stop_percent / 100)
                position.trailing_stop_price = min(new_trailing_price, position.entry_price)
                position.last_trailing_update = datetime.now()
                
                self.logger.warning(f"📈 TRAILING ACTIVADO SHORT {position.symbol}: ${position.trailing_stop_price:.4f}")
            
            # 4. Mover trailing
            elif position.trailing_stop_active:
                new_trailing_price = position.lowest_price_since_entry * (1 + position.trailing_stop_percent / 100)
                if new_trailing_price < position.trailing_stop_price:
                    old_price = position.trailing_stop_price
                    position.trailing_stop_price = new_trailing_price
                    position.trailing_movements += 1
                    position.last_trailing_update = datetime.now()
                    
                    self.logger.info(f"📈 TRAILING MOVIDO SHORT {position.symbol}: ${old_price:.4f} → ${new_trailing_price:.4f}")
            
            # 5. Verificar ejecución
            if position.trailing_stop_active and current_price >= position.trailing_stop_price:
                stop_triggered = True
                trigger_reason = "TRAILING_STOP"
                
                final_pnl = ((position.entry_price - position.trailing_stop_price) / position.entry_price) * 100
                self.logger.warning(f"🛑 TRAILING STOP EJECUTADO SHORT {position.symbol}: PnL +{final_pnl:.2f}%")
        
        # VERIFICAR STOP LOSS TRADICIONAL (solo si trailing no está activo)
        if not position.trailing_stop_active and position.stop_loss_price:
            if position.side == 'BUY' and current_price <= position.stop_loss_price:
                stop_triggered = True
                trigger_reason = "STOP_LOSS"
                loss_pnl = ((position.stop_loss_price - position.entry_price) / position.entry_price) * 100
                self.logger.warning(f"🛑 STOP LOSS TRADICIONAL {position.symbol}: {loss_pnl:.2f}%")
            elif position.side == 'SELL' and current_price >= position.stop_loss_price:
                stop_triggered = True
                trigger_reason = "STOP_LOSS"
        
        return position, stop_triggered, trigger_reason

    async def _get_real_balance_for_position(self, symbol: str) -> Dict:
        """🎯 Verificación DIRECTA del balance en Binance para máxima precisión"""
        try:
            base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('BTC', '')
            balances = await self.portfolio_manager.get_account_balances()
            
            if balances and base_asset in balances:
                balance_info = balances[base_asset]
                total_balance = balance_info.get('total', 0.0)
                
                return {
                    'has_position': total_balance >= 0.001,
                    'balance': total_balance,
                    'free': balance_info.get('free', 0.0),
                    'locked': balance_info.get('locked', 0.0),
                    'asset': base_asset,
                    'symbol': symbol
                }
            else:
                return {
                    'has_position': False,
                    'balance': 0.0,
                    'free': 0.0,
                    'locked': 0.0,
                    'asset': base_asset,
                    'symbol': symbol
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error verificando balance real para {symbol}: {e}")
            return {'has_position': False, 'balance': 0.0}

    async def _register_position_by_order_id(self, position: Position):
        """📝 Registrar posición por Order ID en el risk manager"""
        try:
            if position.order_id and position.symbol:
                # Verificar si ya está registrada
                if position.order_id not in self.risk_manager.active_positions:
                    # Crear entrada en risk manager usando order_id como clave
                    self.risk_manager.active_positions[position.order_id] = {
                        'symbol': position.symbol,
                        'side': position.side,
                        'size': position.size,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'order_id': position.order_id,
                        'registered_at': datetime.now()
                    }
                    self.logger.info(f"📝 Posición registrada por Order ID: {position.order_id} ({position.symbol})")
                else:
                    # Actualizar precio actual
                    self.risk_manager.active_positions[position.order_id]['current_price'] = position.current_price
                    
        except Exception as e:
            self.logger.error(f"❌ Error registrando posición por Order ID {position.order_id}: {e}")

    async def _verify_position_exists_by_order_id(self, position: Position) -> bool:
        """🔍 Verificar si una posición específica existe por Order ID"""
        try:
            if not position.order_id:
                self.logger.warning(f"⚠️ Posición {position.symbol} sin Order ID - no se puede verificar")
                return False
            
            # 1. Verificar en risk manager
            if position.order_id not in self.risk_manager.active_positions:
                self.logger.debug(f"❌ Order ID {position.order_id} no encontrado en risk manager")
                return False
            
            # 2. Verificar balance real en Binance
            balance_check = await self._get_real_balance_for_position(position.symbol)
            if not balance_check['has_position']:
                self.logger.debug(f"❌ Balance real para {position.symbol} no confirmado")
                return False
            
            # 3. Verificar que el balance sea suficiente para la posición
            required_balance = position.size
            actual_balance = balance_check['balance']
            
            if actual_balance < required_balance * 0.95:  # 5% de tolerancia
                self.logger.warning(f"⚠️ Balance insuficiente para {position.symbol}: {actual_balance:.6f} < {required_balance:.6f}")
                return False
            
            self.logger.debug(f"✅ Posición {position.order_id} ({position.symbol}) verificada exitosamente")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verificando posición por Order ID {position.order_id}: {e}")
            return False

    async def _cleanup_position_by_order_id(self, position: Position):
        """🧹 Limpiar posición específica por Order ID"""
        try:
            cleaned_items = []
            
            # 1. Eliminar del risk manager por Order ID
            if position.order_id and position.order_id in self.risk_manager.active_positions:
                del self.risk_manager.active_positions[position.order_id]
                cleaned_items.append(f"Risk Manager (Order ID: {position.order_id})")
            
            # 2. Eliminar del cache de trailing stops
            if position.order_id and position.order_id in self.trailing_cache:
                del self.trailing_cache[position.order_id]
                self._save_trailing_cache()
                cleaned_items.append(f"Trailing Cache (Order ID: {position.order_id})")
            
            # 3. También limpiar por símbolo (compatibilidad)
            if position.symbol in self.risk_manager.active_positions:
                del self.risk_manager.active_positions[position.symbol]
                cleaned_items.append(f"Risk Manager (Symbol: {position.symbol})")
            
            # 4. Invalidar cache del portfolio manager
            if hasattr(self.portfolio_manager, 'last_snapshot_time'):
                self.portfolio_manager.last_snapshot_time = None
                cleaned_items.append("Portfolio Manager Cache")
            
            if cleaned_items:
                self.logger.info(f"🧹 Limpieza completada para {position.symbol}: {', '.join(cleaned_items)}")
            else:
                self.logger.debug(f"🧹 No había elementos para limpiar para {position.symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error limpiando posición {position.order_id}: {e}")

    def _save_trailing_cache(self):
        """💾 Guardar cache completo de trailing stops"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.trailing_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"❌ Error guardando cache completo: {e}")

    def _restore_trailing_state(self, position: Position) -> Position:
        """🔄 Restaurar estado del trailing stop desde cache por Order ID"""
        if position.order_id and position.order_id in self.trailing_cache:
            cached_state = self.trailing_cache[position.order_id]
            
            # Verificar que el cache corresponde a la misma posición
            if cached_state.get('symbol') == position.symbol and cached_state.get('entry_price') == position.entry_price:
                position.trailing_stop_active = cached_state.get('trailing_stop_active', False)
                position.trailing_stop_price = cached_state.get('trailing_stop_price', None)
                position.highest_price_since_entry = cached_state.get('highest_price_since_entry', position.entry_price)
                position.lowest_price_since_entry = cached_state.get('lowest_price_since_entry', position.entry_price)
                position.trailing_movements = cached_state.get('trailing_movements', 0)
                
                # Restaurar fecha si existe
                last_update_str = cached_state.get('last_trailing_update')
                if last_update_str:
                    try:
                        position.last_trailing_update = datetime.fromisoformat(last_update_str)
                    except:
                        position.last_trailing_update = None
                
                self.logger.info(f"🔄 TRAILING RESTAURADO {position.symbol} (Order ID: {position.order_id}) - Movimientos: {position.trailing_movements}")
            else:
                self.logger.warning(f"⚠️ Cache inconsistente para Order ID {position.order_id} - ignorando")
        
        return position

    async def _execute_stop_loss_by_order_id(self, position: Position, trigger_reason: str):
        """🛑 Ejecutar stop loss para posición específica por Order ID"""
        try:
            self.logger.warning(f"🚨 EJECUTANDO {trigger_reason} para Order ID {position.order_id}: {position.symbol}")
            
            # Calcular PnL final
            if trigger_reason == "TRAILING_STOP" and position.trailing_stop_price:
                exit_price = position.trailing_stop_price
            else:
                exit_price = position.current_price
                
            pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100
            pnl_usd = (exit_price - position.entry_price) * position.size
            
            self.logger.info(f"   📊 Order ID: {position.order_id}")
            self.logger.info(f"   📈 PnL Final: {pnl_percent:+.2f}% (${pnl_usd:+.2f})")
            self.logger.info(f"   💰 Tamaño: {position.size} {position.symbol.replace('USDT', '')}")
            
            # Verificar una vez más que la posición existe
            if not await self._verify_position_exists_by_order_id(position):
                self.logger.warning(f"⚠️ Posición {position.order_id} ya no existe - omitiendo ejecución")
                await self._cleanup_position_by_order_id(position)
                return
            
            # Ejecutar orden usando el risk manager
            sell_result = await self.risk_manager.close_position(
                symbol=position.symbol,
                exit_price=exit_price,
                reason=f"AUTO_{trigger_reason}_ORDER_{position.order_id}"
            )
            
            if sell_result and sell_result.get('success'):
                self.logger.info(f"✅ {trigger_reason} EJECUTADO EXITOSAMENTE para Order ID {position.order_id}")
                
                # Limpiar todas las referencias a esta posición
                await self._cleanup_position_by_order_id(position)
                
            else:
                error_msg = "Desconocido"
                if sell_result and isinstance(sell_result, dict):
                    error_msg = sell_result.get('error', 'Resultado sin detalles de error')
                    
                    # Si la posición no se encontró, limpiar automáticamente
                    if 'no encontrada' in error_msg.lower() or 'not found' in error_msg.lower():
                        self.logger.info(f"🧹 Order ID {position.order_id} no encontrado en Binance - limpiando automáticamente")
                        await self._cleanup_position_by_order_id(position)
                        
                elif sell_result is None:
                    error_msg = "Función close_position retornó None"
                    await self._cleanup_position_by_order_id(position)
                
                self.logger.error(f"❌ FALLO EN {trigger_reason} para Order ID {position.order_id}: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"❌ Error crítico ejecutando {trigger_reason} para Order ID {position.order_id}: {e}")

    async def monitor_positions(self):
        """🎯 Monitor avanzado con reconocimiento por Order ID"""
        
        self.logger.info("🎯 Monitor avanzado de trailing stops iniciado (Sistema Order ID)")
        self.running = True
        
        while self.running:
            try:
                # Obtener snapshot de posiciones actuales
                snapshot = await self.portfolio_manager.get_portfolio_snapshot()
                
                if snapshot and snapshot.active_positions:
                    self.consecutive_empty_snapshots = 0
                    self.logger.debug(f"📊 Monitoreando {len(snapshot.active_positions)} posiciones activas")
                    
                    for position in snapshot.active_positions:
                        try:
                            # 🔑 PASO 1: Registrar posición por Order ID
                            await self._register_position_by_order_id(position)
                            
                            # 🔍 PASO 2: Verificar existencia por Order ID
                            if not await self._verify_position_exists_by_order_id(position):
                                self.logger.debug(f"⚠️ Posición Order ID {position.order_id} no verificada - omitiendo")
                                await self._cleanup_position_by_order_id(position)
                                continue
                            
                            # 🔄 PASO 3: Restaurar estado del trailing stop
                            position = self._restore_trailing_state(position)
                            
                            # 📈 PASO 4: Obtener precio actual
                            current_price = await self.portfolio_manager.get_current_price(position.symbol)
                            
                            if current_price:
                                position.current_price = current_price
                                
                                # 🎯 PASO 5: Actualizar trailing stop con lógica avanzada
                                updated_pos, stop_triggered, trigger_reason = self.update_trailing_stop_professional(
                                    position, current_price
                                )
                                
                                # 💾 PASO 6: Guardar estado actualizado
                                self._save_trailing_state(updated_pos)
                                
                                # 🚨 PASO 7: EJECUCIÓN cuando se activa stop
                                if stop_triggered:
                                    self.logger.warning(f"🚨 STOP ACTIVADO para Order ID {position.order_id}: {position.symbol} - {trigger_reason}")
                                    
                                    # ✅ VERIFICACIÓN FINAL por Order ID
                                    if await self._verify_position_exists_by_order_id(updated_pos):
                                        self.logger.info(f"✅ Ejecutando {trigger_reason} para Order ID {position.order_id}")
                                        await self._execute_stop_loss_by_order_id(updated_pos, trigger_reason)
                                    else:
                                        self.logger.warning(f"⚠️ Order ID {position.order_id} desapareció antes del stop - omitiendo")
                                        await self._cleanup_position_by_order_id(updated_pos)
                                
                                # 🎯 PASO 8: Verificar take profit tradicional
                                elif not updated_pos.trailing_stop_active and updated_pos.take_profit_price:
                                    take_profit_triggered = False
                                    
                                    if position.side == 'BUY' and current_price >= updated_pos.take_profit_price:
                                        take_profit_triggered = True
                                    elif position.side == 'SELL' and current_price <= updated_pos.take_profit_price:
                                        take_profit_triggered = True
                                    
                                    if take_profit_triggered:
                                        self.logger.info(f"🎯 TAKE PROFIT activado para Order ID {position.order_id}")
                                        if await self._verify_position_exists_by_order_id(updated_pos):
                                            await self._execute_stop_loss_by_order_id(updated_pos, "TAKE_PROFIT")
                                        else:
                                            await self._cleanup_position_by_order_id(updated_pos)
                                    
                        except Exception as e:
                            self.logger.error(f"❌ Error monitoreando Order ID {position.order_id if position.order_id else 'N/A'}: {e}")
                else:
                    self.consecutive_empty_snapshots += 1
                    if self.consecutive_empty_snapshots >= self.max_empty_snapshots:
                        self.logger.debug("🔍 No hay posiciones activas - reduciendo frecuencia temporalmente")
                
                # ⚡ TIMING OPTIMIZADO para trailing stop reactivo
                if self.consecutive_empty_snapshots >= self.max_empty_snapshots:
                    sleep_time = self.empty_sleep_time  # 45 segundos sin posiciones
                else:
                    sleep_time = self.base_sleep_time  # 15 segundos con posiciones activas
                    
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                self.logger.info("🛑 Monitor de trailing stops detenido.")
                break
            except Exception as e:
                self.logger.error(f"💥 Error en el monitor de trailing stops: {e}")
                await asyncio.sleep(30)

    def get_positions_report(self) -> str:
        """📊 Generar reporte de posiciones registradas por Order ID"""
        try:
            if not self.risk_manager.active_positions:
                return "📭 No hay posiciones registradas por Order ID"
            
            report = "📊 **POSICIONES REGISTRADAS POR ORDER ID**\n"
            
            for order_id, pos_data in self.risk_manager.active_positions.items():
                if isinstance(pos_data, dict) and 'symbol' in pos_data:
                    symbol = pos_data.get('symbol', 'Unknown')
                    size = pos_data.get('size', 0)
                    entry_price = pos_data.get('entry_price', 0)
                    current_price = pos_data.get('current_price', 0)
                    
                    pnl_percent = 0
                    if entry_price > 0:
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    
                    report += f"\n🎯 **{symbol} (Order ID: {order_id})**\n"
                    report += f"├─ Tamaño: {size}\n"
                    report += f"├─ Entrada: ${entry_price:.4f}\n"
                    report += f"├─ Actual: ${current_price:.4f}\n"
                    report += f"└─ PnL: {pnl_percent:+.2f}%\n"
            
            return f"{report}\n📊 **Total registradas: {len(self.risk_manager.active_positions)}**"
            
        except Exception as e:
            return f"❌ Error generando reporte: {e}"

    def stop_monitoring(self):
        """🛑 Detener el monitor"""
        self.logger.info("🛑 Deteniendo monitor de trailing stops...")
        self.running = False
    
    async def register_position_by_order_id(self, order_id: str, symbol: str, side: str, quantity: float, entry_price: float, confidence: float = 1.0):
        """📝 Registrar nueva posición por Order ID para monitoreo avanzado"""
        try:
            # Crear posición con datos iniciales
            position = Position(
                symbol=symbol,
                side=side,
                size=quantity,
                entry_price=entry_price,
                current_price=entry_price,  # Inicialmente igual al precio de entrada
                market_value=quantity * entry_price,
                unrealized_pnl_usd=0.0,
                unrealized_pnl_percent=0.0,
                entry_time=datetime.now(),
                duration_minutes=0,
                order_id=order_id
            )
            
            # Configurar trailing stops iniciales
            position.highest_price_since_entry = entry_price
            position.lowest_price_since_entry = entry_price
            
            # Agregar a lista de posiciones activas
            self.active_positions.append(position)
            
            # Registrar en cache persistente
            await self._register_position_by_order_id(position)
            
            self.logger.info(f"📝 Posición registrada por Order ID: {order_id} ({symbol})")
            self.logger.info(f"   💰 Cantidad: {quantity} @ ${entry_price:.4f}")
            self.logger.info(f"   🎯 Confianza: {confidence:.1%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error registrando posición por Order ID {order_id}: {e}")
            return False


# === EJEMPLO DE USO ===

async def test_advanced_monitor():
    """🧪 Prueba del monitor avanzado"""
    from simple_professional_manager import TradingManager
    
    print("🎯 PROBANDO MONITOR AVANZADO DE TRAILING STOPS")
    print("=" * 60)
    
    manager = TradingManager()
    await manager.initialize()
    
    # Crear monitor avanzado
    monitor = AdvancedTrailingMonitor(
        portfolio_manager=manager.portfolio_manager,
        risk_manager=manager.risk_manager,
        logger=manager.logger
    )
    
    print("✅ Monitor avanzado inicializado")
    print(f"📊 Cache actual: {len(monitor.trailing_cache)} posiciones")
    
    await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(test_advanced_monitor())