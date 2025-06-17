#!/usr/bin/env python3
"""
🔍 MONITOR DE TRADING EN TIEMPO REAL
Monitor para rastrear señales, filtros y ejecución de órdenes
"""

import asyncio
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TradingMonitor:
    """📡 Monitor de trading en tiempo real"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.signals_detected = 0
        self.signals_executed = 0
        self.signals_rejected = 0
        self.rejection_reasons = {}
        
        print("📡 Trading Monitor iniciado")
        print("🎯 Monitoreando señales y ejecución de órdenes...")
        print("=" * 60)
    
    async def monitor_signals(self):
        """📊 Monitorear señales del sistema"""
        
        # Simular monitoreo de señales (en un sistema real, esto se integraría con el trading manager)
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        while True:
            try:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Verificando señales...")
                
                for symbol in symbols:
                    # Simular detección de señal (esto se reemplazaría con la lógica real)
                    signal_confidence = 0.75  # Ejemplo
                    
                    if signal_confidence > 0.70:
                        self.signals_detected += 1
                        print(f"🎯 SEÑAL DETECTADA: BUY {symbol} - Confianza: {signal_confidence:.1%}")
                        
                        # Verificar filtros
                        can_execute, reason = await self._check_execution_filters(symbol, signal_confidence)
                        
                        if can_execute:
                            self.signals_executed += 1
                            print(f"✅ SEÑAL EJECUTADA: {symbol}")
                        else:
                            self.signals_rejected += 1
                            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
                            print(f"❌ SEÑAL RECHAZADA: {symbol} - Razón: {reason}")
                
                # Mostrar estadísticas cada 5 ciclos
                if self.signals_detected > 0 and self.signals_detected % 5 == 0:
                    await self._show_statistics()
                
                await asyncio.sleep(30)  # Verificar cada 30 segundos
                
            except Exception as e:
                print(f"❌ Error en monitor: {e}")
                await asyncio.sleep(10)
    
    async def _check_execution_filters(self, symbol: str, confidence: float) -> tuple:
        """🛡️ Verificar filtros de ejecución"""
        
        # 1. Verificar confianza mínima
        min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))
        if confidence < min_confidence:
            return False, f"Confianza insuficiente: {confidence:.1%} < {min_confidence:.1%}"
        
        # 2. Verificar balance disponible
        # En un sistema real, esto se obtendría de Binance API
        balance = 117.94  # Ejemplo del diagnóstico
        min_trade = float(os.getenv('MIN_TRADE_VALUE_USDT', '11'))
        
        if balance < min_trade:
            return False, f"Balance insuficiente: ${balance:.2f} < ${min_trade}"
        
        # 3. Verificar posiciones máximas
        max_positions = int(os.getenv('MAX_SIMULTANEOUS_POSITIONS', '2'))
        current_positions = 0  # En un sistema real, esto se obtendría del risk manager
        
        if current_positions >= max_positions:
            return False, f"Posiciones máximas alcanzadas: {current_positions}/{max_positions}"
        
        # 4. Verificar pérdida diaria
        max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '5'))
        current_daily_loss = 0  # En un sistema real, esto se calcularía
        
        if current_daily_loss >= max_daily_loss:
            return False, f"Pérdida diaria máxima alcanzada: {current_daily_loss:.1f}%"
        
        # 5. Verificar modo trading
        dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        trade_mode = os.getenv('TRADE_MODE', 'dry_run')
        
        if dry_run:
            return False, "DRY_RUN=true - Solo simulación"
        
        if trade_mode == 'dry_run':
            return False, "TRADE_MODE=dry_run - Solo simulación"
        
        return True, "Todos los filtros pasados"
    
    async def _show_statistics(self):
        """📊 Mostrar estadísticas del monitor"""
        
        uptime = datetime.now() - self.start_time
        execution_rate = (self.signals_executed / self.signals_detected * 100) if self.signals_detected > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS DEL MONITOR")
        print(f"⏱️ Tiempo activo: {uptime}")
        print(f"🎯 Señales detectadas: {self.signals_detected}")
        print(f"✅ Señales ejecutadas: {self.signals_executed}")
        print(f"❌ Señales rechazadas: {self.signals_rejected}")
        print(f"📈 Tasa de ejecución: {execution_rate:.1f}%")
        
        if self.rejection_reasons:
            print("\n🚫 RAZONES DE RECHAZO:")
            for reason, count in self.rejection_reasons.items():
                print(f"   • {reason}: {count} veces")
        
        print("=" * 60)
    
    async def run(self):
        """🚀 Ejecutar monitor"""
        try:
            await self.monitor_signals()
        except KeyboardInterrupt:
            print("\n\n🛑 Monitor detenido por usuario")
            await self._show_statistics()

async def main():
    """🎯 Función principal"""
    monitor = TradingMonitor()
    await monitor.run()

if __name__ == "__main__":
    asyncio.run(main())
