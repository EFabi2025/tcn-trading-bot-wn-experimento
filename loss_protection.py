# 🛡️ SISTEMA DE PROTECCIÓN POST-PÉRDIDAS
# Implementación para prevenir revenge trading después de posiciones cerradas con pérdidas

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class LossProtectionConfig:
    """⚙️ Configuración del sistema de protección post-pérdidas"""

    # Tiempos de espera según magnitud de la pérdida
    small_loss_cooldown_minutes: int = 15      # Pérdida < 2%: 15 min
    medium_loss_cooldown_minutes: int = 30     # Pérdida 2-5%: 30 min
    large_loss_cooldown_minutes: int = 60      # Pérdida > 5%: 60 min

    # Umbrales de pérdida
    small_loss_threshold: float = 2.0          # 2%
    large_loss_threshold: float = 5.0          # 5%

    # Protección por múltiples pérdidas consecutivas
    consecutive_losses_penalty_minutes: int = 90  # 90 min adicionales
    max_consecutive_losses: int = 3                # Máximo 3 pérdidas seguidas

    # Protección por pérdida diaria acumulada
    daily_loss_threshold: float = 10.0        # 10% pérdida diaria
    daily_loss_penalty_minutes: int = 120     # 2 horas de pausa

    # Protección específica por símbolo
    symbol_specific_cooldown: bool = True     # Cooldown por símbolo vs global

    # Bypass para señales de muy alta confianza
    bypass_confidence_threshold: float = 95.0  # 95% confianza puede bypass
    bypass_enabled: bool = True

class LossProtectionManager:
    """🛡️ Manager para prevenir trading después de pérdidas"""

    def __init__(self, config: Optional[LossProtectionConfig] = None):
        self.config = config or LossProtectionConfig()

        # Tracking por símbolo
        self.symbol_loss_history: Dict[str, List[Dict]] = {}

        # Tracking global
        self.global_loss_history: List[Dict] = []

        # Estados de cooldown activos
        self.active_cooldowns: Dict[str, datetime] = {}  # {symbol: end_time}
        self.global_cooldown_end: Optional[datetime] = None

        # Contador de pérdidas consecutivas
        self.consecutive_losses_count: int = 0
        self.last_profitable_trade_time: Optional[datetime] = None

    def register_position_close(self, symbol: str, pnl_percent: float, pnl_usd: float,
                              close_reason: str, entry_time: datetime) -> None:
        """📝 Registrar cierre de posición y aplicar protecciones si hay pérdida"""

        close_data = {
            'symbol': symbol,
            'pnl_percent': pnl_percent,
            'pnl_usd': pnl_usd,
            'close_time': datetime.now(),
            'close_reason': close_reason,
            'entry_time': entry_time,
            'duration_minutes': (datetime.now() - entry_time).total_seconds() / 60
        }

        # Registrar en historial
        if symbol not in self.symbol_loss_history:
            self.symbol_loss_history[symbol] = []

        self.symbol_loss_history[symbol].append(close_data)
        self.global_loss_history.append(close_data)

        # Limpiar historial antiguo (> 24 horas)
        self._cleanup_old_history()

        # Si es pérdida, aplicar protecciones
        if pnl_percent < 0:
            self._apply_loss_protection(symbol, abs(pnl_percent), close_data)
            self.consecutive_losses_count += 1
        else:
            # Reset contador de pérdidas consecutivas
            self.consecutive_losses_count = 0
            self.last_profitable_trade_time = datetime.now()

    def _apply_loss_protection(self, symbol: str, loss_percent: float, close_data: Dict) -> None:
        """🛡️ Aplicar protección específica según magnitud de la pérdida"""

        now = datetime.now()
        cooldown_minutes = 0
        protection_reasons = []
        apply_global_cooldown = False

        # 1. Protección base según magnitud de pérdida
        if loss_percent >= self.config.large_loss_threshold:
            cooldown_minutes = self.config.large_loss_cooldown_minutes
            protection_reasons.append(f"LARGE_LOSS_{loss_percent:.1f}%")
        elif loss_percent >= self.config.small_loss_threshold:
            cooldown_minutes = self.config.medium_loss_cooldown_minutes
            protection_reasons.append(f"MEDIUM_LOSS_{loss_percent:.1f}%")
        else:
            cooldown_minutes = self.config.small_loss_cooldown_minutes
            protection_reasons.append(f"SMALL_LOSS_{loss_percent:.1f}%")

        # 2. Penalidad por pérdidas consecutivas
        if self.consecutive_losses_count >= self.config.max_consecutive_losses:
            cooldown_minutes += self.config.consecutive_losses_penalty_minutes
            protection_reasons.append(f"CONSECUTIVE_LOSSES_{self.consecutive_losses_count}")

        # 3. Protección por pérdida diaria acumulada
        daily_loss = self._calculate_daily_loss_percent()
        if daily_loss >= self.config.daily_loss_threshold:
            # ✅ CORRECCIÓN: Solo aplicar cooldown global si supera significativamente el umbral
            if daily_loss >= self.config.daily_loss_threshold * 1.5:  # 15% para testing (1.5 * 8% = 12%)
                cooldown_minutes = max(cooldown_minutes, self.config.daily_loss_penalty_minutes)
                protection_reasons.append(f"DAILY_LOSS_{daily_loss:.1f}%")
                apply_global_cooldown = True
            else:
                # Solo advertencia, no cooldown global aún
                protection_reasons.append(f"DAILY_LOSS_WARNING_{daily_loss:.1f}%")

        # 4. Aplicar cooldown
        if apply_global_cooldown:
            # Cooldown global por pérdida diaria excesiva
            self.global_cooldown_end = now + timedelta(minutes=cooldown_minutes)
            print(f"🌐 COOLDOWN GLOBAL activado por pérdida diaria extrema: {daily_loss:.1f}%")
        elif self.config.symbol_specific_cooldown:
            # Cooldown específico por símbolo (comportamiento normal)
            self.active_cooldowns[symbol] = now + timedelta(minutes=cooldown_minutes)
        else:
            # Cooldown global configurado
            self.global_cooldown_end = now + timedelta(minutes=cooldown_minutes)

        # Log de protección aplicada
        protection_summary = " + ".join(protection_reasons)
        cooldown_type = "GLOBAL" if apply_global_cooldown else "SYMBOL" if self.config.symbol_specific_cooldown else "GLOBAL"
        print(f"🛡️ PROTECCIÓN POST-PÉRDIDA aplicada:")
        print(f"   📊 {symbol}: -{loss_percent:.1f}% → {cooldown_minutes}min cooldown ({cooldown_type})")
        print(f"   🏷️ Razones: {protection_summary}")
        print(f"   ⏰ Fin de cooldown: {(now + timedelta(minutes=cooldown_minutes)).strftime('%H:%M:%S')}")

    def can_open_position(self, symbol: str, signal_confidence: float = 0.0) -> Tuple[bool, str]:
        """✅ Verificar si se puede abrir una nueva posición"""

        now = datetime.now()

        # 1. Verificar cooldown global
        if self.global_cooldown_end and now < self.global_cooldown_end:
            remaining_minutes = (self.global_cooldown_end - now).total_seconds() / 60

            # Bypass para señales de muy alta confianza
            if (self.config.bypass_enabled and
                signal_confidence >= self.config.bypass_confidence_threshold):
                print(f"🚨 BYPASS ACTIVADO: Confianza {signal_confidence:.1f}% >= {self.config.bypass_confidence_threshold:.1f}%")
                return True, f"BYPASS_HIGH_CONFIDENCE_{signal_confidence:.1f}%"

            return False, f"GLOBAL_COOLDOWN_{remaining_minutes:.1f}min_remaining"

        # 2. Verificar cooldown específico del símbolo
        if symbol in self.active_cooldowns:
            cooldown_end = self.active_cooldowns[symbol]
            if now < cooldown_end:
                remaining_minutes = (cooldown_end - now).total_seconds() / 60

                # Bypass para señales de muy alta confianza
                if (self.config.bypass_enabled and
                    signal_confidence >= self.config.bypass_confidence_threshold):
                    print(f"🚨 BYPASS ACTIVADO: Confianza {signal_confidence:.1f}% >= {self.config.bypass_confidence_threshold:.1f}%")
                    return True, f"BYPASS_HIGH_CONFIDENCE_{signal_confidence:.1f}%"

                return False, f"SYMBOL_COOLDOWN_{remaining_minutes:.1f}min_remaining"
            else:
                # Limpiar cooldown expirado
                del self.active_cooldowns[symbol]

        return True, "NO_PROTECTION_ACTIVE"

    def _calculate_daily_loss_percent(self) -> float:
        """📊 Calcular pérdida acumulada en las últimas 24 horas"""

        now = datetime.now()
        daily_cutoff = now - timedelta(hours=24)

        daily_losses = [
            trade for trade in self.global_loss_history
            if trade['close_time'] >= daily_cutoff and trade['pnl_percent'] < 0
        ]

        if not daily_losses:
            return 0.0

        total_loss_percent = sum(abs(trade['pnl_percent']) for trade in daily_losses)
        return total_loss_percent

    def _cleanup_old_history(self) -> None:
        """🧹 Limpiar historial antiguo (>24 horas)"""

        cutoff = datetime.now() - timedelta(hours=24)

        # Limpiar historial global
        self.global_loss_history = [
            trade for trade in self.global_loss_history
            if trade['close_time'] >= cutoff
        ]

        # Limpiar historial por símbolo
        for symbol in self.symbol_loss_history:
            self.symbol_loss_history[symbol] = [
                trade for trade in self.symbol_loss_history[symbol]
                if trade['close_time'] >= cutoff
            ]

    def get_protection_status(self) -> Dict:
        """📊 Obtener estado actual de las protecciones"""

        now = datetime.now()

        # Cooldowns activos
        active_symbol_cooldowns = {}
        for symbol, end_time in self.active_cooldowns.items():
            if now < end_time:
                remaining_minutes = (end_time - now).total_seconds() / 60
                active_symbol_cooldowns[symbol] = {
                    'end_time': end_time,
                    'remaining_minutes': remaining_minutes
                }

        global_cooldown_info = None
        if self.global_cooldown_end and now < self.global_cooldown_end:
            remaining_minutes = (self.global_cooldown_end - now).total_seconds() / 60
            global_cooldown_info = {
                'end_time': self.global_cooldown_end,
                'remaining_minutes': remaining_minutes
            }

        # Estadísticas recientes
        daily_loss = self._calculate_daily_loss_percent()
        recent_losses = len([t for t in self.global_loss_history[-10:] if t['pnl_percent'] < 0])

        return {
            'active_symbol_cooldowns': active_symbol_cooldowns,
            'global_cooldown': global_cooldown_info,
            'consecutive_losses_count': self.consecutive_losses_count,
            'daily_loss_percent': daily_loss,
            'recent_losses_count': recent_losses,
            'last_profitable_trade': self.last_profitable_trade_time,
            'bypass_threshold': self.config.bypass_confidence_threshold,
            'protection_thresholds': {
                'small_loss': self.config.small_loss_threshold,
                'large_loss': self.config.large_loss_threshold,
                'daily_loss': self.config.daily_loss_threshold
            }
        }

    def format_protection_report(self) -> str:
        """📋 Generar reporte formateado del estado de protección"""

        status = self.get_protection_status()
        report = "\n🛡️ **PROTECCIÓN POST-PÉRDIDAS:**\n"

        # Cooldowns activos
        if status['active_symbol_cooldowns']:
            report += "🔒 **Cooldowns por símbolo:**\n"
            for symbol, info in status['active_symbol_cooldowns'].items():
                report += f"   {symbol}: {info['remaining_minutes']:.1f}min restantes\n"

        if status['global_cooldown']:
            report += f"🌐 **Cooldown global:** {status['global_cooldown']['remaining_minutes']:.1f}min restantes\n"

        # Estadísticas
        report += f"📊 **Estadísticas:**\n"
        report += f"   🔴 Pérdidas consecutivas: {status['consecutive_losses_count']}/{self.config.max_consecutive_losses}\n"
        report += f"   📉 Pérdida diaria: {status['daily_loss_percent']:.1f}%/{status['protection_thresholds']['daily_loss']:.1f}%\n"
        report += f"   🎯 Bypass activado: {status['bypass_threshold']:.1f}% confianza\n"

        if not status['active_symbol_cooldowns'] and not status['global_cooldown']:
            report += "✅ **No hay protecciones activas**\n"

        return report


# 🔧 INTEGRACIÓN EN SimpleProfessionalTradingManager

def integrate_loss_protection():
    """Código para integrar en el SimpleProfessionalTradingManager"""

    # 1. En __init__()
    """
    # ✅ NUEVO: Sistema de protección post-pérdidas
    self.loss_protection = LossProtectionManager()
    """

    # 2. En _close_position()
    """
    # ✅ NUEVO: Registrar cierre para protección post-pérdidas
    self.loss_protection.register_position_close(
        symbol=symbol,
        pnl_percent=pnl_percent,
        pnl_usd=pnl_usd,
        close_reason=reason,
        entry_time=position.entry_time
    )
    """

    # 3. En _consider_new_position() - ANTES de verificar risk management
    """
    # ✅ NUEVO: Verificar protección post-pérdidas
    can_trade, protection_reason = self.loss_protection.can_open_position(symbol, confidence)
    if not can_trade:
        print(f"    🛡️ BLOQUEADO POR PROTECCIÓN POST-PÉRDIDAS: {protection_reason}")
        await self._send_discord_notification(
            f"🛡️ **PROTECCIÓN POST-PÉRDIDAS**\n"
            f"📊 {symbol}: {signal}\n"
            f"🚫 Razón: {protection_reason}\n"
            f"🎯 Confianza: {confidence:.1f}%"
        )
        return
    """

    # 4. En _generate_tcn_report_if_needed()
    """
    # ✅ NUEVO: Agregar estado de protección post-pérdidas al reporte
    protection_report = self.loss_protection.format_protection_report()
    full_report += protection_report
    """

# 📋 VARIABLES DE ENTORNO RECOMENDADAS
ENV_VARIABLES = """
# Protección Post-Pérdidas
LOSS_PROTECTION_SMALL_LOSS_COOLDOWN=15     # Minutos para pérdida < 2%
LOSS_PROTECTION_MEDIUM_LOSS_COOLDOWN=30    # Minutos para pérdida 2-5%
LOSS_PROTECTION_LARGE_LOSS_COOLDOWN=60     # Minutos para pérdida > 5%
LOSS_PROTECTION_CONSECUTIVE_PENALTY=90     # Minutos adicionales por pérdidas consecutivas
LOSS_PROTECTION_DAILY_LOSS_THRESHOLD=10.0  # % pérdida diaria para activar protección
LOSS_PROTECTION_BYPASS_CONFIDENCE=95.0     # % confianza para bypass
LOSS_PROTECTION_SYMBOL_SPECIFIC=true       # Cooldown por símbolo vs global
"""
