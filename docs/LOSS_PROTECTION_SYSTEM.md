# 🛡️ SISTEMA DE PROTECCIÓN POST-PÉRDIDAS

## Descripción General

El Sistema de Protección Post-Pérdidas es un mecanismo avanzado de gestión de riesgo diseñado para prevenir el "revenge trading" (trading de venganza) después de posiciones cerradas con pérdidas. Este sistema implementa cooldowns automáticos y restricciones temporales para proteger el capital del trader.

## 🎯 Objetivos

1. **Prevenir Revenge Trading**: Evitar decisiones emocionales después de pérdidas
2. **Protección del Capital**: Limitar pérdidas consecutivas y acumuladas
3. **Gestión de Riesgo Adaptativa**: Ajustar restricciones según la magnitud de las pérdidas
4. **Bypass Inteligente**: Permitir excepciones para señales de muy alta confianza

## 🚫 Tipos de Protección

### 1. Protección por Magnitud de Pérdida
- **Pérdida Pequeña (< 2%)**: Cooldown de 15 minutos
- **Pérdida Media (2-5%)**: Cooldown de 30 minutos
- **Pérdida Grande (> 5%)**: Cooldown de 60 minutos

### 2. Protección por Pérdidas Consecutivas
- **Contador**: Rastrea pérdidas seguidas sin operaciones rentables
- **Penalización**: +90 minutos adicionales tras 3 pérdidas consecutivas
- **Reset**: Se reinicia con cualquier operación rentable

### 3. Protección por Pérdida Diaria
- **Umbral**: 10% de pérdida acumulada en 24 horas
- **Penalización**: 2 horas de pausa completa
- **Alcance**: Aplica globalmente a todos los símbolos

### 4. Protección Específica por Símbolo
- **Cooldowns Independientes**: Cada símbolo tiene su propio cronómetro
- **Configuración**: Activable/desactivable via `LOSS_PROTECTION_SYMBOL_SPECIFIC`
- **Flexibilidad**: Permite trading en otros símbolos mientras uno está en cooldown

## 🚨 Sistema de Bypass

### Activación
- **Umbral de Confianza**: 95% por defecto (configurable)
- **Configuración**: `LOSS_PROTECTION_BYPASS_CONFIDENCE`
- **Habilitación**: Controlado por `LOSS_PROTECTION_BYPASS_ENABLED`

### Condiciones
```python
if signal_confidence >= bypass_threshold and bypass_enabled:
    # Permitir operación pese a protección activa
    return True, "BYPASS_HIGH_CONFIDENCE_95.0%"
```

## ⚙️ Configuración

### Variables de Entorno

```bash
# Tiempos de cooldown por magnitud
LOSS_PROTECTION_SMALL_LOSS_COOLDOWN=15      # Minutos para pérdida < 2%
LOSS_PROTECTION_MEDIUM_LOSS_COOLDOWN=30     # Minutos para pérdida 2-5%
LOSS_PROTECTION_LARGE_LOSS_COOLDOWN=60      # Minutos para pérdida > 5%

# Protección por pérdidas consecutivas
LOSS_PROTECTION_CONSECUTIVE_PENALTY=90      # Minutos adicionales
LOSS_PROTECTION_MAX_CONSECUTIVE_LOSSES=3    # Límite antes de penalización

# Protección por pérdida diaria
LOSS_PROTECTION_DAILY_LOSS_THRESHOLD=10.0   # % pérdida diaria
LOSS_PROTECTION_DAILY_LOSS_PENALTY=120      # Minutos de pausa

# Configuración específica
LOSS_PROTECTION_SYMBOL_SPECIFIC=true        # Cooldown por símbolo vs global
LOSS_PROTECTION_BYPASS_CONFIDENCE=95.0      # % confianza para bypass
LOSS_PROTECTION_BYPASS_ENABLED=true         # Permitir bypass
```

### Configuración Programática

```python
from loss_protection import LossProtectionManager, LossProtectionConfig

# Configuración personalizada
config = LossProtectionConfig(
    small_loss_cooldown_minutes=20,
    medium_loss_cooldown_minutes=40,
    large_loss_cooldown_minutes=80,
    consecutive_losses_penalty_minutes=120,
    daily_loss_threshold=8.0,
    bypass_confidence_threshold=90.0,
    symbol_specific_cooldown=True
)

# Inicializar manager
loss_protection = LossProtectionManager(config)
```

## 🔄 Flujo de Operación

### 1. Registro de Cierre de Posición
```python
# Al cerrar una posición
loss_protection.register_position_close(
    symbol="BTCUSDT",
    pnl_percent=-2.5,  # Pérdida del 2.5%
    pnl_usd=-25.0,
    close_reason="STOP_LOSS",
    entry_time=datetime(2024, 1, 1, 10, 0)
)
```

### 2. Verificación Antes de Nueva Posición
```python
# Antes de abrir posición
can_trade, reason = loss_protection.can_open_position(
    symbol="BTCUSDT",
    signal_confidence=85.0
)

if not can_trade:
    print(f"Operación bloqueada: {reason}")
    # MEDIUM_LOSS_30min_remaining
```

### 3. Estado y Reportes
```python
# Obtener estado actual
status = loss_protection.get_protection_status()

# Generar reporte formateado
report = loss_protection.format_protection_report()
print(report)
```

## 📊 Ejemplos de Uso

### Escenario 1: Pérdida Pequeña
```
Posición ETHUSDT cerrada: -1.5% (SMALL_LOSS)
→ Cooldown: 15 minutos
→ ETHUSDT bloqueado hasta 14:25
→ Otros símbolos disponibles (si symbol_specific=true)
```

### Escenario 2: Pérdidas Consecutivas
```
1. BTCUSDT: -2.0% → 30min cooldown
2. ETHUSDT: -1.8% → 15min cooldown (consecutiva #2)
3. BNBUSDT: -3.2% → 30min + 90min penalty = 120min cooldown (consecutiva #3)
```

### Escenario 3: Bypass por Alta Confianza
```
XRPUSDT en cooldown (45min restantes)
Señal BUY con 96% confianza
→ BYPASS activado
→ Operación permitida
```

### Escenario 4: Pérdida Diaria Crítica
```
Pérdidas acumuladas: -10.5% en 24h
→ Protección global activada
→ Todos los símbolos bloqueados por 2 horas
→ Solo bypass con >95% confianza permitido
```

## 🎛️ Integración en Trading Manager

### Inicialización
```python
class SimpleProfessionalTradingManager:
    def __init__(self):
        # Configurar desde variables de entorno
        loss_config = LossProtectionConfig(
            small_loss_cooldown_minutes=int(os.getenv('LOSS_PROTECTION_SMALL_LOSS_COOLDOWN', '15')),
            # ... más configuraciones
        )
        self.loss_protection = LossProtectionManager(loss_config)
```

### Verificación Pre-Trading
```python
async def _consider_new_position(self, symbol: str, signal_data: Dict):
    # Verificar protección ANTES de risk management
    can_trade, protection_reason = self.loss_protection.can_open_position(
        symbol, signal_data['confidence']
    )

    if not can_trade:
        await self._send_discord_notification(
            f"🛡️ PROTECCIÓN POST-PÉRDIDAS: {symbol} - {protection_reason}"
        )
        return
```

### Registro de Cierres
```python
async def _close_position(self, order_id: str, reason: str):
    # ... lógica de cierre ...

    # Registrar para protección
    self.loss_protection.register_position_close(
        symbol=symbol,
        pnl_percent=pnl_percent,
        pnl_usd=pnl_usd,
        close_reason=reason,
        entry_time=position.entry_time
    )
```

## 📈 Reportes y Monitoreo

### Reporte de Estado
```
🛡️ **PROTECCIÓN POST-PÉRDIDAS:**
🔒 **Cooldowns por símbolo:**
   BTCUSDT: 25.3min restantes
   ETHUSDT: 8.7min restantes

📊 **Estadísticas:**
   🔴 Pérdidas consecutivas: 2/3
   📉 Pérdida diaria: 4.2%/10.0%
   🎯 Bypass activado: 95.0% confianza

✅ **No hay protecciones activas**
```

### Métricas de Estado
```python
{
    'active_symbol_cooldowns': {
        'BTCUSDT': {'remaining_minutes': 25.3, 'end_time': datetime(...)},
        'ETHUSDT': {'remaining_minutes': 8.7, 'end_time': datetime(...)}
    },
    'global_cooldown': None,
    'consecutive_losses_count': 2,
    'daily_loss_percent': 4.2,
    'recent_losses_count': 3,
    'bypass_threshold': 95.0,
    'protection_thresholds': {
        'small_loss': 2.0,
        'large_loss': 5.0,
        'daily_loss': 10.0
    }
}
```

## 🚨 Notificaciones Discord

### Tipos de Notificaciones

1. **Bloqueo por Protección**
```
🛡️ **PROTECCIÓN POST-PÉRDIDAS**
📊 BTCUSDT: BUY
🚫 Razón: MEDIUM_LOSS_25min_remaining
🎯 Confianza: 78.5%
💡 El sistema está protegiendo contra revenge trading
```

2. **Bypass Activado**
```
🚨 **BYPASS ACTIVADO**
📊 ETHUSDT: BUY
🎯 Confianza: 96.2% >= 95.0%
⚡ Operación permitida pese a protección activa
```

3. **Protección Global**
```
🌐 **PROTECCIÓN GLOBAL ACTIVADA**
📉 Pérdida diaria: 11.2%/10.0%
⏰ Duración: 120 minutos
🛡️ Todos los símbolos pausados
```

## 🔧 Personalización Avanzada

### Configuración por Símbolo
```python
# Configurar umbrales específicos por símbolo
config.symbol_thresholds = {
    'BTCUSDT': {'small': 1.5, 'large': 4.0},  # BTC más conservador
    'ETHUSDT': {'small': 2.5, 'large': 6.0},  # ETH más permisivo
}
```

### Horarios de Protección
```python
# Protección más estricta en horarios específicos
config.time_based_multipliers = {
    'market_open': 1.5,    # 50% más tiempo en apertura
    'high_volatility': 2.0  # Doble tiempo en alta volatilidad
}
```

### Integración con Market Regime
```python
# Ajustar protección según régimen de mercado
if market_context['regime'] == 'BEARISH':
    config.apply_bearish_multiplier(1.5)  # 50% más protección
elif market_context['regime'] == 'BULLISH':
    config.apply_bullish_multiplier(0.7)  # 30% menos protección
```

## 📋 Checklist de Implementación

- [x] ✅ Importar LossProtectionManager
- [x] ✅ Configurar en constructor del Trading Manager
- [x] ✅ Integrar verificación pre-trading
- [x] ✅ Registrar cierres de posición
- [x] ✅ Agregar a reportes TCN
- [x] ✅ Mostrar en display de tiempo real
- [x] ✅ Incluir en estado del sistema
- [x] ✅ Configurar notificaciones Discord
- [x] ✅ Documentar variables de entorno
- [ ] ⏳ Testing en diferentes escenarios
- [ ] ⏳ Optimización de parámetros
- [ ] ⏳ Integración con backtesting

## 🧪 Testing y Validación

### Casos de Prueba

1. **Pérdida Pequeña**: Verificar cooldown de 15 minutos
2. **Pérdida Media**: Verificar cooldown de 30 minutos
3. **Pérdida Grande**: Verificar cooldown de 60 minutos
4. **Pérdidas Consecutivas**: Verificar penalización adicional
5. **Pérdida Diaria**: Verificar protección global
6. **Bypass**: Verificar funcionamiento con alta confianza
7. **Limpieza**: Verificar limpieza de historial antiguo

### Comandos de Testing
```bash
# Testing básico
python -c "from loss_protection import LossProtectionManager; mgr = LossProtectionManager(); print(mgr.get_protection_status())"

# Testing con pérdida simulada
python test_loss_protection.py

# Testing de integración
python simple_professional_manager2.py --test-mode
```

## 🎯 Beneficios del Sistema

1. **Protección Emocional**: Previene decisiones impulsivas post-pérdida
2. **Preservación de Capital**: Limita pérdidas consecutivas y acumuladas
3. **Flexibilidad**: Permite bypass para oportunidades excepcionales
4. **Transparencia**: Reportes claros del estado de protección
5. **Configurabilidad**: Adaptable a diferentes estilos de trading
6. **Integración**: Funciona seamlessly con el sistema existente

## 🚀 Roadmap Futuro

1. **Machine Learning**: Predicción adaptativa de cooldowns óptimos
2. **Análisis de Correlación**: Protección cruzada entre símbolos correlacionados
3. **Integración de Sentimiento**: Ajustar protección según sentimiento del mercado
4. **Análisis de Sesión**: Protección específica por sesiones de trading
5. **Backtesting**: Análisis histórico del impacto del sistema

---

**🛡️ El Sistema de Protección Post-Pérdidas es una herramienta esencial para cualquier trader serio que busque proteger su capital y mantener la disciplina en el trading automatizado.**
