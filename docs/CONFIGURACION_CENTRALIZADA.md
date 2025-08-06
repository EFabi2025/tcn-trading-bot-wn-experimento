# 🔧 CONFIGURACIÓN CENTRALIZADA DEL SISTEMA

## 📋 Resumen

Todas las configuraciones del trading bot están ahora centralizadas en:
1. **`src/core/config.py`** - Configuración principal con validación Pydantic
2. **`.env`** - Variables de entorno para personalización
3. **`config_example.env`** - Plantilla con valores por defecto

## ⚙️ Configuraciones Principales

### 🛡️ **GESTIÓN DE RIESGO**

```env
# Stop Loss y Take Profit
STOP_LOSS_PERCENT=1.4           # Stop loss automático (1.4%)
TAKE_PROFIT_PERCENT=4.0         # Take profit automático (4.0%)

# Trailing Stop
TRAILING_STOP_PERCENT=1.4       # Trailing stop (1.4%)
TRAILING_ACTIVATION_THRESHOLD=0.4  # Activar trailing en +0.4% ganancia
ENABLE_TRAILING_STOPS=true      # Habilitar trailing stops

# Límites de pérdida
MAX_DAILY_LOSS_PERCENT=5.0      # Máxima pérdida diaria (5%)
MAX_DRAWDOWN_PERCENT=15.0       # Máximo drawdown permitido
```

### 📊 **GESTIÓN DE POSICIONES**

```env
# Tamaños de posición
MAX_POSITION_PERCENT=2.0        # Máximo por posición (2% del balance)
MAX_OPEN_POSITIONS=9            # Máximo posiciones simultáneas
MIN_ORDER_AMOUNT=10             # Monto mínimo por orden (USDT)

# Exposición total
MAX_TOTAL_EXPOSURE_PERCENT=40.0 # Máxima exposición total (40%)
MAX_CONCURRENT_POSITIONS=3      # Máximo posiciones por símbolo
```

### 🎯 **MODELO ML Y SEÑALES**

```env
# Umbrales de confianza
MIN_CONFIDENCE_THRESHOLD=0.70   # Confianza mínima para trade (70%)
MIN_SELL_CONFIDENCE_THRESHOLD=0.75  # Confianza mínima para SELL (75%)

# Parámetros del modelo
PREDICTION_INTERVAL_MINUTES=5   # Intervalo de predicción
DATA_WINDOW_SIZE=100           # Tamaño ventana de datos
```

### 🌐 **BINANCE API**

```env
# Credenciales (REQUERIDAS)
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
BINANCE_TESTNET=true           # true=testnet, false=producción

# Rate limiting
API_RATE_LIMIT=1200            # Requests por minuto
ORDER_COOLDOWN_SECONDS=5       # Espera entre órdenes
```

## 🔄 Migración de Configuraciones Hardcodeadas

### ✅ **ANTES (Hardcodeado)**
```python
# En múltiples archivos
trailing_stop_percent = 2.0
stop_loss_percent = 3.0
take_profit_percent = 6.0
```

### ✅ **DESPUÉS (Centralizado)**
```python
# En src/core/config.py
from src.core.config import get_settings
settings = get_settings()

stop_loss = settings.stop_loss_percent * 100  # 1.4%
trailing_stop = settings.trailing_stop_percent * 100  # 1.4%
take_profit = settings.take_profit_percent * 100  # 4.0%
```

## 📁 Archivos Actualizados

### 🔧 **Configuración Principal**
- `src/core/config.py` - ✅ Agregadas configuraciones de trailing stop
- `config_example.env` - ✅ Plantilla actualizada con nuevos valores

### 🛡️ **Risk Management**
- `advanced_risk_manager.py` - ✅ Usa configuración centralizada
- `config/trading_config.py` - ✅ Valores actualizados

### 📈 **Portfolio Management**
- `professional_portfolio_manager.py` - ✅ Carga configuración desde .env
- `simple_professional_manager.py` - ✅ Compatible con configuración central

## 🎯 Valores por Defecto Actualizados

| Parámetro | Antes | Después | Motivo |
|-----------|-------|---------|--------|
| `STOP_LOSS_PERCENT` | 3.0% | **1.4%** | Mayor protección de capital |
| `TRAILING_STOP_PERCENT` | 2.0% | **1.4%** | Consistencia con stop loss |
| `TAKE_PROFIT_PERCENT` | 6.0% | **4.0%** | Objetivos más realistas |
| `MAX_DAILY_LOSS_PERCENT` | 10.0% | **5.0%** | Gestión de riesgo conservadora |

## 🚀 Cómo Personalizar

### 1. **Crear archivo .env**
```bash
cp config_example.env .env
```

### 2. **Editar valores según necesidades**
```env
# Ejemplo: Trading más agresivo
STOP_LOSS_PERCENT=1.0
TAKE_PROFIT_PERCENT=5.0
MAX_POSITION_PERCENT=3.0

# Ejemplo: Trading conservador
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=3.0
MAX_POSITION_PERCENT=1.5
```

### 3. **Validación automática**
El sistema valida automáticamente:
- ✅ Take profit > Stop loss
- ✅ Valores dentro de rangos permitidos
- ✅ Configuraciones coherentes entre sí

## 🔍 Verificación de Configuración

### **Al iniciar el sistema:**
```
🛡️ Límites de riesgo configurados:
   📊 Max posición: 15.0% ($15.30)
   🚨 Max pérdida diaria: 5.0%
   🛑 Stop Loss: 1.4%
   🎯 Take Profit: 4.0%
   📈 Trailing Stop: 1.4%
   💵 Mínimo Binance: $11.00 USDT
```

### **En logs de posiciones:**
```
🛡️ Stops inicializados para BTCUSDT:
   📍 Entrada: $43,250.0000
   🛑 Stop Loss: $42,644.50 (-1.4%)
   🎯 Take Profit: $44,980.00 (+4.0%)
   📈 Trailing: INACTIVO (activar en +1.0%)
```

## ⚠️ Importante

1. **Archivo .env local**: Siempre prevalece sobre valores por defecto
2. **Validación estricta**: El sistema rechaza configuraciones inválidas
3. **Reinicio requerido**: Cambios en .env requieren reiniciar el bot
4. **Backup recomendado**: Guardar configuración que funcione bien

## 🔧 Troubleshooting

### **Problema: "Stop Loss sigue mostrando 2%"**
```bash
# Verificar archivo .env
cat .env | grep TRAILING_STOP_PERCENT

# Si no existe, agregarlo:
echo "TRAILING_STOP_PERCENT=1.4" >> .env

# Reiniciar el bot
```

### **Problema: "Configuración no se aplica"**
1. Verificar sintaxis del archivo .env
2. No usar espacios alrededor del =
3. Reiniciar completamente el bot
4. Verificar permisos del archivo .env

---

✅ **RESULTADO**: Sistema completamente centralizado con configuración consistente y validada.
