# 🔍 ANÁLISIS DE SEÑALES DE TRADING TCN

Este conjunto de scripts permite analizar y validar las señales generadas por el sistema TCN de trading para asegurar que las predicciones sean coherentes y estén funcionando correctamente.

## 📋 Scripts Disponibles

### 1. `run_signal_analysis.py` - Script Principal
**Ejecutor principal que combina todos los análisis**

```bash
# Análisis completo (recomendado)
python run_signal_analysis.py

# Verificación rápida (solo validación de mapeo)
python run_signal_analysis.py --quick

# Solo validación de mapeo de señales
python run_signal_analysis.py --validation-only

# Solo análisis de coherencia técnica
python run_signal_analysis.py --analysis-only

# Ayuda detallada
python run_signal_analysis.py --help-usage
```

### 2. `validate_signal_mapping.py` - Validador de Mapeo
**Valida que el mapeo de señales sea consistente entre entrenamiento y predicción**

```bash
python validate_signal_mapping.py
```

**Verifica:**
- ✅ Consistencia del mapeo `{0: 'SELL', 1: 'HOLD', 2: 'BUY'}`
- ✅ Suma de probabilidades = 1.0
- ✅ Señal corresponde a mayor probabilidad
- ✅ Confianza corresponde a mayor probabilidad

### 3. `analyze_trading_signals.py` - Analizador de Coherencia
**Analiza la coherencia entre señales TCN e indicadores técnicos**

```bash
python analyze_trading_signals.py
```

**Analiza:**
- 📊 RSI (14, 21), MACD, Stochastic
- 📈 Bollinger Bands, Moving Averages
- 🎯 Coherencia TCN vs análisis técnico
- 📋 Estadísticas de acuerdo por símbolo

## 🎯 ¿Qué Información Proporcionan?

### Validación de Mapeo
```
🔍 ANÁLISIS DE COHERENCIA DE SEÑALES - ETHUSDT
============================================================

💹 PREDICCIÓN TCN ACTUAL:
   Señal: BUY
   Confianza: 67.3%
   Precio actual: $3,245.67

📈 ANÁLISIS TÉCNICO ACTUAL:
   Señal técnica: BUY
   Señales de compra: 4
   Señales de venta: 1
   RSI: 42.3
   MACD: 0.0156
   Stochastic K: 38.7

🎯 COHERENCIA:
   ✅ COHERENTE
   Tasa de acuerdo (24h): 73.5%
   Acuerdos: 18/24
```

### Resumen General
```
📊 RESUMEN GENERAL DE ANÁLISIS DE SEÑALES
======================================================================
BTCUSDT    | Coherencia:  78.2% | ✅ BUENA
ETHUSDT    | Coherencia:  73.5% | ✅ BUENA
BNBUSDT    | Coherencia:  65.8% | ⚠️ REGULAR
XRPUSDT    | Coherencia:  71.1% | ✅ BUENA

📈 COHERENCIA PROMEDIO: 72.2%
✅ SISTEMA FUNCIONANDO CORRECTAMENTE
```

## 📊 Interpretación de Resultados

### Tasa de Coherencia
- **≥ 70%**: ✅ **BUENA** - Sistema funcionando correctamente
- **50-69%**: ⚠️ **REGULAR** - Requiere ajustes menores
- **< 50%**: ❌ **MALA** - Requiere revisión urgente

### Indicadores Técnicos Analizados

#### RSI (Relative Strength Index)
- **< 30**: Oversold (señal de compra)
- **> 70**: Overbought (señal de venta)
- **30-45**: Zona de compra neutral
- **55-70**: Zona de venta neutral

#### MACD
- **> 0.1**: Fuertemente positivo (compra)
- **> 0**: Positivo (compra débil)
- **< -0.1**: Fuertemente negativo (venta)
- **< 0**: Negativo (venta débil)

#### Stochastic
- **< 20**: Oversold (compra)
- **> 80**: Overbought (venta)

#### Bollinger Bands
- **Posición < 0.2**: Cerca del límite inferior (compra)
- **Posición > 0.8**: Cerca del límite superior (venta)
- **Width < 0.02**: Squeeze detectado (volatilidad baja)

## 🚨 Problemas Detectados y Soluciones

### ❌ Señales Invertidas (SOLUCIONADO)
**Problema:** El mapeo de señales estaba invertido entre entrenamiento y predicción.

**Solución Aplicada:**
- ✅ Corregido `signal_map` en `predict_signal()`
- ✅ Corregido `class_names` en `predict()`
- ✅ Alineado con entrenamiento: `{0: 'SELL', 1: 'HOLD', 2: 'BUY'}`

### ⚠️ Baja Coherencia
**Síntomas:**
- Tasa de acuerdo < 60%
- Señales TCN contradicen indicadores técnicos

**Posibles Causas:**
- Thresholds de modelo muy agresivos/conservadores
- Datos de entrenamiento desactualizados
- Condiciones de mercado atípicas

**Soluciones:**
1. Ajustar thresholds específicos por símbolo
2. Reentrenar modelo con datos más recientes
3. Implementar filtros de contexto de mercado

## 📁 Archivos Generados

### Reportes Individuales
- `signal_analysis_BTCUSDT_20241215_143022.txt`
- `signal_analysis_ETHUSDT_20241215_143022.txt`
- `signal_analysis_BNBUSDT_20241215_143022.txt`
- `signal_analysis_XRPUSDT_20241215_143022.txt`

### Reportes de Validación
- `signal_mapping_validation_20241215_143022.txt`

### Resumen JSON
- `trading_signals_summary_20241215_143022.json`

## 🔧 Configuración y Personalización

### Ajustar Thresholds Técnicos
Editar en `analyze_trading_signals.py`:

```python
self.technical_thresholds = {
    'rsi_oversold': 30,        # RSI oversold
    'rsi_overbought': 70,      # RSI overbought
    'macd_strong_positive': 0.1,   # MACD fuerte
    'macd_strong_negative': -0.1,  # MACD fuerte negativo
    # ... más thresholds
}
```

### Añadir Nuevos Símbolos
Editar la lista de símbolos:

```python
self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
```

## 🚀 Uso Recomendado

### Verificación Diaria
```bash
# Verificación rápida diaria (30 segundos)
python run_signal_analysis.py --quick
```

### Análisis Semanal
```bash
# Análisis completo semanal (3-5 minutos)
python run_signal_analysis.py
```

### Después de Cambios en el Código
```bash
# Validar que los cambios no rompieron el mapeo
python run_signal_analysis.py --validation-only
```

### Diagnóstico de Problemas
```bash
# Si hay problemas de rendimiento, analizar coherencia
python run_signal_analysis.py --analysis-only
```

## 📞 Soporte y Troubleshooting

### Error: "ModuleNotFoundError: No module named 'talib'"
```bash
python -m pip install TA-Lib
```

### Error: "No se pudieron obtener datos de mercado"
- Verificar conexión a internet
- Verificar que la API de Binance esté disponible
- Revisar rate limits

### Baja coherencia persistente
1. Verificar que el modelo está entrenado correctamente
2. Revisar thresholds específicos del símbolo
3. Considerar reentrenamiento con datos más recientes

## 📈 Mejoras Futuras

- [ ] Análisis de backtesting automático
- [ ] Integración con notificaciones Discord
- [ ] Dashboard web en tiempo real
- [ ] Análisis de correlación entre símbolos
- [ ] Detección automática de regímenes de mercado
- [ ] Optimización automática de thresholds
