# 🎯 CONFIGURACIÓN AVANZADA DE TIMEFRAMES - TCN HYBRID TRAINER

## 📊 Nuevas Opciones Configurables

El entrenador TCN Hybrid ahora incluye **configuración avanzada** para diferentes timeframes con opciones personalizables.

### 🎯 Timeframes Soportados

| Timeframe | Horizonte Predicción | Días Datos | Lookback | Arquitectura |
|-----------|---------------------|------------|----------|--------------|
| **1m** | 6 períodos | 60 días | 24 períodos | Simplificada |
| **3m** | 8 períodos | 90 días | 32 períodos | Balanceada |
| **5m** | 10 períodos | 120 días | 30 períodos | Completa |
| **15m** | 12 períodos | 150 días | 40 períodos | Robusta |
| **1h** | 16 períodos | 180 días | 48 períodos | Completa |
| **4h** | 20 períodos | 200 días | 60 períodos | Robusta |

### 🔧 Parámetros Configurables

#### 1. **Timeframe** 📊
- **Opciones**: 1m, 3m, 5m, 15m, 1h, 4h
- **Default**: 5m
- **Descripción**: Intervalo de tiempo para los datos de entrenamiento

#### 2. **Horizonte de Predicción** 🎯
- **Rango**: 6-20 períodos
- **Default**: Automático según timeframe
- **Descripción**: Número de períodos hacia el futuro para predecir
- **Recomendaciones**:
  - Timeframes cortos (1m-5m): 6-12 períodos
  - Timeframes largos (15m-4h): 8-20 períodos

#### 3. **Días de Datos** 📈
- **Rango**: 60-200 días
- **Default**: Automático según timeframe
- **Descripción**: Cantidad de datos históricos para entrenar
- **Recomendaciones**:
  - Timeframes cortos: 60-120 días
  - Timeframes largos: 150-200 días

#### 4. **Ventana de Lookback** 🔍
- **Rango**: 24-60 períodos
- **Default**: Automático según timeframe
- **Descripción**: Número de períodos históricos para analizar
- **Recomendaciones**:
  - Timeframes cortos: 24-32 períodos
  - Timeframes largos: 40-60 períodos

### 🚀 Configuraciones Optimizadas por Timeframe

#### **1m - Entrenamiento Rápido**
```python
timeframe = '1m'
prediction_horizon = 6
data_days = 60
lookback_window = 24
```
- **Uso**: Trading de alta frecuencia
- **Arquitectura**: Simplificada para velocidad
- **Datos**: 60 días (suficiente para 1m)

#### **3m - Balanceado**
```python
timeframe = '3m'
prediction_horizon = 8
data_days = 90
lookback_window = 32
```
- **Uso**: Trading de corto plazo
- **Arquitectura**: Balanceada entre velocidad y precisión
- **Datos**: 90 días (equilibrio)

#### **5m - Estándar**
```python
timeframe = '5m'
prediction_horizon = 10
data_days = 120
lookback_window = 30
```
- **Uso**: Trading diario estándar
- **Arquitectura**: Completa
- **Datos**: 120 días (robusto)

#### **15m - Largo Plazo**
```python
timeframe = '15m'
prediction_horizon = 12
data_days = 150
lookback_window = 40
```
- **Uso**: Swing trading
- **Arquitectura**: Robusta
- **Datos**: 150 días (más histórico)

#### **1h - Análisis Diario**
```python
timeframe = '1h'
prediction_horizon = 16
data_days = 180
lookback_window = 48
```
- **Uso**: Análisis diario
- **Arquitectura**: Completa
- **Datos**: 180 días (muy robusto)

#### **4h - Análisis Semanal**
```python
timeframe = '4h'
prediction_horizon = 20
data_days = 200
lookback_window = 60
```
- **Uso**: Análisis semanal
- **Arquitectura**: Robusta
- **Datos**: 200 días (máximo histórico)

### 🎯 Ejemplos de Uso

#### **Ejemplo 1: Trading de Alta Frecuencia (1m)**
```bash
python tcn_hybrid_trainer.py
# Seleccionar: 1m
# Horizonte: 6 (default)
# Días: 60 (default)
# Lookback: 24 (default)
```

#### **Ejemplo 2: Trading Diario Personalizado (5m)**
```bash
python tcn_hybrid_trainer.py
# Seleccionar: 5m
# Horizonte: 15 (personalizado)
# Días: 150 (personalizado)
# Lookback: 40 (personalizado)
```

#### **Ejemplo 3: Análisis Semanal (4h)**
```bash
python tcn_hybrid_trainer.py
# Seleccionar: 4h
# Horizonte: 20 (default)
# Días: 200 (default)
# Lookback: 60 (default)
```

### 📊 Ventajas de la Configuración Avanzada

1. **Flexibilidad Total**: Personalizar todos los parámetros
2. **Optimización por Timeframe**: Configuraciones pre-optimizadas
3. **Escalabilidad**: Desde 1m hasta 4h
4. **Adaptabilidad**: Ajustar según estrategia de trading
5. **Eficiencia**: Configuraciones automáticas inteligentes

### 🔧 Configuración Automática

El sistema incluye **configuración automática inteligente**:

- **Timeframes cortos**: Menos datos, arquitectura simplificada
- **Timeframes largos**: Más datos, arquitectura robusta
- **Horizontes adaptativos**: Según la frecuencia del timeframe
- **Lookback optimizado**: Balance entre memoria y precisión

### 📁 Estructura de Guardado

Los modelos se guardan con la siguiente estructura:
```
models/
├── definitivo_v3_1m_xrpusdt/
├── definitivo_v3_3m_xrpusdt/
├── definitivo_v3_5m_xrpusdt/
├── definitivo_v3_15m_xrpusdt/
├── definitivo_v3_1h_xrpusdt/
└── definitivo_v3_4h_xrpusdt/
```

### 🎯 Compatibilidad

- ✅ **Predictor**: Compatible con todos los timeframes
- ✅ **Features**: Motor centralizado funciona con todos
- ✅ **Modelos**: Guardados en formato estándar
- ✅ **Escalado**: RobustScaler compatible
- ✅ **Clases**: 3 clases (SELL/HOLD/BUY) en todos

### 🚀 Próximos Pasos

1. **Entrenar modelos** para diferentes timeframes
2. **Probar configuraciones** personalizadas
3. **Optimizar parámetros** según resultados
4. **Implementar ensemble** multi-timeframe
5. **Validar rendimiento** en diferentes condiciones

---

**🎯 Resultado**: Sistema completamente configurable para cualquier estrategia de trading desde alta frecuencia hasta análisis semanal.
