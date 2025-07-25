# 🎯 SISTEMA DE ENSAMBLE TCN - GUÍA COMPLETA

## 🚀 RESUMEN EJECUTIVO

Este sistema de ensamble combina modelos TCN entrenados en diferentes timeframes (1m y 5m) para generar predicciones más robustas y estables. El ensamble mejora la precisión al aprovechar las fortalezas de cada timeframe:

- **Modelo 1m**: Captura patrones de alta frecuencia y señales rápidas
- **Modelo 5m**: Identifica tendencias y patrones de mediano plazo

---

## 📁 ARCHIVOS DEL SISTEMA

### 🎯 **Archivos Principales**

| Archivo | Descripción |
|---------|-------------|
| `tcn_ensemble_trainer.py` | Entrenador que crea modelos para ambos timeframes |
| `tcn_ensemble_predictor.py` | Predictor que combina las predicciones |
| `run_ensemble_trading.py` | Demo completo del sistema |
| `tcn_hybrid_trainer.py` | Entrenador híbrido para 5m (ya existía) |

### 📊 **Estructura de Modelos Generados**

```
models/
├── ensemble_1m_btcusdt/
│   ├── best_model.h5
│   ├── model.h5
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── class_weights.pkl
│   └── ensemble_metrics.pkl
├── ensemble_5m_btcusdt/
│   ├── best_model.h5
│   ├── model.h5
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── class_weights.pkl
│   └── ensemble_metrics.pkl
└── ... (para cada símbolo)
```

---

## 🔧 CONFIGURACIÓN DEL SISTEMA

### **Timeframes y Configuración**

| Timeframe | Lookback Window | Prediction Horizon | Uso Principal |
|-----------|----------------|-------------------|---------------|
| **1m** | 48 velas (48 min) | 12 velas (12 min) | Señales rápidas |
| **5m** | 24 velas (2 horas) | 6 velas (30 min) | Tendencias |

### **Pesos del Ensamble**

- **Modelo 1m**: 40% - Menor peso para alta frecuencia
- **Modelo 5m**: 60% - Mayor peso para tendencias

### **Símbolos Soportados**

- ✅ BTCUSDT
- ✅ ETHUSDT  
- ✅ BNBUSDT
- ✅ XRPUSDT

---

## 🚀 GUÍA DE USO RÁPIDO

### **1. Entrenar Modelos de Ensamble**

```bash
# Entrenar modelos para todos los símbolos
python tcn_ensemble_trainer.py

# O ejecutar el demo completo
python run_ensemble_trading.py
```

### **2. Generar Predicciones**

```python
from tcn_ensemble_predictor import TCNEnsemblePredictor

# Crear predictor
predictor = TCNEnsemblePredictor()

# Cargar modelos
predictor.load_ensemble_models()

# Generar predicción para un símbolo
result = await predictor.predict_ensemble('BTCUSDT', 'weighted_average')

# Generar predicciones para todos los símbolos
all_results = await predictor.predict_all_symbols('confidence_based')
```

### **3. Ejecutar Demo Completo**

```bash
python run_ensemble_trading.py
```

---

## 🎯 MÉTODOS DE COMBINACIÓN

### **1. Weighted Average** (Recomendado)
- Combina predicciones usando promedio ponderado
- Considera accuracy del modelo y peso del timeframe
- Más estable y equilibrado

### **2. Confidence Based**
- Selecciona la predicción con mayor confianza
- Si confianza > 85%, usa esa predicción directamente
- Fallback a weighted average si no hay alta confianza

### **3. Consensus**
- Busca consenso entre predicciones
- Si hay acuerdo total, promedia confianzas
- Si no hay consenso, usa mayoría con confianza reducida

---

## 📊 EJEMPLO DE USO AVANZADO

```python
import asyncio
from tcn_ensemble_predictor import TCNEnsemblePredictor

async def trading_analysis():
    # Crear predictor
    predictor = TCNEnsemblePredictor()
    
    # Cargar modelos
    if not predictor.load_ensemble_models():
        print("Error: No se pudieron cargar los modelos")
        return
    
    # Analizar mercado completo
    results = await predictor.predict_all_symbols('weighted_average')
    
    # Filtrar señales de alta confianza
    high_confidence_signals = []
    
    for symbol, result in results.items():
        if result['ensemble_confidence'] > 0.75:
            high_confidence_signals.append({
                'symbol': symbol,
                'signal': result['ensemble_signal'],
                'confidence': result['ensemble_confidence']
            })
    
    # Mostrar señales de alta confianza
    print("🎯 SEÑALES DE ALTA CONFIANZA:")
    for signal in high_confidence_signals:
        print(f"   {signal['symbol']}: {signal['signal']} ({signal['confidence']:.3f})")

# Ejecutar
asyncio.run(trading_analysis())
```

---

## 🔍 INTERPRETACIÓN DE RESULTADOS

### **Estructura de Respuesta del Ensamble**

```python
{
    'symbol': 'BTCUSDT',
    'ensemble_signal': 'BUY',           # Señal final
    'ensemble_confidence': 0.847,       # Confianza del ensamble
    'ensemble_probabilities': {         # Probabilidades combinadas
        'SELL': 0.123,
        'HOLD': 0.030,
        'BUY': 0.847
    },
    'individual_predictions': [         # Predicciones individuales
        {
            'timeframe': '1m',
            'signal': 'BUY',
            'confidence': 0.820,
            'weight': 0.4
        },
        {
            'timeframe': '5m',
            'signal': 'BUY',
            'confidence': 0.890,
            'weight': 0.6
        }
    ],
    'combination_method': 'weighted_average',
    'total_weight': 1.0
}
```

### **Niveles de Confianza**

| Rango | Interpretación | Acción Recomendada |
|-------|----------------|-------------------|
| **> 0.85** | Muy alta confianza | Ejecutar trade |
| **0.70 - 0.85** | Alta confianza | Considerar trade |
| **0.60 - 0.70** | Confianza media | Cautela |
| **< 0.60** | Baja confianza | Evitar trade |

---

## ⚙️ ARQUITECTURA TÉCNICA

### **Diferencias entre Modelos 1m y 5m**

| Aspecto | Modelo 1m | Modelo 5m |
|---------|-----------|-----------|
| **Capas TCN** | 6 capas (más profundo) | 5 capas (más amplio) |
| **Filtros** | [64,128,256,512,256,128] | [96,192,384,192,96] |
| **Dilaciones** | [1,2,4,8,16,32] | [1,2,4,8,16] |
| **Learning Rate** | 0.0005 | 0.0003 |
| **Batch Size** | 32 | 64 |

### **Proceso de Entrenamiento**

1. **Obtención de Datos**: 15 días de datos históricos
2. **Feature Engineering**: 66 features técnicos usando motor centralizado
3. **Etiquetado Balanceado**: Thresholds específicos por símbolo
4. **Entrenamiento**: Con class weights para evitar sesgo
5. **Validación**: Split estratificado 80/20
6. **Guardado**: Modelo + scaler + features + métricas

---

## 🎯 VENTAJAS DEL SISTEMA DE ENSAMBLE

### **1. Mayor Robustez**
- Reduce errores de predicciones individuales
- Combina información de múltiples timeframes
- Menos susceptible a ruido de mercado

### **2. Mejor Estabilidad**
- Señales más consistentes
- Reduce cambios frecuentes de señal
- Mayor confianza en las predicciones

### **3. Flexibilidad**
- Múltiples métodos de combinación
- Configuración ajustable de pesos
- Fácil expansión a nuevos timeframes

### **4. Transparencia**
- Predicciones individuales visibles
- Métricas de confianza detalladas
- Trazabilidad completa del proceso

---

## 🚀 SIGUIENTES PASOS

### **Integración en Trading Manager**

1. **Reemplazar Predictor Actual**
   ```python
   # En lugar de usar un solo predictor
   from tcn_ensemble_predictor import TCNEnsemblePredictor
   
   self.predictor = TCNEnsemblePredictor()
   self.predictor.load_ensemble_models()
   ```

2. **Configurar Umbrales de Confianza**
   ```python
   # En tu trading manager
   MIN_ENSEMBLE_CONFIDENCE = 0.75
   HIGH_CONFIDENCE_THRESHOLD = 0.85
   ```

3. **Implementar Lógica de Trading**
   ```python
   async def generate_signals(self):
       results = await self.predictor.predict_all_symbols('confidence_based')
       
       for symbol, result in results.items():
           if result['ensemble_confidence'] > MIN_ENSEMBLE_CONFIDENCE:
               # Ejecutar lógica de trading
               await self.execute_signal(symbol, result)
   ```

### **Monitoreo y Optimización**

1. **Métricas de Rendimiento**
   - Accuracy del ensamble vs modelos individuales
   - Distribución de confianzas
   - Tiempo de respuesta

2. **Ajuste de Pesos**
   - Evaluar rendimiento por timeframe
   - Ajustar pesos según performance
   - Considerar condiciones de mercado

3. **Expansión del Sistema**
   - Agregar nuevos timeframes (15m, 1h)
   - Incorporar más símbolos
   - Implementar meta-learning

---

## 📋 TROUBLESHOOTING

### **Problemas Comunes**

1. **"No se pudieron cargar los modelos"**
   - Verificar que existen los archivos en `models/ensemble_*_symbol/`
   - Ejecutar primero `tcn_ensemble_trainer.py`

2. **"Error calculando features"**
   - Verificar conexión a API de Binance
   - Comprobar que `centralized_features_engine2.py` funciona

3. **"Datos insuficientes"**
   - Incrementar `hours` en `get_market_data()`
   - Verificar disponibilidad de datos históricos

### **Logs y Debug**

```python
# Activar logging detallado
import logging
logging.basicConfig(level=logging.DEBUG)

# Verificar carga de modelos
predictor = TCNEnsemblePredictor()
success = predictor.load_ensemble_models()
print(f"Modelos cargados: {success}")
```

---

## 🏆 CONCLUSIÓN

El sistema de ensamble TCN proporciona una solución robusta y escalable para generar predicciones de trading más precisas. Al combinar modelos de diferentes timeframes con múltiples estrategias de combinación, ofrece mayor estabilidad y confianza que los modelos individuales.

La implementación modular permite fácil integración en sistemas existentes y expansión futura según las necesidades del proyecto.

---

**🎯 ¡Sistema listo para producción!** 🚀 