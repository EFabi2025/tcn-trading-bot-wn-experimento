# 🧠 ARCHIVOS DE ENTRENAMIENTO Y MODELOS RECUPERADOS

## ✅ **Archivos Recuperados Exitosamente**

### 🎯 **Entrenamiento por Pares (Individual)**
- `train_bnbusdt_only.py` - Entrenamiento específico para BNB/USDT
- `train_btcusdt_only.py` - Entrenamiento específico para BTC/USDT
- `train_ethusdt_only.py` - Entrenamiento específico para ETH/USDT

### 🚀 **Sistema TCN Definitivo**
- `tcn_definitivo_predictor.py` - Predictor principal del sistema TCN
- `tcn_definitivo_trainer.py` - Entrenador avanzado de modelos TCN
- `threshold_calibrator.py` - Calibrador de umbrales de confianza

### 🔄 **Sistema de Reentrenamiento**
- `advanced_retrain_tcn.py` - Reentrenamiento avanzado de modelos TCN
- `retrain_all_quick.py` - Reentrenamiento rápido de todos los pares
- `retrain_tcn_anti_bias.py` - Reentrenamiento con corrección de sesgo
- `final_tcn_retrain.py` - Reentrenamiento final optimizado

### ⚖️ **Entrenamiento Balanceado**
- `advanced_balanced_tcn.py` - Entrenamiento con balanceo avanzado de datos

### 🔍 **Análisis y Desarrollo**
- `analyze_model_bias.py` - Análisis de sesgo en modelos
- `analyze_model_requirements.py` - Análisis de requisitos de modelos
- `simple_pair_models.py` - Modelos simples por pares

## 🎯 **Propósito de Cada Categoría**

### 📚 **Entrenamiento Individual (`*_only.py`)**
Estos archivos permiten entrenar modelos específicos para cada par de trading:
- Optimización específica por activo
- Parámetros personalizados según volatilidad
- Datasets enfocados en características únicas de cada par

### 🧠 **Sistema TCN Definitivo**
El núcleo del sistema de machine learning:
- **Predictor**: Interfaz principal para generar señales
- **Trainer**: Motor de entrenamiento con técnicas avanzadas
- **Calibrator**: Optimización de umbrales de confianza

### 🔄 **Reentrenamiento Automático**
Sistema para mantener los modelos actualizados:
- Reentrenamiento periódico con datos frescos
- Corrección de drift en los modelos
- Balanceo automático de datasets

### 🔍 **Herramientas de Análisis**
Para desarrollo y optimización continua:
- Detección de sesgos en predicciones
- Análisis de performance por condiciones de mercado
- Evaluación de requisitos computacionales

## 🚨 **IMPORTANTE: NO ELIMINAR**

Estos archivos son **CRÍTICOS** para:
- ✅ Entrenar nuevos modelos cuando cambien las condiciones de mercado
- ✅ Reentrenar modelos existentes con datos actualizados
- ✅ Desarrollar mejoras en el sistema de ML
- ✅ Calibrar umbrales según performance histórica
- ✅ Analizar y corregir sesgos en predicciones

## 📋 **Uso Recomendado**

### 🔄 **Reentrenamiento Mensual**
```bash
python retrain_all_quick.py
```

### 🎯 **Entrenamiento Específico**
```bash
python train_btcusdt_only.py  # Para BTC
python train_ethusdt_only.py  # Para ETH
python train_bnbusdt_only.py  # Para BNB
```

### ⚖️ **Calibración de Umbrales**
```bash
python threshold_calibrator.py
```

### 🔍 **Análisis de Modelos**
```bash
python analyze_model_bias.py
python analyze_model_requirements.py
```

---
**📅 Recuperado:** 17 de Junio 2025
**🎯 Estado:** Archivos esenciales para desarrollo futuro
**⚠️ Prioridad:** ALTA - No eliminar en futuras limpiezas
