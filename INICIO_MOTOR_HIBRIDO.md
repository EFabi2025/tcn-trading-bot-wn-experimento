# 🚀 INICIO DEL BOT CON MOTOR HÍBRIDO

## 🎯 **OPCIONES DE INICIO**

### **1. 🪟 Windows PowerShell (Recomendado)**
```powershell
.\start_bot.ps1
```

### **2. 🪟 Windows Command Prompt**
```cmd
start_hybrid_bot.bat
```

### **3. 🐍 Python Directo**
```bash
python start_hybrid_trading.py
```

## ✅ **CARACTERÍSTICAS DEL MOTOR HÍBRIDO**

### **🔧 MEJORAS TÉCNICAS:**
- **Features limpias**: 57 problemas corregidos automáticamente
- **Normalización robusta**: Rango [-1, 1] optimizado
- **Decorrelación**: Features redundantes eliminadas
- **Calidad alta**: 0.84/1.0 en métricas de calidad

### **📊 MEJORAS EN PREDICCIONES:**
- **BTCUSDT**: 60.8% → **70.9%** (+10.1% confianza)
- **BNBUSDT**: 40.7% → **52.8%** (+12.1% confianza)
- **Señales más definidas**: Menos HOLD, más BUY/SELL ejecutables

### **🛡️ SEGURIDAD:**
- **Fallback automático**: Si falla el motor híbrido, usa el original
- **Validación robusta**: Verificación de calidad de features
- **Sin riesgo**: Código original intacto y funcional

## 📋 **REQUISITOS PREVIOS**

### **1. Archivo .env configurado:**
```bash
# Mínimo requerido
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
BINANCE_BASE_URL=https://api.binance.com  # Para producción
TCN_BUY_CONFIDENCE_THRESHOLD=0.50
TCN_SELL_CONFIDENCE_THRESHOLD=0.50
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
```

### **2. Entorno virtual activado:**
```bash
# Crear si no existe
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 🧪 **VERIFICAR ANTES DE INICIAR**

### **Test de Integración:**
```bash
python test_hybrid_integration.py
```

**Salida esperada:**
```
✅ Trading Manager inicializado correctamente
✅ Motor híbrido integrado y funcionando
✅ Predicciones generadas con features optimizadas
✅ Sistema con fallback seguro al motor original
```

## 🚀 **INICIO PASO A PASO**

### **Opción 1: PowerShell (Más completo)**
1. Abre PowerShell en la carpeta del bot
2. Si es la primera vez, ejecuta:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
3. Inicia el bot:
   ```powershell
   .\start_bot.ps1
   ```

### **Opción 2: Batch File (Más simple)**
1. Doble click en `start_hybrid_bot.bat`
2. O desde cmd: `start_hybrid_bot.bat`

### **Opción 3: Python Directo (Para desarrolladores)**
```bash
# Activar entorno
.venv\Scripts\activate

# Iniciar bot híbrido
python start_hybrid_trading.py
```

## 📊 **MONITOREO EN TIEMPO REAL**

### **Logs del Motor Híbrido:**
```
🔮 BTCUSDT: Predicción con features híbridas (calidad: 0.84)
🔮 Predicción para BTCUSDT: Señal=BUY, Confianza=0.71 [hybrid_optimized, Q:0.84]
```

### **Indicadores de Calidad:**
- **[hybrid_optimized]**: Motor híbrido funcionando
- **[definitivo_fallback]**: Usando motor original
- **Q:0.84**: Calidad de features (0-1)

## ⚠️ **SOLUCIÓN DE PROBLEMAS**

### **Error: "Motor híbrido no disponible"**
```bash
# Verificar archivos
ls hybrid_features_engine.py
ls start_hybrid_trading.py

# Re-ejecutar test
python test_hybrid_integration.py
```

### **Error: "Archivo .env no encontrado"**
```bash
# Copiar ejemplo
cp config_example.env .env

# Editar con tus credenciales
notepad .env
```

### **Error: "Entorno virtual no encontrado"**
```bash
# Crear entorno
python -m venv venv

# Activar e instalar
venv\Scripts\activate
pip install -r requirements.txt
```

## 🎉 **CONFIRMACIÓN DE ÉXITO**

### **El bot está funcionando correctamente si ves:**
```
🎉 ¡BOT INICIADO EXITOSAMENTE!
📊 Características del bot:
   ✅ Motor de Features Híbridas activado
   ✅ Fallback automático al motor original
   ✅ Umbrales de confianza optimizados
   ✅ Gestión avanzada de riesgo
```

### **Predicciones exitosas:**
```
🔮 BTCUSDT: Predicción con features híbridas (calidad: 0.84)
   ✅ BTCUSDT:
      🎯 Señal: BUY
      📊 Confianza: 70.9%
      🔧 Motor: hybrid_optimized
      ⭐ Calidad: 0.84
```

## 📞 **SOPORTE**

- **Documentación completa**: `CAMPOS_ENV_REQUERIDOS.md`
- **Configuración**: `config_example.env`
- **Tests**: `test_hybrid_integration.py`
- **Diagnóstico**: `diagnose_features_issues.py` 