# 🪟 TRADING BOT - GUÍA COMPLETA WINDOWS

## 🎯 Sistema de Trading Profesional con Diversificación de Portafolio

### ✨ **CARACTERÍSTICAS PRINCIPALES**

🛡️ **PROTECCIÓN TOTAL:**
- Sistema de diversificación automática
- Límites de concentración por símbolo (40% máx)
- Máximo 3 posiciones por símbolo
- Protección sin liquidación forzada
- Gestión de riesgo avanzada

🧠 **INTELIGENCIA ARTIFICIAL:**
- Red Neuronal TCN (Temporal Convolutional Network)
- Predicciones en tiempo real
- Optimización para Windows/CPU
- Análisis de correlaciones

📊 **MONITOREO COMPLETO:**
- Reportes de diversificación en tiempo real
- Alertas Discord automáticas
- Logs detallados
- Métricas de rendimiento

---

## 🚀 **INSTALACIÓN RÁPIDA**

### **Opción 1: Instalación Automática (Recomendada)**

1. **Descargar el proyecto:**
   ```cmd
   git clone [URL_DEL_REPOSITORIO]
   cd [NOMBRE_DEL_PROYECTO]
   ```

2. **Ejecutar instalador automático:**
   ```cmd
   install_windows.bat
   ```

3. **Configurar credenciales:**
   - Editar archivo `.env` con tus API keys de Binance
   - Configurar Discord webhook (opcional)

4. **Iniciar sistema:**
   ```cmd
   start_trading.bat
   ```

### **Opción 2: Instalación Manual**

Ver guía detallada en: [`WINDOWS_DEPLOYMENT_GUIDE.md`](WINDOWS_DEPLOYMENT_GUIDE.md)

---

## 🔄 **ACTUALIZACIÓN DEL SISTEMA**

Para actualizar a la última versión con nuevas características:

```cmd
update_windows.bat
```

Este script:
- ✅ Hace backup de tu configuración
- ✅ Descarga últimos cambios
- ✅ Actualiza dependencias
- ✅ Verifica compatibilidad
- ✅ Restaura tu configuración

---

## 🧪 **VERIFICACIÓN DEL SISTEMA**

Antes de usar el sistema, ejecuta la verificación de compatibilidad:

```cmd
python windows_compatibility_check.py
```

Este script verifica:
- ✅ Versión de Python compatible
- ✅ Dependencias instaladas
- ✅ Configuración TensorFlow
- ✅ Conectividad Binance
- ✅ Variables de entorno
- ✅ Sistema de diversificación

---

## ⚙️ **CONFIGURACIÓN**

### **1. Variables de Entorno (.env)**

```env
# Credenciales Binance (OBLIGATORIO)
BINANCE_API_KEY=tu_api_key_real
BINANCE_SECRET_KEY=tu_secret_key_real

# Configuración del sistema
ENVIRONMENT=production

# Notificaciones Discord (OPCIONAL)
DISCORD_WEBHOOK_URL=tu_webhook_url

# Parámetros de riesgo (OPCIONAL - usa defaults si no se especifica)
MAX_POSITION_SIZE_PERCENT=15.0
STOP_LOSS_PERCENT=1.4
TAKE_PROFIT_PERCENT=6.0
```

### **2. Configuración de Diversificación**

El sistema incluye configuración automática de diversificación en `config/trading_config.py`:

- **Límite por símbolo:** 40% del portafolio
- **Límite por categoría:** 60% del portafolio
- **Máximo posiciones por símbolo:** 3
- **Símbolos soportados:** BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, DOTUSDT, SOLUSDT

---

## 🎯 **CARACTERÍSTICAS DEL SISTEMA DE DIVERSIFICACIÓN**

### **🛡️ Protección Automática**
- **NO liquida posiciones existentes**
- Bloquea nuevas órdenes que excedan límites
- Ajusta tamaños de posición automáticamente
- Prioriza símbolos para mejor diversificación

### **📊 Análisis en Tiempo Real**
- Score de diversificación (0-100)
- Concentraciones por símbolo y categoría
- Detección de sobre-concentración
- Recomendaciones automáticas

### **🔔 Alertas Inteligentes**
- Notificaciones Discord cuando se exceden límites
- Reportes integrados en salida TCN
- Logs detallados de decisiones
- Métricas de diversificación

---

## 📋 **EJECUCIÓN DEL SISTEMA**

### **Inicio Rápido**
```cmd
start_trading.bat
```

### **Inicio Manual**
```cmd
# Activar entorno virtual
venv\Scripts\activate.bat

# Ejecutar sistema
python run_trading_manager.py
```

### **Monitoreo**
- **Logs:** Carpeta `logs\`
- **Reportes:** Integrados en salida del sistema
- **Discord:** Alertas automáticas (si configurado)

---

## 🔍 **TROUBLESHOOTING WINDOWS**

### **Errores Comunes**

**❌ "Microsoft Visual C++ 14.0 is required"**
```
Solución: Instalar Visual Studio Build Tools
https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

**❌ "Permission denied"**
```
Solución: Ejecutar PowerShell como Administrador
```

**❌ "SSL Certificate verify failed"**
```
Solución: Instalar con certificados confiables
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

**❌ "TensorFlow not optimized"**
```
Solución: Normal en Windows, el sistema funciona correctamente con CPU
```

### **Verificación de Funcionamiento**

```cmd
# Verificar procesos
tasklist | findstr python

# Ver logs en tiempo real
Get-Content logs\trading.log -Wait -Tail 50

# Verificar conexión
netstat -an | findstr :443
```

---

## 🔐 **SEGURIDAD EN WINDOWS**

### **Configuración de Firewall**
```cmd
netsh advfirewall firewall add rule name="Python Trading Bot" dir=out action=allow program="C:\ruta\proyecto\venv\Scripts\python.exe"
```

### **Exclusiones de Antivirus**
- Agregar carpeta del proyecto a exclusiones
- Agregar `python.exe` a exclusiones
- Agregar `venv\Scripts\` a exclusiones

---

## 📊 **DIFERENCIAS WINDOWS vs macOS**

| Característica | Windows | macOS |
|----------------|---------|-------|
| **Activación venv** | `venv\Scripts\activate.bat` | `source venv/bin/activate` |
| **TensorFlow** | CPU optimizado | Metal optimizado |
| **Rutas** | `\` (automático) | `/` |
| **Servicios** | NSSM/Task Scheduler | launchd |
| **Logs encoding** | UTF-8 explícito | UTF-8 automático |

---

## 🎯 **EJEMPLO DE FUNCIONAMIENTO**

### **Salida Típica del Sistema:**

```
🚀 Iniciando Trading Bot...
✅ Conexión Binance establecida - Balance: $1,234.56
🧠 Cargando modelo TCN...
✅ Modelo TCN cargado exitosamente

📊 ANÁLISIS DE DIVERSIFICACIÓN:
   Total portafolio: $1,234.56
   Diversificación score: 75.2/100
   Símbolos activos: 4
   ✅ Portafolio bien diversificado

🎯 TCN PREDICCIÓN - BTCUSDT:
   Señal: BUY (Confianza: 78.5%)
   ✅ Diversificación: Permitido (BTCUSDT: 25.3% del portafolio)
   📊 Ejecutando orden: BUY 0.001 BTC a $45,230.50

⚠️ ALERTA DIVERSIFICACIÓN:
   BNBUSDT concentración: 42.1% (límite: 40%)
   🚫 Bloqueando nuevas órdenes BNBUSDT
   💡 Priorizando: ADAUSDT, DOTUSDT, SOLUSDT
```

---

## 📞 **SOPORTE**

### **Archivos de Diagnóstico**
- `windows_compatibility_report.txt` - Reporte de compatibilidad
- `logs\trading.log` - Logs del sistema
- `logs\diversification.log` - Logs de diversificación

### **Comandos de Diagnóstico**
```cmd
# Verificar configuración
python -c "from config.trading_config import config_manager; config_manager.print_config_summary()"

# Test de conexión Binance
python -c "from binance.client import Client; import os; from dotenv import load_dotenv; load_dotenv(); print('OK' if Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY')).get_account_status() else 'ERROR')"

# Test de diversificación
python -c "from portfolio_diversification_manager import PortfolioDiversificationManager; from config.trading_config import TradingConfig; print('OK' if PortfolioDiversificationManager() else 'ERROR')"
```

---

## 🎉 **¡LISTO PARA TRADING!**

Una vez completada la instalación y configuración:

1. ✅ **Sistema verificado** con `windows_compatibility_check.py`
2. ✅ **Credenciales configuradas** en `.env`
3. ✅ **Diversificación activa** protegiendo tu portafolio
4. ✅ **Monitoreo funcionando** con logs y alertas

**¡Tu sistema de trading profesional está listo para operar en Windows!** 🚀

---

*Sistema desarrollado con las mejores prácticas de trading algorítmico y optimizado específicamente para Windows.*
