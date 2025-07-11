# 🔄 GUÍA DE SINCRONIZACIÓN HISTÓRICA CON BINANCE

## 📋 Configuración Inicial

### 1. 🔑 Configurar API Keys de Binance

#### Opción A: Variables de Entorno
```bash
export BINANCE_API_KEY="tu_api_key_aqui"
export BINANCE_API_SECRET="tu_api_secret_aqui"
```

#### Opción B: Archivo .env (Recomendado)
```bash
# Crear archivo .env
touch .env

# Agregar credenciales (reemplaza con tus keys reales)
echo 'BINANCE_API_KEY=tu_api_key_aqui' >> .env
echo 'BINANCE_API_SECRET=tu_api_secret_aqui' >> .env
```

### 2. 📡 Obtener API Keys

#### Para Testnet (Recomendado para pruebas):
- 🌐 **URL**: https://testnet.binance.vision/
- 📝 Crear cuenta y generar API keys
- ✅ **Ventaja**: Sin dinero real, datos reales de mercado

#### Para Producción:
- 🌐 **URL**: https://www.binance.com/en/my/settings/api-management
- ⚠️ **CUIDADO**: Maneja dinero real
- 🔒 **Permisos requeridos**: Lectura de cuenta, lectura de historial

### 3. 🔧 Permisos de API Keys
Asegúrate de habilitar:
- ✅ **Read Info** (Lectura de información)
- ✅ **Enable Spot & Margin Trading** (Solo si quieres hacer trading)
- ❌ **Enable Futures** (No necesario)
- ❌ **Enable Withdrawals** (No recomendado)

---

## 🚀 Ejecución de Sincronización

### Paso 1: Verificar Configuración
```bash
python run_historical_sync.py
```

### Paso 2: Ejecutar Sincronización
El script te guiará a través de:
1. 🔍 Verificación de API keys
2. ⚙️ Configuración de parámetros
3. 🔄 Sincronización automática
4. 📊 Reporte de resultados

### Paso 3: Verificar Resultados
```bash
python verify_sync_results.py
```

---

## 📊 Qué se Sincroniza

### 💰 Balance de Cuenta
- Balance actual en USDT
- Assets disponibles y bloqueados
- Información de la cuenta

### 📈 Historial de Trades
- Trades de los últimos 30 días (configurable)
- Símbolos: BTCUSDT, ETHUSDT, BNBUSDT
- Precios de entrada, cantidades, comisiones
- Timestamps y metadatos

### 📊 Métricas Calculadas
- PnL histórico
- Volumen de trading
- Comisiones pagadas
- Distribución por símbolo

---

## ⚙️ Configuración Avanzada

### Cambiar Símbolos a Sincronizar
Edita `run_historical_sync.py`:
```python
trading_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
```

### Cambiar Período Histórico
```python
days_back = 60  # Últimos 60 días
```

### Cambiar a Producción
```python
use_testnet = False  # ⚠️ CUIDADO: Ambiente real
```

---

## 🗃️ Base de Datos Resultante

### Tablas Principales:
- **`trades`**: Historial completo de operaciones
- **`performance_metrics`**: Métricas de rendimiento
- **`system_logs`**: Logs de sincronización
- **`market_data_cache`**: Cache de datos de mercado

### Consultas Útiles:
```sql
-- Ver todos los trades
SELECT * FROM trades ORDER BY entry_time DESC;

-- Balance actual
SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT 1;

-- Trades por símbolo
SELECT symbol, COUNT(*) as count FROM trades GROUP BY symbol;
```

---

## 🔍 Verificación de Datos

### Script de Verificación
```bash
python verify_sync_results.py
```

### Verificaciones Incluidas:
- ✅ Conteo de registros por tabla
- ✅ Análisis de trades por símbolo
- ✅ Verificación de calidad de datos
- ✅ Historial de balance
- ✅ Logs de sincronización

---

## 🚨 Solución de Problemas

### Error: "API keys no configuradas"
```bash
# Verificar variables de entorno
echo $BINANCE_API_KEY

# Verificar archivo .env
cat .env
```

### Error: "Unauthorized" (401)
- ✅ Verificar que las API keys sean correctas
- ✅ Verificar que los permisos estén habilitados
- ✅ Para testnet, usar keys de testnet.binance.vision

### Error: "IP not whitelisted"
- ✅ En Binance, agregar tu IP a la whitelist
- ✅ O deshabilitar restricción de IP

### No se encuentran trades históricos
- ✅ Verificar que hayas hecho trades en el período seleccionado
- ✅ Verificar que uses el ambiente correcto (testnet vs producción)

---

## 📈 Próximos Pasos

Después de la sincronización exitosa:

1. ✅ **Verificar datos**: `python verify_sync_results.py`
2. 🔄 **Configurar trading automático**: Usar datos como baseline
3. 📊 **Análisis de performance**: Evaluar estrategias históricas
4. 🤖 **Activar bot**: Con datos históricos como referencia

---

## 🔒 Mejores Prácticas de Seguridad

1. **Nunca** compartas tus API keys
2. **Siempre** usa testnet para pruebas
3. **Limita** permisos de API al mínimo necesario
4. **Revisa** regularmente el acceso de APIs
5. **Usa** .env files (nunca hardcodees keys)
6. **Agrega** .env al .gitignore

---

## 📞 Soporte

Si encuentras problemas:
1. 🔍 Revisa los logs en `trading_bot.db`
2. 📊 Ejecuta `verify_sync_results.py`
3. 🔧 Verifica configuración de API keys
4. 📝 Revisa los mensajes de error en consola
