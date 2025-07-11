# 🔐 CONFIGURACIÓN DE API KEYS DE BINANCE

## 📋 Pasos para Configurar Trading Real

### 1️⃣ **Crear API Keys en Binance**

#### Para Testnet (Recomendado para Pruebas):
1. Ve a: https://testnet.binance.vision/
2. Regístrate con tu email
3. Ve a **API Management**
4. Crea una nueva API Key
5. Guarda tu **API Key** y **Secret Key**

#### Para Trading Real (¡CUIDADO!):
1. Ve a: https://binance.com
2. Inicia sesión en tu cuenta
3. Ve a **Account** → **API Management**
4. Crea una nueva API Key
5. **IMPORTANTE**: Configura restricciones de IP y permisos

### 2️⃣ **Configurar Archivo .env**

```bash
# Copia el archivo de ejemplo
cp env_example .env

# Edita con tus credenciales reales
nano .env
```

### 3️⃣ **Ejemplo de Archivo .env Configurado**

```env
# 🚨 API KEYS DE BINANCE (REEMPLAZAR)
BINANCE_API_KEY=tu_api_key_real_aqui
BINANCE_SECRET_KEY=tu_secret_key_real_aqui

# 🌍 ENTORNO (testnet para pruebas, production para real)
ENVIRONMENT=testnet

# 🔗 URLs DE API
BINANCE_TESTNET_URL=https://testnet.binance.vision
BINANCE_PRODUCTION_URL=https://api.binance.com

# 💬 DISCORD WEBHOOK (tu webhook configurado)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/12345/abcdef

# 📊 CONFIGURACIÓN DE TRADING
MAX_POSITION_SIZE_PERCENT=5      # Máximo 5% del balance por trade
MAX_DAILY_LOSS_PERCENT=3         # Máximo 3% de pérdida diaria
MIN_CONFIDENCE_THRESHOLD=0.70    # Mínimo 70% de confianza
TRADE_MODE=dry_run               # dry_run = simulación, real = dinero real

# 🔔 NOTIFICACIONES
ENABLE_DISCORD_NOTIFICATIONS=true
NOTIFICATION_COOLDOWN_MINUTES=3

# 🗄️ BASE DE DATOS
DATABASE_URL=sqlite:///database/trading_bot.db

# 📝 LOGGING
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
```

### 4️⃣ **Configuración Segura para Producción**

#### Permisos de API Key (Solo para Production):
- ✅ **Spot & Margin Trading**: Enable
- ✅ **Futures**: Disable (si no usas futuros)
- ✅ **Read Info**: Enable
- ❌ **Enable Withdrawals**: NEVER
- ❌ **Enable Internal Transfer**: NEVER

#### Restricciones de IP:
- Configura tu IP específica
- NUNCA uses "Unrestricted"

### 5️⃣ **Niveles de Seguridad**

#### 🟢 **Modo Seguro (Recomendado)**:
```env
ENVIRONMENT=testnet
TRADE_MODE=dry_run
MAX_POSITION_SIZE_PERCENT=2
```

#### 🟡 **Modo Testnet Real**:
```env
ENVIRONMENT=testnet
TRADE_MODE=real
MAX_POSITION_SIZE_PERCENT=5
```

#### 🔴 **Modo Producción (¡PELIGROSO!)**:
```env
ENVIRONMENT=production
TRADE_MODE=real
MAX_POSITION_SIZE_PERCENT=3
```

### 6️⃣ **Comandos para Ejecutar**

```bash
# Verificar configuración
python real_trading_setup.py

# Opción 1: Verificar configuración
# Opción 2: Test de conexión a Binance
# Opción 3: Trading 30 minutos
# Opción 4: Trading 60 minutos
# Opción 5: Monitoreo continuo
```

### 7️⃣ **⚠️ ADVERTENCIAS IMPORTANTES**

#### 🚨 **NUNCA HAGAS ESTO**:
- ❌ Subir tu archivo `.env` a GitHub
- ❌ Compartir tus API keys
- ❌ Usar modo `production` sin entender los riesgos
- ❌ Configurar withdrawals en tu API key
- ❌ Usar IP "Unrestricted"

#### ✅ **SIEMPRE HACER**:
- ✅ Empezar con `testnet`
- ✅ Usar `dry_run` para pruebas
- ✅ Configurar límites conservadores
- ✅ Monitorear todas las operaciones
- ✅ Mantener API keys seguras

### 8️⃣ **Estructura de Archivos**

```
proyecto/
├── .env                    # TUS CREDENCIALES (NO SUBIR A GIT)
├── env_example            # PLANTILLA PÚBLICA
├── real_trading_setup.py  # SISTEMA DE TRADING REAL
├── production_model_*.h5  # MODELOS TCN ENTRENADOS
└── config/
    └── trading_config.json
```

### 9️⃣ **Flujo Recomendado**

1. **Configurar Testnet** → Probar todas las funciones
2. **Verificar Modelos** → Asegurar TCN funcionan
3. **Modo Dry-Run** → Simular trades sin dinero
4. **Testnet Real** → Trades reales con dinero de prueba
5. **Producción** → Solo cuando estés 100% seguro

### 🔟 **Soporte y Debugging**

Si hay errores:
1. Verificar que `.env` existe
2. Verificar que API keys son correctas
3. Verificar permisos de API key
4. Verificar conexión a internet
5. Revisar logs para detalles

## 🚀 ¡LISTO PARA TRADING REAL!

Una vez configurado tu `.env`, ejecuta:
```bash
python real_trading_setup.py
```

Y selecciona la opción que desees. ¡El sistema usará tus API keys reales de Binance!
