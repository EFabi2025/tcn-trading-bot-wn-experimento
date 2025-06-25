# 📄 CAMPOS REQUERIDOS PARA ARCHIVO .env

## 🎯 **CONFIGURACIÓN ACTUALIZADA CON UMBRALES 50%**

Crea/edita tu archivo `.env` con estos campos:

```bash
# 🔑 BINANCE API CREDENTIALS (OBLIGATORIO)
# ========================================
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui

# 🌐 BINANCE ENVIRONMENT
# ======================
BINANCE_BASE_URL=https://testnet.binance.vision
# BINANCE_BASE_URL=https://api.binance.com  # Para producción
ENVIRONMENT=testnet

# 🎯 UMBRALES TCN (ACTUALIZADOS A 50% PAREJO)
# ===========================================
TCN_BUY_CONFIDENCE_THRESHOLD=0.50
TCN_SELL_CONFIDENCE_THRESHOLD=0.50

# 📊 TRADING SYMBOLS
# ==================
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT

# 🛡️ GESTIÓN DE RIESGO
# =====================
MAX_POSITION_SIZE_PERCENT=15.0
MAX_TOTAL_EXPOSURE_PERCENT=40.0
MAX_DAILY_LOSS_PERCENT=10.0
MAX_DRAWDOWN_PERCENT=15.0
STOP_LOSS_PERCENT=3.0
TAKE_PROFIT_PERCENT=6.0
MIN_POSITION_VALUE_USDT=11.0
MAX_CONCURRENT_POSITIONS=3

# 🔔 DISCORD NOTIFICATIONS (OPCIONAL)
# ===================================
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/TU_WEBHOOK_ID/TU_WEBHOOK_TOKEN
DISCORD_REPORT_INTERVAL_SECONDS=180
DISCORD_MIN_TRADE_VALUE_USD=12.0
DISCORD_MIN_PNL_PERCENT_NOTIFY=2.0
DISCORD_MAX_NOTIFICATIONS_PER_HOUR=8
DISCORD_SUPPRESS_SIMILAR_MINUTES=10

# ⚙️ SISTEMA
# ===========
CHECK_INTERVAL_SECONDS=60
MONITORING_INTERVAL_SECONDS=30
HEARTBEAT_INTERVAL_SECONDS=300

# 💙 FILTRO DE RÉGIMEN DE MERCADO
# ===============================
ENABLE_MARKET_REGIME_FILTER=true
MARKET_REGIME_SYMBOL=BTCUSDT
MARKET_REGIME_TIMEFRAME=4h
MARKET_REGIME_EMA_SHORT=21
MARKET_REGIME_EMA_LONG=55
MARKET_REGIME_ATR_PERIOD=14
MARKET_REGIME_ATR_MULTIPLIER=1.5
```

## 🎯 **CAMPOS CRÍTICOS ACTUALIZADOS**

### ⚡ **MÁS IMPORTANTES (Obligatorios)**
```bash
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
TCN_BUY_CONFIDENCE_THRESHOLD=0.50    # ⭐ NUEVO: Era 0.75
TCN_SELL_CONFIDENCE_THRESHOLD=0.50   # ⭐ NUEVO: Era 0.70
```

### 📊 **CONFIGURACIÓN BÁSICA**
```bash
BINANCE_BASE_URL=https://testnet.binance.vision
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
MAX_POSITION_SIZE_PERCENT=15.0
MAX_CONCURRENT_POSITIONS=3
```

### 🔔 **NOTIFICACIONES (Opcional pero recomendado)**
```bash
DISCORD_WEBHOOK_URL=tu_webhook_discord_aqui
DISCORD_REPORT_INTERVAL_SECONDS=180
```

## 🚀 **PARA EMPEZAR RÁPIDO**

Si quieres el mínimo absoluto para probar:

```bash
# Archivo .env mínimo
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
BINANCE_BASE_URL=https://testnet.binance.vision
TCN_BUY_CONFIDENCE_THRESHOLD=0.50
TCN_SELL_CONFIDENCE_THRESHOLD=0.50
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
```

## ⚠️ **NOTAS IMPORTANTES**

1. **🔑 API Keys**: Obtén las tuyas desde [Binance API Management](https://binance.com/en/my/settings/api-management)

2. **🧪 Testnet**: Usa `https://testnet.binance.vision` para pruebas seguras

3. **🎯 Umbrales 50%**: Los nuevos umbrales permiten señales más ejecutables gracias a las features híbridas

4. **📱 Discord**: Crea un webhook en tu servidor para recibir notificaciones

5. **🛡️ Seguridad**: NUNCA compartas tu archivo `.env` con nadie

## 📋 **VERIFICACIÓN**

Para verificar que tu `.env` funciona:
```bash
python test_simple_hybrid.py
```

Debería mostrar:
- ✅ Modelos cargados
- 🎯 Umbrales al 50%
- 📊 Predicciones con features híbridas limpias 