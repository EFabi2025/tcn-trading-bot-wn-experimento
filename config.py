"""
⚙️ CONFIGURACIÓN CENTRALIZADA PARA EL TRADING BOT
Carga y valida todos los parámetros de configuración desde variables de entorno.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class TradingConfig:
    """
    Encapsula toda la configuración del bot, cargada desde variables de entorno
    con valores por defecto seguros.
    """
    def __init__(self):
        # --- Configuración de API Binance ---
        self.BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
        self.BINANCE_SECRET_KEY: str = os.getenv('BINANCE_SECRET_KEY', '')
        self.BINANCE_BASE_URL: str = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'production')

        # --- Estrategia de Trading y Símbolos ---
        symbols_str = os.getenv('TRADING_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT')
        self.TRADING_SYMBOLS: list[str] = [s.strip().upper() for s in symbols_str.split(',')]

        # --- Parámetros de Riesgo ---
        self.MAX_POSITION_SIZE_PERCENT: float = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '15.0'))
        self.MAX_TOTAL_EXPOSURE_PERCENT: float = float(os.getenv('MAX_TOTAL_EXPOSURE_PERCENT', '40.0'))
        self.MAX_DAILY_LOSS_PERCENT: float = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10.0'))
        self.MAX_DRAWDOWN_PERCENT: float = float(os.getenv('MAX_DRAWDOWN_PERCENT', '15.0'))
        self.STOP_LOSS_PERCENT: float = float(os.getenv('STOP_LOSS_PERCENT', '3.0'))
        self.TAKE_PROFIT_PERCENT: float = float(os.getenv('TAKE_PROFIT_PERCENT', '6.0'))
        self.MIN_POSITION_VALUE_USDT: float = float(os.getenv('MIN_POSITION_VALUE_USDT', '11.0'))
        self.MAX_CONCURRENT_POSITIONS: int = int(os.getenv('MAX_CONCURRENT_POSITIONS', '3'))

        # --- Umbrales del Modelo TCN ---
        self.TCN_BUY_CONFIDENCE_THRESHOLD: float = float(os.getenv('TCN_BUY_CONFIDENCE_THRESHOLD', '0.75'))
        self.TCN_SELL_CONFIDENCE_THRESHOLD: float = float(os.getenv('TCN_SELL_CONFIDENCE_THRESHOLD', '0.70'))

        # --- Configuración del Manager ---
        self.CHECK_INTERVAL_SECONDS: int = int(os.getenv('CHECK_INTERVAL_SECONDS', '60'))
        self.MONITORING_INTERVAL_SECONDS: int = int(os.getenv('MONITORING_INTERVAL_SECONDS', '30'))
        self.HEARTBEAT_INTERVAL_SECONDS: int = int(os.getenv('HEARTBEAT_INTERVAL_SECONDS', '300'))
        
        # --- Configuración de Notificaciones Discord ---
        self.DISCORD_WEBHOOK_URL: str = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.DISCORD_MIN_TRADE_VALUE_USD: float = float(os.getenv('DISCORD_MIN_TRADE_VALUE_USD', '12.0'))
        self.DISCORD_MIN_PNL_PERCENT_NOTIFY: float = float(os.getenv('DISCORD_MIN_PNL_PERCENT_NOTIFY', '2.0'))
        self.DISCORD_MAX_NOTIFICATIONS_PER_HOUR: int = int(os.getenv('DISCORD_MAX_NOTIFICATIONS_PER_HOUR', '8'))
        self.DISCORD_SUPPRESS_SIMILAR_MINUTES: int = int(os.getenv('DISCORD_SUPPRESS_SIMILAR_MINUTES', '10'))

        self.MAX_NOTIFICATIONS_PER_HOUR: int = int(os.getenv('MAX_NOTIFICATIONS_PER_HOUR', '8'))

        self.validate()

    def validate(self):
        """Valida que los parámetros críticos estén presentes y sean lógicos."""
        if not self.BINANCE_API_KEY or not self.BINANCE_SECRET_KEY:
            print("⚠️ ADVERTENCIA: Las claves de API de Binance no están configuradas. El bot no podrá ejecutar órdenes reales.")

        if not self.TRADING_SYMBOLS:
            raise ValueError("La variable de entorno TRADING_SYMBOLS no puede estar vacía.")
            
        if not 0 < self.MAX_POSITION_SIZE_PERCENT <= 100:
            raise ValueError("MAX_POSITION_SIZE_PERCENT debe estar entre 0 y 100.")
            
        if not 0 < self.TCN_BUY_CONFIDENCE_THRESHOLD < 1:
            raise ValueError("TCN_BUY_CONFIDENCE_THRESHOLD debe ser un valor entre 0 y 1.")

        print("✅ Configuración cargada y validada exitosamente.")

# Crear una instancia única para ser importada en otros módulos
trading_config = TradingConfig() 