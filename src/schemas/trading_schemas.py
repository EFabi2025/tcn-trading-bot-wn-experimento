"""
Esquemas Pydantic para validación de datos de trading.

Garantiza la integridad y validación estricta de todos los datos
que fluyen por el sistema de trading.
"""
from pydantic import BaseModel, field_validator, Field
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime
from enum import Enum


class OrderSideSchema(str, Enum):
    """Esquema para tipos de orden."""
    BUY = "BUY"
    SELL = "SELL"


class OrderTypeSchema(str, Enum):
    """Esquema para tipos de orden."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"


class TradingConfigSchema(BaseModel):
    """Configuración principal del trading bot."""
    
    # Configuración de Binance
    binance_api_key: str = Field(..., min_length=1)
    binance_secret: str = Field(..., min_length=1)
    testnet: bool = Field(default=True)
    
    # Configuración de trading
    symbols: List[str] = Field(default=["BTCUSDT"], min_items=1)
    base_asset: str = Field(default="USDT")
    max_position_percent: float = Field(default=0.02, ge=0.001, le=0.1)  # 2% max
    min_order_amount: Decimal = Field(default=Decimal("10"), gt=0)
    
    # Configuración de riesgo
    max_daily_loss_percent: float = Field(default=0.05, ge=0.01, le=0.2)  # 5% max
    stop_loss_percent: float = Field(default=0.02, ge=0.005, le=0.1)  # 2%
    take_profit_percent: float = Field(default=0.04, ge=0.01, le=0.2)  # 4%
    
    # Configuración del modelo
    model_confidence_threshold: float = Field(default=0.7, ge=0.5, le=0.95)
    prediction_interval_minutes: int = Field(default=5, ge=1, le=60)
    
    # Configuración de logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_to_file: bool = Field(default=True)
    
    # Configuración de notificaciones
    enable_notifications: bool = Field(default=False)
    webhook_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        use_enum_values = True
    
    @field_validator('symbols')
    @classmethod
    def symbols_must_be_valid(cls, v):
        """Valida que los símbolos sean válidos."""
        for symbol in v:
            if not symbol.endswith('USDT'):
                raise ValueError(f'Symbol {symbol} must end with USDT')
            if len(symbol) < 6:
                raise ValueError(f'Symbol {symbol} too short')
        return v
    
    @field_validator('max_position_percent')
    @classmethod
    def position_percent_validation(cls, v):
        """Valida porcentaje de posición."""
        if v > 0.1:  # No más del 10%
            raise ValueError('Position percent cannot exceed 10%')
        return v


class OrderRequestSchema(BaseModel):
    """Esquema para solicitudes de orden."""
    
    symbol: str = Field(..., min_length=6, max_length=20)
    side: OrderSideSchema
    quantity: Decimal = Field(..., gt=0, decimal_places=8)
    order_type: OrderTypeSchema = Field(default=OrderTypeSchema.MARKET)
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=8)
    stop_price: Optional[Decimal] = Field(None, gt=0, decimal_places=8)
    time_in_force: str = Field(default="GTC")
    
    @field_validator('symbol')
    @classmethod
    def symbol_must_be_valid(cls, v):
        """Valida formato del símbolo."""
        if not v.endswith('USDT'):
            raise ValueError('Only USDT pairs allowed')
        return v.upper()
    
    @field_validator('quantity')
    @classmethod
    def quantity_must_be_positive(cls, v):
        """Valida que la cantidad sea positiva."""
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v


class MarketDataSchema(BaseModel):
    """Esquema para datos de mercado."""
    
    symbol: str = Field(..., min_length=6)
    price: Decimal = Field(..., gt=0, decimal_places=8)
    volume: Decimal = Field(..., ge=0, decimal_places=8)
    timestamp: datetime
    bid: Optional[Decimal] = Field(None, gt=0, decimal_places=8)
    ask: Optional[Decimal] = Field(None, gt=0, decimal_places=8)
    high_24h: Optional[Decimal] = Field(None, gt=0, decimal_places=8)
    low_24h: Optional[Decimal] = Field(None, gt=0, decimal_places=8)
    
    # Validación bid/ask removida por simplicidad en pydantic v2


class TradingSignalSchema(BaseModel):
    """Esquema para señales de trading."""
    
    symbol: str = Field(..., min_length=6)
    action: OrderSideSchema
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_price: Decimal = Field(..., gt=0, decimal_places=8)
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('confidence')
    @classmethod
    def confidence_validation(cls, v):
        """Valida el nivel de confianza."""
        if v < 0.5:
            raise ValueError('Confidence must be at least 0.5')
        return v
    
    @field_validator('symbol')
    @classmethod
    def symbol_validation(cls, v):
        """Valida el símbolo."""
        if not v.endswith('USDT'):
            raise ValueError('Only USDT pairs supported')
        return v.upper()


class BalanceSchema(BaseModel):
    """Esquema para balance de cuenta."""
    
    asset: str = Field(..., min_length=1, max_length=10)
    free: Decimal = Field(..., ge=0, decimal_places=8)
    locked: Decimal = Field(..., ge=0, decimal_places=8)
    
    @property
    def total(self) -> Decimal:
        """Balance total."""
        return self.free + self.locked
    
    @field_validator('asset')
    @classmethod
    def asset_validation(cls, v):
        """Valida el asset."""
        return v.upper()


class OrderSchema(BaseModel):
    """Esquema para órdenes ejecutadas."""
    
    id: str = Field(..., min_length=1)
    symbol: str = Field(..., min_length=6)
    side: OrderSideSchema
    quantity: Decimal = Field(..., gt=0, decimal_places=8)
    price: Decimal = Field(..., gt=0, decimal_places=8)
    status: str
    timestamp: datetime
    filled_quantity: Decimal = Field(default=Decimal('0'), ge=0, decimal_places=8)
    commission: Decimal = Field(default=Decimal('0'), ge=0, decimal_places=8)
    
    # Validación filled_quantity removida por simplicidad en pydantic v2


class RiskParametersSchema(BaseModel):
    """Esquema para parámetros de riesgo."""
    
    max_position_size_percent: float = Field(..., ge=0.001, le=0.1)
    stop_loss_percent: float = Field(..., ge=0.005, le=0.1)
    take_profit_percent: float = Field(..., ge=0.01, le=0.2)
    max_daily_loss_percent: float = Field(..., ge=0.01, le=0.2)
    max_open_positions: int = Field(..., ge=1, le=10)
    min_confidence_threshold: float = Field(..., ge=0.5, le=0.95)
    
    # Validación take_profit removida por simplicidad en pydantic v2


class DatabaseConfigSchema(BaseModel):
    """Configuración de base de datos."""
    
    database_url: str = Field(..., min_length=1)
    pool_size: int = Field(default=5, ge=1, le=20)
    max_overflow: int = Field(default=10, ge=1, le=50)
    pool_timeout: int = Field(default=30, ge=5, le=300)
    echo: bool = Field(default=False)
    
    @field_validator('database_url')
    @classmethod
    def database_url_validation(cls, v):
        """Valida URL de base de datos."""
        if not (v.startswith('sqlite://') or v.startswith('postgresql://')):
            raise ValueError('Only SQLite and PostgreSQL supported')
        return v


class NotificationConfigSchema(BaseModel):
    """Configuración de notificaciones."""
    
    enabled: bool = Field(default=False)
    webhook_url: Optional[str] = None
    send_signals: bool = Field(default=True)
    send_trades: bool = Field(default=True)
    send_errors: bool = Field(default=True)
    
    # Validación webhook_url removida por simplicidad en pydantic v2 