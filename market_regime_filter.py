#!/usr/bin/env python3
"""
🏛️ Filtro de Régimen de Mercado
================================

Este módulo analiza la condición general del mercado utilizando un activo de referencia
(como BTCUSDT) para determinar el "régimen" actual. Actúa como una capa de
seguridad para evitar operaciones en condiciones de mercado desfavorables.
"""

import logging
from enum import Enum
import pandas as pd
from ta.trend import ema_indicator, adx
from ta.volatility import average_true_range

# Importar desde el proyecto actual
from real_binance_predictor import BinanceDataProvider
from config import trading_config

class MarketRegime(Enum):
    """Define los posibles estados del régimen de mercado."""
    BULLISH = "BULLISH"          # Tendencia alcista clara
    BEARISH = "BEARISH"          # Tendencia bajista clara
    RANGING = "RANGING"          # Mercado lateral, sin tendencia definida
    HIGH_VOLATILITY = "HIGH_VOLATILITY" # Movimientos bruscos, pánico o euforia

class MarketRegimeFilter:
    """
    Clase que determina el régimen de mercado actual para una gestión de
    riesgo a nivel macro.
    """
    def __init__(self, data_provider: BinanceDataProvider, logger: logging.Logger):
        """
        Inicializa el filtro de régimen de mercado.

        Args:
            data_provider (BinanceDataProvider): Instancia para obtener datos de mercado.
            logger (logging.Logger): Logger para registrar información y errores.
        """
        self.data_provider = data_provider
        self.logger = logger
        self.config = trading_config
        self.symbol = self.config.MARKET_REGIME_SYMBOL
        self.timeframe = self.config.MARKET_REGIME_TIMEFRAME
        self.logger.info(
            f"🏛️ Filtro de Régimen de Mercado inicializado para {self.symbol} "
            f"en temporalidad de {self.timeframe}."
        )

    async def get_market_regime(self) -> tuple[MarketRegime, dict]:
        """
        Analiza los datos del mercado y devuelve el régimen actual.

        Returns:
            tuple[MarketRegime, dict]: Una tupla con el régimen de mercado
                                       y un diccionario con detalles del análisis.
        """
        try:
            # 1. Obtener datos históricos (klines)
            klines = await self.data_provider.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=self.config.MARKET_REGIME_EMA_LONG + 50 # Datos suficientes para EMAs
            )
            if not klines or len(klines) < self.config.MARKET_REGIME_EMA_LONG:
                self.logger.warning(f"No se pudieron obtener suficientes datos para {self.symbol} en {self.timeframe}.")
                return MarketRegime.RANGING, {"reason": "Datos insuficientes"}

            # 2. Convertir a DataFrame de Pandas
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # 3. Calcular Indicadores Técnicos
            # Medias Móviles Exponenciales (EMAs)
            ema_short = ema_indicator(df['close'], window=self.config.MARKET_REGIME_EMA_SHORT)
            ema_long = ema_indicator(df['close'], window=self.config.MARKET_REGIME_EMA_LONG)

            # Average True Range (ATR) para volatilidad
            atr = average_true_range(df['high'], df['low'], df['close'], window=self.config.MARKET_REGIME_ATR_PERIOD)
            
            # Obtener los últimos valores
            last_close = df['close'].iloc[-1]
            last_ema_short = ema_short.iloc[-1]
            last_ema_long = ema_long.iloc[-1]
            last_atr_percentage = (atr.iloc[-1] / last_close) * 100

            # 4. Determinar el Régimen de Mercado
            
            # Chequeo de alta volatilidad (prioritario)
            volatility_threshold = self.config.MARKET_REGIME_ATR_MULTIPLIER
            if last_atr_percentage > volatility_threshold:
                details = {"reason": f"ATR ({last_atr_percentage:.2f}%) > Umbral ({volatility_threshold:.2f}%)"}
                self.logger.warning(f"Régimen detectado: ALTA VOLATILIDAD para {self.symbol}. {details['reason']}")
                return MarketRegime.HIGH_VOLATILITY, details
            
            # Chequeo de tendencia (Bullish/Bearish)
            is_bullish = last_ema_short > last_ema_long and last_close > last_ema_short
            is_bearish = last_ema_short < last_ema_long and last_close < last_ema_short

            if is_bullish:
                details = {"reason": f"EMA{self.config.MARKET_REGIME_EMA_SHORT} > EMA{self.config.MARKET_REGIME_EMA_LONG}"}
                return MarketRegime.BULLISH, details
            
            if is_bearish:
                details = {"reason": f"EMA{self.config.MARKET_REGIME_EMA_SHORT} < EMA{self.config.MARKET_REGIME_EMA_LONG}"}
                return MarketRegime.BEARISH, details

            # Si no hay tendencia clara, es un mercado en rango
            details = {"reason": "Sin una tendencia clara definida por las EMAs."}
            return MarketRegime.RANGING, details

        except Exception as e:
            self.logger.error(f"❌ Error al determinar el régimen de mercado: {e}", exc_info=True)
            return MarketRegime.RANGING, {"reason": "Error en análisis", "error": str(e)} 