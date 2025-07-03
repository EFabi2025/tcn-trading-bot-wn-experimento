#!/usr/bin/env python3
"""
🏛️ Filtro de Régimen de Mercado Profesional
============================================

Este módulo implementa un sistema avanzado para determinar el régimen de mercado
basado en un consenso de múltiples indicadores y pares de trading. El objetivo es
identificar con alta precisión si el mercado se encuentra en una fase alcista,
bajista o neutral, para adaptar las estrategias de trading y gestionar el riesgo.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Importaciones del proyecto
from real_binance_predictor import BinanceDataProvider
from config import trading_config

# Constantes de configuración (podrían moverse a config.py si se usan en otros lugares)
TRADING_SYMBOLS = trading_config.TRADING_SYMBOLS
MARKET_REGIME_TIMEFRAME = '5m'
MARKET_REGIME_DATA_LIMIT = 600  # Puntos de datos para 5m (~2 días)

class MarketRegimeFilter:
    """
    Determina el régimen de mercado mediante un análisis de consenso profesional
    a través de múltiples activos e indicadores.
    """
    def __init__(self, data_provider: BinanceDataProvider, logger: logging.Logger):
        self.data_provider = data_provider
        self.logger = logger
        self.trading_symbols = TRADING_SYMBOLS
        self.timeframe = MARKET_REGIME_TIMEFRAME
        self.limit = MARKET_REGIME_DATA_LIMIT

    async def get_market_regime(self) -> Tuple[str, float, Dict]:
        """
        Calcula el régimen de mercado general basado en un consenso de los pares de trading.

        Returns:
            Tuple[str, float, Dict]: Una tupla con el régimen final ('BULLISH', 'BEARISH', 'NEUTRAL'),
                                     la confianza del consenso, y un diccionario con detalles.
        """
        self.logger.info("🏛️  Iniciando análisis de régimen de mercado profesional...")
        
        pair_regimes = {}

        # 1. Analizar cada par de trading individualmente
        for symbol in self.trading_symbols:
            try:
                regime_details = await self._analyze_pair_regime(symbol)
                if regime_details:
                    pair_regimes[symbol] = regime_details
            except Exception as e:
                self.logger.error(f"❌ Error analizando régimen para {symbol}: {e}")

        if not pair_regimes:
            self.logger.warning("No se pudo analizar el régimen para ningún par. Asumiendo NEUTRAL.")
            return 'NEUTRAL', 0.0, {}

        # 2. Consolidar los resultados para un consenso final
        final_regime, confidence, consensus_details = self._get_consensus_regime(pair_regimes)

        self.logger.info(f"🏛️  Régimen de Mercado Final: {final_regime} (Confianza: {confidence:.2%})")
        if final_regime == 'BEARISH':
            self.logger.warning(f"🚨 ¡MERCADO BAJISTA DETECTADO! Operaciones de compra limitadas. 🚨")
        
        return final_regime, confidence, consensus_details

    async def _analyze_pair_regime(self, symbol: str) -> Dict:
        """
        Analiza un único par para determinar su régimen de mercado local.
        """
        klines = await self.data_provider.get_klines(symbol, self.timeframe, self.limit)
        if not klines or len(klines) < 577: # Mínimo para trend de 2d (576 periodos)
            self.logger.warning(f"Datos insuficientes para {symbol} para el análisis de régimen completo, saltando.")
            return None

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])

        # --- Calcular indicadores ---
        # 1. Momentum (4h, 12h, 24h)
        df['momentum_4h'] = df['close'].pct_change(48)   # 4h = 48 * 5m
        df['momentum_12h'] = df['close'].pct_change(144) # 12h = 144 * 5m
        df['momentum_24h'] = df['close'].pct_change(288) # 24h = 288 * 5m

        # 2. Tendencia reciente de 2 días
        df['recent_trend_2d'] = df['close'].pct_change(576) # 48h = 576 * 5m

        # 3. EMA Trend
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_trend'] = (df['close'] - df['ema_20']) / df['ema_20']
        
        latest = df.iloc[-1]
        bullish_count = 0
        bearish_count = 0

        # --- Lógica de votación con pesos ---
        # Momentum 4h
        if latest['momentum_4h'] > 0.01: bullish_count += 2
        if latest['momentum_4h'] < -0.01: bearish_count += 3 # Más peso a caídas
        
        # Momentum 12h
        if latest['momentum_12h'] > 0.025: bullish_count += 3
        if latest['momentum_12h'] < -0.025: bearish_count += 4 # Más peso
        
        # Momentum 24h
        if latest['momentum_24h'] > 0.03: bullish_count += 3
        if latest['momentum_24h'] < -0.03: bearish_count += 4
        
        # Tendencia 2 días
        if latest['recent_trend_2d'] > 0.04: bullish_count += 4
        if latest['recent_trend_2d'] < -0.04: bearish_count += 5 # Peso extra a bajistas

        # EMA Trend
        if latest['ema_trend'] > 0.01: bullish_count += 1
        if latest['ema_trend'] < -0.01: bearish_count += 2

        # --- Determinar régimen del par ---
        if bearish_count >= bullish_count:
            pair_regime = 'BEARISH'
        elif bullish_count > bearish_count + 1:
            pair_regime = 'BULLISH'
        else:
            pair_regime = 'NEUTRAL'
            
        return {
            'regime': pair_regime,
            'bullish_score': bullish_count,
            'bearish_score': bearish_count,
            'recent_trend_2d': latest['recent_trend_2d']
        }

    def _get_consensus_regime(self, pair_regimes: Dict) -> Tuple[str, float, Dict]:
        """
        Consolida los regímenes de pares individuales en un régimen de mercado final.
        """
        regime_votes = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
        total_pairs = len(pair_regimes)

        for symbol, details in pair_regimes.items():
            regime_votes[details['regime']] += 1

        # Calcular ratios y fuerza del consenso
        bullish_ratio = regime_votes['BULLISH'] / total_pairs
        bearish_ratio = regime_votes['BEARISH'] / total_pairs
        consensus_strength = max(bullish_ratio, bearish_ratio) if total_pairs > 0 else 0

        final_regime = 'NEUTRAL'
        confidence = 0.5

        # --- Lógica de decisión de consenso ---
        # Condición BEARISH (más sensible)
        if (regime_votes['BEARISH'] >= regime_votes['BULLISH'] and
            bearish_ratio > 0.45 and consensus_strength > 0.4):
            final_regime = 'BEARISH'
            confidence = consensus_strength * bearish_ratio

        # Condición BULLISH (requiere más confirmación)
        elif (regime_votes['BULLISH'] > regime_votes['BEARISH'] and
              regime_votes['BULLISH'] > regime_votes['NEUTRAL'] and
              bullish_ratio > 0.5):
            final_regime = 'BULLISH'
            confidence = consensus_strength * bullish_ratio
        
        # --- Override automático por tendencia bajista ---
        # Asegurarse de que hay valores válidos para calcular la media
        valid_trends = [data['recent_trend_2d'] for data in pair_regimes.values() if 'recent_trend_2d' in data and pd.notna(data['recent_trend_2d'])]
        if valid_trends:
            avg_trend_2d = np.mean(valid_trends)
            if pd.notna(avg_trend_2d) and avg_trend_2d < -0.03 and final_regime == 'NEUTRAL':
                final_regime = 'BEARISH'
                confidence = 0.75  # Asignar confianza alta por ser un override de seguridad
                self.logger.warning(f"OVERRIDE: Tendencia promedio de 2 días ({avg_trend_2d:.2%}) es muy negativa. Forzando régimen a BEARISH.")
        else:
            avg_trend_2d = np.nan


        details = {
            'votes': regime_votes,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'consensus_strength': consensus_strength,
            'avg_trend_2d': avg_trend_2d,
            'individual_regimes': {s: d['regime'] for s, d in pair_regimes.items()}
        }

        return final_regime, confidence, details 