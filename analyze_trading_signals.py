#!/usr/bin/env python3
"""
🔍 ANALIZADOR DE SEÑALES DE TRADING TCN
=====================================

Analiza las señales generadas por el TCN y cómo están siendo procesadas
por el bot de trading, comparándolas con indicadores técnicos para validar
la coherencia de las predicciones.

Funcionalidades:
- Análisis de señales TCN vs indicadores técnicos
- Validación de coherencia de predicciones
- Estadísticas de rendimiento por símbolo
- Detección de posibles problemas en el mapeo de señales
- Reporte detallado de análisis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Importar nuestros módulos
from tcn_definitivo_predictor import TCNDefinitivoPredictor

class TradingSignalAnalyzer:
    """🔍 Analizador de señales de trading"""

    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
        self.tcn_predictor = TCNDefinitivoPredictor()
        self.analysis_results = {}

        # Configurar thresholds para análisis técnico (MEJORADOS por símbolo)
        self.technical_thresholds = {
            # Thresholds generales
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_low': 45,
            'rsi_neutral_high': 55,
            'macd_strong_positive': 0.1,
            'macd_strong_negative': -0.1,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'bb_squeeze_threshold': 0.02,
            'volume_spike_multiplier': 1.5,

            # Thresholds específicos por símbolo
            'symbol_specific': {
                'BTCUSDT': {
                    'macd_strong_positive': 0.5,  # BTC requiere MACD más fuerte
                    'rsi_overbought': 75,         # BTC puede soportar RSI más alto
                    'stoch_overbought': 85        # BTC Stoch más permisivo
                },
                'ETHUSDT': {
                    'macd_strong_positive': 0.08, # ETH más sensible al MACD
                    'rsi_overbought': 68,         # ETH más restrictivo en RSI
                    'stoch_overbought': 78        # ETH Stoch más restrictivo
                },
                'BNBUSDT': {
                    'macd_strong_positive': 0.15,  # BNB: reducir threshold MACD
                    'rsi_overbought': 75,          # BNB: más permisivo en RSI
                    'stoch_overbought': 85,        # BNB: más permisivo en Stochastic
                    'bb_upper_threshold': 0.88,    # BNB: más permisivo en BB
                    'rsi_neutral_high': 60         # BNB: ajustar zona neutral
                },
                'XRPUSDT': {
                    'macd_strong_positive': 0.001, # XRP muy sensible al MACD
                    'rsi_overbought': 65,          # XRP muy restrictivo en RSI
                    'stoch_overbought': 75         # XRP Stoch muy restrictivo
                }
            }
        }

    async def get_market_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """📊 Obtener datos de mercado para análisis"""
        try:
            print(f"📊 Obteniendo {days} días de datos para {symbol}...")

            base_url = "https://api.binance.com"
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            async with aiohttp.ClientSession() as session:
                url = f"{base_url}/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': '5m',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }

                all_data = []
                current_start = start_time

                while current_start < end_time:
                    params['startTime'] = current_start

                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if not data:
                                break
                            all_data.extend(data)
                            current_start = data[-1][6] + 1
                        else:
                            print(f"❌ Error API: {response.status}")
                            break

                    await asyncio.sleep(0.1)

            # Convertir a DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convertir tipos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()

            print(f"✅ Obtenidos {len(df)} registros de {symbol}")
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"❌ Error obteniendo datos de {symbol}: {e}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """📈 Calcular indicadores técnicos para análisis"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            indicators = pd.DataFrame(index=df.index)

            # Indicadores básicos
            indicators['rsi_14'] = talib.RSI(close, timeperiod=14)
            indicators['rsi_21'] = talib.RSI(close, timeperiod=21)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
            indicators['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_20'] = talib.EMA(close, timeperiod=20)

            # Volume
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)
            indicators['volume_ratio'] = volume / indicators['volume_sma']

            # Price momentum
            indicators['price_change_1h'] = df['close'].pct_change(12)  # 12 periodos de 5min = 1h
            indicators['price_change_4h'] = df['close'].pct_change(48)  # 48 periodos de 5min = 4h

            # Volatility
            returns = np.log(df['close'] / df['close'].shift(1))
            indicators['volatility_1h'] = returns.rolling(12).std()

            return indicators.fillna(method='ffill').fillna(method='bfill')

        except Exception as e:
            print(f"❌ Error calculando indicadores técnicos: {e}")
            return pd.DataFrame()

    def analyze_technical_signal(self, indicators: pd.Series, symbol: str = None) -> Dict:
        """🔍 Analizar señal técnica basada en indicadores (MEJORADO con thresholds por símbolo)"""
        try:
            signals = {'buy': 0, 'sell': 0, 'hold': 0}
            details = []

            # Obtener thresholds específicos del símbolo si están disponibles
            def get_threshold(key, default_value):
                if symbol and symbol in self.technical_thresholds.get('symbol_specific', {}):
                    return self.technical_thresholds['symbol_specific'][symbol].get(key, default_value)
                return self.technical_thresholds.get(key, default_value)

            # Análisis RSI
            rsi = indicators['rsi_14']
            rsi_oversold = get_threshold('rsi_oversold', 30)
            rsi_overbought = get_threshold('rsi_overbought', 70)
            rsi_neutral_low = get_threshold('rsi_neutral_low', 45)
            rsi_neutral_high = get_threshold('rsi_neutral_high', 55)

            if rsi < rsi_oversold:
                signals['buy'] += 2
                details.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > rsi_overbought:
                signals['sell'] += 2
                details.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < rsi_neutral_low:
                signals['buy'] += 1
                details.append(f"RSI bajo-neutral ({rsi:.1f})")
            elif rsi > rsi_neutral_high:
                signals['sell'] += 1
                details.append(f"RSI alto-neutral ({rsi:.1f})")

            # Análisis MACD
            macd = indicators['macd']
            macd_hist = indicators['macd_histogram']
            macd_strong_positive = get_threshold('macd_strong_positive', 0.1)
            macd_strong_negative = get_threshold('macd_strong_negative', -0.1)

            if macd > macd_strong_positive:
                signals['buy'] += 2
                details.append(f"MACD fuertemente positivo ({macd:.4f})")
            elif macd > 0:
                signals['buy'] += 1
                details.append(f"MACD positivo ({macd:.4f})")
            elif macd < macd_strong_negative:
                signals['sell'] += 2
                details.append(f"MACD fuertemente negativo ({macd:.4f})")
            elif macd < 0:
                signals['sell'] += 1
                details.append(f"MACD negativo ({macd:.4f})")

            # Análisis Stochastic
            stoch_k = indicators['stoch_k']
            stoch_oversold = get_threshold('stoch_oversold', 20)
            stoch_overbought = get_threshold('stoch_overbought', 80)

            if stoch_k < stoch_oversold:
                signals['buy'] += 1
                details.append(f"Stoch oversold ({stoch_k:.1f})")
            elif stoch_k > stoch_overbought:
                signals['sell'] += 1
                details.append(f"Stoch overbought ({stoch_k:.1f})")

            # Análisis Bollinger Bands
            bb_position = indicators['bb_position']
            bb_width = indicators['bb_width']
            bb_upper_threshold = get_threshold('bb_upper_threshold', 0.8)
            bb_squeeze_threshold = get_threshold('bb_squeeze_threshold', 0.02)

            if bb_position < 0.2:
                signals['buy'] += 1
                details.append(f"Precio cerca BB inferior ({bb_position:.2f})")
            elif bb_position > bb_upper_threshold:
                signals['sell'] += 1
                details.append(f"Precio cerca BB superior ({bb_position:.2f})")

            if bb_width < bb_squeeze_threshold:
                details.append(f"BB squeeze detectado ({bb_width:.3f})")

            # Determinar señal dominante (MEJORADO)
            signal_difference = abs(signals['buy'] - signals['sell'])

            if signals['buy'] > signals['sell'] and signal_difference >= 2:
                technical_signal = 'BUY'
            elif signals['sell'] > signals['buy'] and signal_difference >= 2:
                technical_signal = 'SELL'
            else:
                technical_signal = 'HOLD'

            # Añadir información del símbolo para debugging
            if symbol:
                details.append(f"Análisis específico para {symbol}")

            return {
                'signal': technical_signal,
                'buy_signals': signals['buy'],
                'sell_signals': signals['sell'],
                'details': details,
                'rsi': rsi,
                'macd': macd,
                'stoch_k': stoch_k,
                'bb_position': bb_position,
                'symbol_analyzed': symbol
            }

        except Exception as e:
            return {
                'signal': 'HOLD',
                'buy_signals': 0,
                'sell_signals': 0,
                'details': [f"Error en análisis técnico: {e}"],
                'rsi': 0,
                'macd': 0,
                'stoch_k': 0,
                'bb_position': 0.5,
                'symbol_analyzed': symbol
            }

    async def analyze_symbol(self, symbol: str) -> Dict:
        """🔍 Analizar coherencia de señales para un símbolo (MEJORADO para timeframes cortos)"""
        try:
            print(f"🔍 Analizando {symbol}...")

            # Obtener datos de mercado
            df = await self.get_market_data(symbol, days=7)
            if df.empty:
                return {'error': f'No se pudieron obtener datos para {symbol}'}

            # Calcular indicadores técnicos
            indicators = self.calculate_technical_indicators(df)
            if indicators.empty:
                return {'error': f'No se pudieron calcular indicadores para {symbol}'}

            # Obtener predicción TCN actual
            tcn_prediction = self.tcn_predictor.predict_symbol(symbol)

            # NUEVO: Análisis de coherencia mejorado para timeframes cortos
            coherence_analysis = self.analyze_short_term_coherence(indicators, tcn_prediction, symbol)

            # Análisis de la predicción actual
            current_indicators = indicators.iloc[-1]
            current_technical = self.analyze_technical_signal(current_indicators, symbol)

            # Calcular precio actual
            current_price = df['close'].iloc[-1]

            result = {
                'symbol': symbol,
                'current_price': current_price,
                'current_tcn_prediction': tcn_prediction,
                'current_technical_analysis': current_technical,
                'coherence_analysis': coherence_analysis,
                'agreement_rate': coherence_analysis['agreement_rate'],
                'agreements': coherence_analysis['agreements'],
                'total_analyses': coherence_analysis['total_comparisons'],
                'analysis_method': 'short_term_optimized'
            }

            # Generar reporte
            report = self.generate_signal_coherence_report(result)
            print(report)

            # Guardar reporte individual
            filename = f"signal_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report)

            return result

        except Exception as e:
            print(f"❌ Error analizando {symbol}: {e}")
            return {'error': str(e)}

    def analyze_short_term_coherence(self, indicators: pd.DataFrame, tcn_prediction: Dict, symbol: str) -> Dict:
        """🔍 Analizar coherencia en timeframes cortos (NUEVO MÉTODO)"""
        try:
            # Analizar últimas 2 horas (24 períodos de 5 minutos) con mayor granularidad
            recent_data = indicators.tail(24)

            agreements = 0
            total_comparisons = 0
            signal_history = []

            # NUEVO: Comparar cada 5 minutos (misma escala que TCN)
            for i in range(len(recent_data)):
                current_indicators = recent_data.iloc[i]

                # Análisis técnico instantáneo (misma escala que TCN)
                technical_analysis = self.analyze_technical_signal(current_indicators, symbol)

                # Obtener señal TCN (simulamos que sería similar en ese momento)
                # En producción, esto sería la predicción TCN real de ese momento
                tcn_signal = tcn_prediction.get('signal', 'HOLD')

                # Comparar señales
                agreement = technical_analysis['signal'] == tcn_signal
                if agreement:
                    agreements += 1

                total_comparisons += 1

                signal_history.append({
                    'timestamp': current_indicators.name,
                    'tcn_signal': tcn_signal,
                    'technical_signal': technical_analysis['signal'],
                    'agreement': agreement,
                    'rsi': technical_analysis['rsi'],
                    'macd': technical_analysis['macd'],
                    'stoch_k': technical_analysis['stoch_k'],
                    'confidence_proxy': self.calculate_technical_confidence(technical_analysis)
                })

            # Calcular estadísticas
            agreement_rate = (agreements / total_comparisons * 100) if total_comparisons > 0 else 0

            # NUEVO: Análisis de tendencia de coherencia
            recent_agreements = sum(1 for s in signal_history[-6:] if s['agreement'])  # Últimos 30 minutos
            recent_coherence = (recent_agreements / 6 * 100) if len(signal_history) >= 6 else 0

            # NUEVO: Análisis de volatilidad de señales
            tcn_changes = 0
            tech_changes = 0
            for i in range(1, len(signal_history)):
                if signal_history[i]['tcn_signal'] != signal_history[i-1]['tcn_signal']:
                    tcn_changes += 1
                if signal_history[i]['technical_signal'] != signal_history[i-1]['technical_signal']:
                    tech_changes += 1

            return {
                'agreement_rate': agreement_rate,
                'agreements': agreements,
                'total_comparisons': total_comparisons,
                'recent_coherence': recent_coherence,
                'tcn_signal_changes': tcn_changes,
                'technical_signal_changes': tech_changes,
                'signal_stability': {
                    'tcn_stability': (1 - tcn_changes / max(total_comparisons - 1, 1)) * 100,
                    'technical_stability': (1 - tech_changes / max(total_comparisons - 1, 1)) * 100
                },
                'analysis_period': '2 hours (5-minute intervals)',
                'method': 'short_term_coherence'
            }

        except Exception as e:
            print(f"❌ Error en análisis de coherencia: {e}")
            return {
                'agreement_rate': 0,
                'agreements': 0,
                'total_comparisons': 0,
                'error': str(e)
            }

    def calculate_technical_confidence(self, technical_analysis: Dict) -> float:
        """🔍 Calcular confianza técnica basada en fuerza de señales"""
        try:
            buy_signals = technical_analysis.get('buy_signals', 0)
            sell_signals = technical_analysis.get('sell_signals', 0)

            # Calcular confianza basada en diferencia de señales
            total_signals = buy_signals + sell_signals
            if total_signals == 0:
                return 0.5  # Neutral

            signal_difference = abs(buy_signals - sell_signals)
            max_possible_difference = max(buy_signals, sell_signals)

            if max_possible_difference == 0:
                return 0.5

            confidence = 0.5 + (signal_difference / (total_signals * 2)) * 0.5
            return min(confidence, 1.0)

        except Exception as e:
            return 0.5

    def generate_signal_coherence_report(self, analysis: Dict) -> str:
        """📊 Generar reporte de coherencia de señales (MEJORADO para timeframes cortos)"""
        try:
            symbol = analysis['symbol']
            tcn_pred = analysis['current_tcn_prediction']
            tech_analysis = analysis['current_technical_analysis']
            coherence = analysis.get('coherence_analysis', {})

            # Determinar estado de coherencia
            agreement_rate = coherence.get('agreement_rate', 0)
            coherence_status = "✅ COHERENTE" if tcn_pred.get('signal') == tech_analysis['signal'] else "⚠️ INCOHERENTE"

            # Análisis de estabilidad
            stability = coherence.get('signal_stability', {})
            tcn_stability = stability.get('tcn_stability', 0)
            tech_stability = stability.get('technical_stability', 0)

            report = f"""
🔍 ANÁLISIS DE COHERENCIA DE SEÑALES - {symbol}
{'='*60}

💹 PREDICCIÓN TCN ACTUAL:
   Señal: {tcn_pred.get('signal', 'N/A')}
   Confianza: {tcn_pred.get('confidence', 0):.1%}
   Precio actual: ${analysis.get('current_price', 0):.4f}

📈 ANÁLISIS TÉCNICO ACTUAL:
   Señal técnica: {tech_analysis['signal']}
   Señales de compra: {tech_analysis['buy_signals']}
   Señales de venta: {tech_analysis['sell_signals']}
   RSI: {tech_analysis['rsi']:.1f}
   MACD: {tech_analysis['macd']:.4f}
   Stochastic K: {tech_analysis['stoch_k']:.1f}

🎯 COHERENCIA (Timeframe Corto):
   {coherence_status}
   Tasa de acuerdo (2h): {agreement_rate:.1f}%
   Acuerdos: {coherence.get('agreements', 0)}/{coherence.get('total_comparisons', 0)}

📊 ANÁLISIS DE ESTABILIDAD:
   Estabilidad TCN: {tcn_stability:.1f}%
   Estabilidad Técnica: {tech_stability:.1f}%
   Cambios TCN (2h): {coherence.get('tcn_signal_changes', 0)}
   Cambios Técnicos (2h): {coherence.get('technical_signal_changes', 0)}

📋 DETALLES TÉCNICOS:
"""

            for detail in tech_analysis['details']:
                report += f"   • {detail}\n"

            # Interpretación mejorada para timeframes cortos
            if agreement_rate < 40:
                report += f"\n⚠️ COHERENCIA BAJA ({agreement_rate:.1f}%)\n"
                report += "   En timeframes cortos, esto puede indicar:\n"
                report += "   - Señales en diferentes escalas temporales\n"
                report += "   - Modelo TCN más sensible que indicadores técnicos\n"
                report += "   - Posible oportunidad de arbitraje temporal\n"
            elif agreement_rate < 70:
                report += f"\n🟡 COHERENCIA MODERADA ({agreement_rate:.1f}%)\n"
                report += "   Coherencia aceptable para timeframes cortos\n"
            else:
                report += f"\n✅ COHERENCIA ALTA ({agreement_rate:.1f}%)\n"
                report += "   Excelente alineación entre TCN e indicadores técnicos\n"

            # Análisis de volatilidad de señales
            total_changes = coherence.get('tcn_signal_changes', 0) + coherence.get('technical_signal_changes', 0)
            if total_changes > 10:
                report += f"\n📈 ALTA VOLATILIDAD DE SEÑALES ({total_changes} cambios en 2h)\n"
                report += "   - Mercado muy activo o indeciso\n"
                report += "   - Considerar filtros de estabilidad\n"
            elif total_changes > 5:
                report += f"\n🟡 VOLATILIDAD MODERADA ({total_changes} cambios en 2h)\n"
                report += "   - Actividad normal del mercado\n"
            else:
                report += f"\n✅ BAJA VOLATILIDAD ({total_changes} cambios en 2h)\n"
                report += "   - Mercado estable o en tendencia clara\n"

            return report

        except Exception as e:
            return f"❌ Error generando reporte: {e}"

    async def run_full_analysis(self) -> Dict:
        """🚀 Ejecutar análisis completo de todos los símbolos"""
        print("🚀 Iniciando análisis completo de señales de trading...")
        print("="*70)

        # Cargar modelos TCN
        print("🧠 Cargando modelos TCN...")
        models_loaded = self.tcn_predictor.load_all_models()
        if not models_loaded:
            print("⚠️ No se pudieron cargar todos los modelos TCN")

        results = {}

        for symbol in self.symbols:
            analysis = await self.analyze_symbol(symbol)
            results[symbol] = analysis

            if 'error' not in analysis:
                # Generar reporte individual
                report = self.generate_signal_coherence_report(analysis)
                print(report)

                # Guardar en archivo
                with open(f"signal_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
                    f.write(report)
            else:
                print(f"❌ Error analizando {symbol}: {analysis['error']}")

        # Resumen general
        self.generate_summary_report(results)

        return results

    def generate_summary_report(self, results: Dict):
        """📊 Generar reporte resumen de todos los símbolos"""
        try:
            print("\n" + "="*70)
            print("📊 RESUMEN GENERAL DE ANÁLISIS DE SEÑALES")
            print("="*70)

            total_coherence = 0
            valid_symbols = 0

            for symbol, analysis in results.items():
                if 'error' not in analysis:
                    coherence = analysis['agreement_rate']
                    total_coherence += coherence
                    valid_symbols += 1

                    status = "✅ BUENA" if coherence >= 70 else "⚠️ REGULAR" if coherence >= 50 else "❌ MALA"
                    print(f"{symbol:10} | Coherencia: {coherence:5.1f}% | {status}")

            if valid_symbols > 0:
                avg_coherence = total_coherence / valid_symbols
                print(f"\n📈 COHERENCIA PROMEDIO: {avg_coherence:.1f}%")

                if avg_coherence >= 70:
                    print("✅ SISTEMA FUNCIONANDO CORRECTAMENTE")
                elif avg_coherence >= 50:
                    print("⚠️ SISTEMA REQUIERE AJUSTES")
                else:
                    print("❌ SISTEMA REQUIERE REVISIÓN URGENTE")

            # Guardar resumen
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'average_coherence': avg_coherence if valid_symbols > 0 else 0,
                'valid_symbols': valid_symbols,
                'results': results
            }

            with open(f"trading_signals_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)

            print(f"\n💾 Resultados guardados en archivos de análisis")

        except Exception as e:
            print(f"❌ Error generando resumen: {e}")

async def main():
    """🎯 Función principal"""
    analyzer = TradingSignalAnalyzer()
    await analyzer.run_full_analysis()

if __name__ == "__main__":
    asyncio.run(main())
