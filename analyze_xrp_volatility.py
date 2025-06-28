#!/usr/bin/env python3
"""
📊 ANÁLISIS DE VOLATILIDAD XRPUSDT
=================================

Análisis estadístico de volatilidad real para calcular thresholds óptimos
siguiendo la metodología de los modelos TCN definitivos exitosos.

Objetivo: Calcular thresholds que produzcan distribución balanceada:
- 30% SELL, 40% HOLD, 30% BUY
"""

import numpy as np
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class XRPVolatilityAnalyzer:
    """Analizador de volatilidad para XRPUSDT"""
    
    def __init__(self):
        """Inicializar cliente Binance"""
        try:
            # Intentar con credenciales
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            
            if api_key and secret_key:
                self.client = Client(api_key, secret_key)
                print("📡 Cliente Binance autenticado")
            else:
                self.client = Client()  # Cliente público
                print("📡 Cliente Binance público")
                
        except Exception as e:
            print(f"❌ Error inicializando cliente: {e}")
            raise
    
    def analyze_symbol_volatility(self, symbol: str = "XRPUSDT", days: int = 30):
        """
        Analizar volatilidad real de XRPUSDT para calcular thresholds óptimos
        
        Args:
            symbol: Par de trading
            days: Días de datos históricos para análisis
            
        Returns:
            dict: Thresholds y estadísticas calculadas
        """
        print(f"\n🔍 ANÁLISIS DE VOLATILIDAD {symbol}")
        print("=" * 60)
        print(f"📅 Período: {days} días de datos reales")
        
        # 1. Obtener datos reales de Binance
        print(f"\n📊 Obteniendo datos reales de {symbol}...")
        klines_data = self._get_historical_data(symbol, days)
        
        if not klines_data:
            raise ValueError("No se pudieron obtener datos históricos")
        
        # 2. Convertir a DataFrame
        df = self._klines_to_dataframe(klines_data)
        print(f"✅ Datos procesados: {len(df):,} velas de 5 minutos")
        print(f"   📅 Período: {df.index[0]} - {df.index[-1]}")
        
        # 3. Calcular returns de 5 minutos
        returns = df['close'].pct_change().dropna()
        print(f"   📈 Returns calculados: {len(returns):,} muestras")
        
        # 4. Análisis estadístico detallado
        stats = self._calculate_detailed_stats(returns)
        
        # 5. Calcular thresholds balanceados
        thresholds = self._calculate_balanced_thresholds(returns, stats)
        
        # 6. Validar thresholds
        distribution = self._validate_thresholds(returns, thresholds)
        
        # 7. Compilar resultados
        results = {
            'symbol': symbol,
            'analysis_period_days': days,
            'total_samples': len(returns),
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'current': float(df['close'].iloc[-1])
            },
            'volatility_stats': stats,
            'thresholds': thresholds,
            'predicted_distribution': distribution,
            'data_quality': self._assess_data_quality(df, returns)
        }
        
        # 8. Mostrar resultados
        self._print_results(results)
        
        return results
    
    def _get_historical_data(self, symbol: str, days: int) -> list:
        """Obtener datos históricos de Binance"""
        try:
            # Calcular klines necesarios (5 minutos)
            klines_per_day = 12 * 24  # 288 klines de 5m por día
            total_klines = days * klines_per_day
            
            # Obtener en chunks si es necesario
            all_klines = []
            remaining = total_klines
            end_time = datetime.now()
            
            while remaining > 0 and len(all_klines) < total_klines:
                chunk_size = min(1000, remaining)  # Límite Binance
                
                chunk_klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_5MINUTE,
                    limit=chunk_size,
                    endTime=int(end_time.timestamp() * 1000)
                )
                
                if not chunk_klines:
                    break
                
                all_klines = chunk_klines + all_klines
                remaining -= len(chunk_klines)
                
                # Actualizar end_time para siguiente chunk
                earliest_time = datetime.fromtimestamp(chunk_klines[0][0] / 1000)
                end_time = earliest_time - timedelta(minutes=5)
            
            print(f"   📦 Obtenidos {len(all_klines):,} klines históricos")
            return all_klines
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return []
    
    def _klines_to_dataframe(self, klines_data: list) -> pd.DataFrame:
        """Convertir klines a DataFrame"""
        try:
            df_data = []
            for kline in klines_data:
                df_data.append({
                    'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception as e:
            print(f"❌ Error procesando datos: {e}")
            return pd.DataFrame()
    
    def _calculate_detailed_stats(self, returns: pd.Series) -> dict:
        """Calcular estadísticas detalladas de volatilidad"""
        print(f"\n📈 ESTADÍSTICAS DE VOLATILIDAD")
        print("-" * 40)
        
        stats = {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'volatility_daily': float(returns.std() * np.sqrt(288)),  # 288 períodos de 5m por día
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'percentiles': {
                'p1': float(returns.quantile(0.01)),
                'p5': float(returns.quantile(0.05)),
                'p10': float(returns.quantile(0.10)),
                'p15': float(returns.quantile(0.15)),
                'p25': float(returns.quantile(0.25)),
                'p50': float(returns.quantile(0.50)),
                'p75': float(returns.quantile(0.75)),
                'p85': float(returns.quantile(0.85)),
                'p90': float(returns.quantile(0.90)),
                'p95': float(returns.quantile(0.95)),
                'p99': float(returns.quantile(0.99))
            }
        }
        
        print(f"   📊 Media: {stats['mean']:.6f}")
        print(f"   📊 Std Dev: {stats['std']:.6f}")
        print(f"   📊 Volatilidad Diaria: {stats['volatility_daily']:.4f} ({stats['volatility_daily']*100:.2f}%)")
        print(f"   📊 Skewness: {stats['skewness']:.4f}")
        print(f"   📊 Kurtosis: {stats['kurtosis']:.4f}")
        
        return stats
    
    def _calculate_balanced_thresholds(self, returns: pd.Series, stats: dict) -> dict:
        """
        Calcular thresholds para distribución balanceada 30% SELL, 40% HOLD, 30% BUY
        """
        print(f"\n🎯 CÁLCULO DE THRESHOLDS BALANCEADOS")
        print("-" * 40)
        
        # Estrategia 1: Percentiles simétricos (15% y 85%)
        sell_threshold_p15 = stats['percentiles']['p15']
        buy_threshold_p85 = stats['percentiles']['p85']
        
        # Estrategia 2: Basado en desviaciones estándar
        std_multiplier = 0.8  # Ajustable
        sell_threshold_std = stats['mean'] - std_multiplier * stats['std']
        buy_threshold_std = stats['mean'] + std_multiplier * stats['std']
        
        # Estrategia 3: Híbrida (promedio de ambas)
        sell_threshold_hybrid = (sell_threshold_p15 + sell_threshold_std) / 2
        buy_threshold_hybrid = (buy_threshold_p85 + buy_threshold_std) / 2
        
        thresholds = {
            'percentile_based': {
                'sell': sell_threshold_p15,
                'buy': buy_threshold_p85
            },
            'std_based': {
                'sell': sell_threshold_std,
                'buy': buy_threshold_std
            },
            'hybrid': {
                'sell': sell_threshold_hybrid,
                'buy': buy_threshold_hybrid
            }
        }
        
        print(f"   📊 Percentiles (15%/85%): SELL {sell_threshold_p15:.6f}, BUY {buy_threshold_p85:.6f}")
        print(f"   📊 Std-based (±0.8σ): SELL {sell_threshold_std:.6f}, BUY {buy_threshold_std:.6f}")
        print(f"   📊 Híbrido (recomendado): SELL {sell_threshold_hybrid:.6f}, BUY {buy_threshold_hybrid:.6f}")
        
        return thresholds
    
    def _validate_thresholds(self, returns: pd.Series, thresholds: dict) -> dict:
        """Validar que los thresholds produzcan distribución balanceada"""
        print(f"\n✅ VALIDACIÓN DE THRESHOLDS")
        print("-" * 40)
        
        distributions = {}
        
        for strategy, thresh in thresholds.items():
            labels = []
            for ret in returns:
                if ret < thresh['sell']:
                    labels.append('SELL')
                elif ret > thresh['buy']:
                    labels.append('BUY')
                else:
                    labels.append('HOLD')
            
            distribution = pd.Series(labels).value_counts(normalize=True).sort_index()
            distributions[strategy] = {
                'SELL': float(distribution.get('SELL', 0)),
                'HOLD': float(distribution.get('HOLD', 0)),
                'BUY': float(distribution.get('BUY', 0))
            }
            
            print(f"   🎯 {strategy.upper()}:")
            print(f"      SELL: {distributions[strategy]['SELL']:.1%}")
            print(f"      HOLD: {distributions[strategy]['HOLD']:.1%}")
            print(f"      BUY: {distributions[strategy]['BUY']:.1%}")
        
        return distributions
    
    def _assess_data_quality(self, df: pd.DataFrame, returns: pd.Series) -> dict:
        """Evaluar calidad de los datos"""
        quality = {
            'completeness': len(df) / (30 * 288),  # % de datos esperados
            'price_continuity': (df['close'].diff().abs() / df['close']).mean(),
            'volume_consistency': df['volume'].std() / df['volume'].mean(),
            'outliers_count': len(returns[np.abs(returns) > 3 * returns.std()]),
            'missing_data': df.isnull().sum().sum()
        }
        
        return quality
    
    def _print_results(self, results: dict):
        """Mostrar resultados finales"""
        print(f"\n🎉 RESULTADOS FINALES PARA {results['symbol']}")
        print("=" * 60)
        
        # Recomendar mejor estrategia
        hybrid_dist = results['predicted_distribution']['hybrid']
        target_dist = {'SELL': 0.30, 'HOLD': 0.40, 'BUY': 0.30}
        
        # Calcular error de la distribución híbrida vs objetivo
        error = sum(abs(hybrid_dist[k] - target_dist[k]) for k in target_dist.keys())
        
        print(f"📊 THRESHOLDS RECOMENDADOS (Híbridos):")
        print(f"   🔻 SELL Threshold: {results['thresholds']['hybrid']['sell']:.6f}")
        print(f"   🔺 BUY Threshold: {results['thresholds']['hybrid']['buy']:.6f}")
        print(f"   📈 Error vs distribución objetivo: {error:.3f}")
        
        print(f"\n📈 DISTRIBUCIÓN PREDICHA:")
        print(f"   🔻 SELL: {hybrid_dist['SELL']:.1%} (objetivo: 30%)")
        print(f"   ⏸️  HOLD: {hybrid_dist['HOLD']:.1%} (objetivo: 40%)")
        print(f"   🔺 BUY: {hybrid_dist['BUY']:.1%} (objetivo: 30%)")
        
        print(f"\n📊 ESTADÍSTICAS CLAVE:")
        print(f"   💰 Rango de precios: ${results['price_range']['min']:.4f} - ${results['price_range']['max']:.4f}")
        print(f"   📈 Volatilidad diaria: {results['volatility_stats']['volatility_daily']*100:.2f}%")
        print(f"   📊 Muestras analizadas: {results['total_samples']:,}")
        
        # Guardar resultados
        import json
        output_file = f"results/xrp_volatility_analysis.json"
        os.makedirs('results', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Resultados guardados en: {output_file}")


def main():
    """Ejecutar análisis de volatilidad de XRP"""
    print("🚀 ANÁLISIS DE VOLATILIDAD XRPUSDT")
    print("Metodología: TCN Definitivo para distribución balanceada")
    print("=" * 70)
    
    try:
        analyzer = XRPVolatilityAnalyzer()
        results = analyzer.analyze_symbol_volatility("XRPUSDT", days=30)
        
        print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"🎯 Use los thresholds híbridos para entrenar el modelo XRP definitivo")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return None


if __name__ == "__main__":
    main() 