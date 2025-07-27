#!/usr/bin/env python3
"""
🧪 TEST COMPARACIÓN FEATURES: TA-LIB vs PANDAS-TA
=================================================

Script para determinar si hay diferencias entre las features
calculadas con TA-Lib vs pandas-ta.

Si las diferencias son mínimas: NO hay que reentrenar
Si las diferencias son significativas: SÍ hay que reentrenar
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from centralized_features_engine2 import CentralizedFeaturesEngine
from centralized_features_engine_pandas_ta import CentralizedFeaturesEnginePandasTA


class FeaturesComparator:
    """🔬 Comparador de features entre TA-Lib y pandas-ta"""

    def __init__(self):
        self.talib_engine = CentralizedFeaturesEngine()
        self.pandas_ta_engine = CentralizedFeaturesEnginePandasTA()

    def create_test_data(self, days: int = 30) -> pd.DataFrame:
        """📊 Crear datos de prueba realistas"""

        # Generar datos OHLCV realistas
        periods = days * 24 * 60  # Minutos
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')

        np.random.seed(42)  # Para reproducibilidad

        # Simular precio con random walk + trend
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, periods)  # Pequeño drift positivo
        prices = base_price * np.exp(np.cumsum(returns))

        # Generar OHLCV
        noise_factor = 0.001

        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, noise_factor, periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, noise_factor * 2, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, noise_factor * 2, periods))),
            'close': prices,
            'volume': np.random.lognormal(10, 0.5, periods)
        }, index=dates)

        # Asegurar lógica OHLC correcta
        df['high'] = np.maximum.reduce([df['open'], df['high'], df['close']])
        df['low'] = np.minimum.reduce([df['open'], df['low'], df['close']])

        print(f"✅ Datos de prueba generados: {len(df)} registros de {days} días")
        return df

    def compare_features(self, df: pd.DataFrame, feature_set: str = 'tcn_definitivo') -> dict:
        """🔬 Comparar features entre ambos motores"""

        print(f"\n🔬 COMPARANDO FEATURES: {feature_set}")
        print("=" * 60)

        try:
            # Calcular features con ambos motores
            print("🔄 Calculando features con TA-Lib...")
            features_talib = self.talib_engine.calculate_features(df.copy(), feature_set)

            print("🔄 Calculando features con pandas-ta...")
            features_pandas_ta = self.pandas_ta_engine.calculate_features(df.copy(), feature_set)

            # Analizar diferencias
            results = self._analyze_differences(features_talib, features_pandas_ta, feature_set)

            return results

        except Exception as e:
            print(f"❌ Error comparando features: {e}")
            return {'error': str(e)}

    def _analyze_differences(self, df_talib: pd.DataFrame, df_pandas_ta: pd.DataFrame, feature_set: str) -> dict:
        """📊 Analizar diferencias entre features"""

        # Features comunes
        common_features = list(set(df_talib.columns) & set(df_pandas_ta.columns))
        missing_talib = set(df_pandas_ta.columns) - set(df_talib.columns)
        missing_pandas_ta = set(df_talib.columns) - set(df_pandas_ta.columns)

        print(f"📊 Features comunes: {len(common_features)}")
        print(f"⚠️ Faltantes en TA-Lib: {len(missing_talib)}")
        print(f"⚠️ Faltantes en pandas-ta: {len(missing_pandas_ta)}")

        if missing_talib:
            print(f"   📝 Faltantes TA-Lib: {list(missing_talib)[:5]}...")
        if missing_pandas_ta:
            print(f"   📝 Faltantes pandas-ta: {list(missing_pandas_ta)[:5]}...")

        # Análisis estadístico
        comparison_stats = {}
        critical_differences = []
        minor_differences = []
        identical_features = []

        print(f"\n🔍 ANÁLISIS ESTADÍSTICO:")
        print("-" * 40)

        for feature in common_features:
            try:
                # Obtener valores (eliminar NaN para comparación)
                vals_talib = df_talib[feature].dropna()
                vals_pandas_ta = df_pandas_ta[feature].dropna()

                # Alinear longitudes
                min_len = min(len(vals_talib), len(vals_pandas_ta))
                if min_len < 10:  # Muy pocos datos
                    continue

                vals_talib = vals_talib.iloc[-min_len:]
                vals_pandas_ta = vals_pandas_ta.iloc[-min_len:]

                # Calcular métricas
                correlation = np.corrcoef(vals_talib, vals_pandas_ta)[0, 1]
                mse = np.mean((vals_talib - vals_pandas_ta) ** 2)
                mae = np.mean(np.abs(vals_talib - vals_pandas_ta))
                mape = np.mean(np.abs((vals_talib - vals_pandas_ta) / (vals_talib + 1e-8))) * 100

                # Diferencia relativa
                max_val = max(np.abs(vals_talib).max(), np.abs(vals_pandas_ta).max())
                relative_mae = mae / (max_val + 1e-8) * 100

                stats = {
                    'correlation': correlation,
                    'mse': mse,
                    'mae': mae,
                    'mape': mape,
                    'relative_mae': relative_mae,
                    'samples': min_len
                }

                comparison_stats[feature] = stats

                # Clasificar diferencias
                if relative_mae > 5.0 or correlation < 0.95:  # Diferencias críticas
                    critical_differences.append((feature, relative_mae, correlation))
                elif relative_mae > 1.0 or correlation < 0.99:  # Diferencias menores
                    minor_differences.append((feature, relative_mae, correlation))
                else:  # Prácticamente idénticos
                    identical_features.append(feature)

            except Exception as e:
                print(f"   ⚠️ Error analizando {feature}: {e}")

        # Resumen de resultados
        print(f"✅ Idénticas: {len(identical_features)}")
        print(f"⚠️ Diferencias menores: {len(minor_differences)}")
        print(f"🚨 Diferencias críticas: {len(critical_differences)}")

        # Mostrar diferencias críticas
        if critical_differences:
            print(f"\n🚨 DIFERENCIAS CRÍTICAS (requieren reentrenamiento):")
            for feature, mae, corr in critical_differences[:10]:
                print(f"   📊 {feature}: MAE={mae:.2f}%, Corr={corr:.3f}")

        # Mostrar diferencias menores
        if minor_differences:
            print(f"\n⚠️ DIFERENCIAS MENORES:")
            for feature, mae, corr in minor_differences[:5]:
                print(f"   📊 {feature}: MAE={mae:.2f}%, Corr={corr:.3f}")

        # Conclusión
        print(f"\n🎯 CONCLUSIÓN:")
        total_features = len(common_features)
        critical_pct = len(critical_differences) / total_features * 100

        if critical_pct > 10:  # Más del 10% de features críticas
            recommendation = "🚨 REENTRENAMIENTO NECESARIO"
            reason = f"{critical_pct:.1f}% features con diferencias críticas"
        elif critical_pct > 5:  # 5-10% críticas
            recommendation = "⚠️ REENTRENAMIENTO RECOMENDADO"
            reason = f"{critical_pct:.1f}% features con diferencias críticas"
        elif len(minor_differences) > total_features * 0.3:  # Muchas diferencias menores
            recommendation = "⚠️ REENTRENAMIENTO RECOMENDADO"
            reason = "Muchas diferencias menores acumuladas"
        else:
            recommendation = "✅ NO REQUIERE REENTRENAMIENTO"
            reason = "Diferencias mínimas o inexistentes"

        print(f"   {recommendation}")
        print(f"   📝 Razón: {reason}")

        return {
            'feature_set': feature_set,
            'total_features': total_features,
            'common_features': len(common_features),
            'identical': len(identical_features),
            'minor_differences': len(minor_differences),
            'critical_differences': len(critical_differences),
            'critical_percentage': critical_pct,
            'recommendation': recommendation,
            'reason': reason,
            'critical_features': [f[0] for f in critical_differences],
            'stats': comparison_stats
        }

    def run_comprehensive_test(self) -> dict:
        """🧪 Ejecutar prueba completa"""

        print("🧪 PRUEBA COMPLETA: COMPARACIÓN TA-LIB vs PANDAS-TA")
        print("=" * 70)

        # Crear datos de prueba
        test_data = self.create_test_data(days=7)  # 7 días de datos

        # Probar todos los feature sets
        results = {}

        for feature_set in ['tcn_definitivo', 'tcn_final']:
            try:
                result = self.compare_features(test_data, feature_set)
                results[feature_set] = result
            except Exception as e:
                print(f"❌ Error con {feature_set}: {e}")
                results[feature_set] = {'error': str(e)}

        # Resumen final
        print(f"\n" + "=" * 70)
        print("🎯 RESUMEN FINAL:")
        print("=" * 70)

        need_retrain = False

        for feature_set, result in results.items():
            if 'error' in result:
                print(f"❌ {feature_set}: ERROR - {result['error']}")
                continue

            print(f"\n📊 {feature_set.upper()}:")
            print(f"   ✅ Features idénticas: {result['identical']}")
            print(f"   ⚠️ Diferencias menores: {result['minor_differences']}")
            print(f"   🚨 Diferencias críticas: {result['critical_differences']}")
            print(f"   📈 % críticas: {result['critical_percentage']:.1f}%")
            print(f"   🎯 {result['recommendation']}")

            if "REENTRENAMIENTO" in result['recommendation']:
                need_retrain = True

        # Recomendación final
        print(f"\n🏁 RECOMENDACIÓN FINAL:")
        if need_retrain:
            print("🚨 SE REQUIERE REENTRENAMIENTO de los modelos")
            print("   📝 Razón: Diferencias significativas detectadas")
            print("   🔧 Acción: Usar train_master_hybrid_trainer.py con pandas-ta")
        else:
            print("✅ NO SE REQUIERE REENTRENAMIENTO")
            print("   📝 Razón: Features prácticamente idénticas")
            print("   🔧 Acción: Puedes cambiar a pandas-ta directamente")

        return results


def main():
    """🎯 Ejecutar comparación completa"""
    comparator = FeaturesComparator()
    results = comparator.run_comprehensive_test()

    # Guardar resultados para análisis posterior
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        import json
        with open(f'features_comparison_{timestamp}.json', 'w') as f:
            # Convertir numpy types para JSON
            json_results = {}
            for k, v in results.items():
                if isinstance(v, dict) and 'stats' in v:
                    v_copy = v.copy()
                    # Simplificar stats para JSON
                    v_copy['stats'] = {feat: {
                        'correlation': float(stats['correlation']) if not np.isnan(stats['correlation']) else None,
                        'relative_mae': float(stats['relative_mae']),
                        'samples': int(stats['samples'])
                    } for feat, stats in v['stats'].items()}
                    json_results[k] = v_copy
                else:
                    json_results[k] = v

            json.dump(json_results, f, indent=2)
        print(f"\n💾 Resultados guardados en: features_comparison_{timestamp}.json")
    except Exception as e:
        print(f"⚠️ No se pudieron guardar resultados: {e}")


if __name__ == "__main__":
    main()
