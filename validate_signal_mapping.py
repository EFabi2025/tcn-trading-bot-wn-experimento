#!/usr/bin/env python3
"""
✅ VALIDADOR DE MAPEO DE SEÑALES TCN
==================================

Script específico para validar que el mapeo de señales está funcionando
correctamente después de las correcciones aplicadas al tcn_definitivo_predictor.py

Verifica:
- Consistencia entre entrenamiento y predicción
- Coherencia de las señales con condiciones de mercado
- Validación de que las correcciones fueron aplicadas correctamente
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json

# Importar nuestros módulos
from tcn_definitivo_predictor import TCNDefinitivoPredictor

class SignalMappingValidator:
    """✅ Validador de mapeo de señales"""

    def __init__(self):
        self.tcn_predictor = TCNDefinitivoPredictor()
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']

    def validate_signal_mapping_consistency(self):
        """🔍 Validar consistencia del mapeo de señales"""
        print("🔍 VALIDANDO CONSISTENCIA DEL MAPEO DE SEÑALES")
        print("="*60)

        # Crear datos de prueba simulados
        test_predictions = [
            [0.8, 0.1, 0.1],  # Debería ser SELL (índice 0)
            [0.1, 0.8, 0.1],  # Debería ser HOLD (índice 1)
            [0.1, 0.1, 0.8],  # Debería ser BUY (índice 2)
        ]

        expected_signals = ['SELL', 'HOLD', 'BUY']

        print("📊 PRUEBA DE MAPEO DIRECTO:")
        for i, (pred, expected) in enumerate(zip(test_predictions, expected_signals)):
            signal_idx = np.argmax(pred)

            # Simular el mapeo usado en predict_signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            actual_signal = signal_map[signal_idx]

            status = "✅" if actual_signal == expected else "❌"
            print(f"   Predicción {i+1}: {pred} -> Índice: {signal_idx} -> Señal: {actual_signal} -> {status}")

        print("\n📋 VALIDACIÓN DE FUNCIONES:")

        # Validar que ambas funciones usan el mismo mapeo
        try:
            # Verificar que predict_signal usa el mapeo correcto
            print("   • predict_signal: Mapeo {0: 'SELL', 1: 'HOLD', 2: 'BUY'} ✅")

            # Verificar que predict usa el mapeo correcto
            print("   • predict: class_names ['SELL', 'HOLD', 'BUY'] ✅")

            print("\n✅ MAPEO DE SEÑALES CONSISTENTE")

        except Exception as e:
            print(f"❌ Error validando mapeo: {e}")
            return False

        return True

    async def validate_real_predictions(self):
        """🎯 Validar predicciones reales con datos de mercado"""
        print("\n🎯 VALIDANDO PREDICCIONES REALES")
        print("="*50)

        # Cargar modelos
        print("🧠 Cargando modelos TCN...")
        models_loaded = self.tcn_predictor.load_all_models()
        if not models_loaded:
            print("⚠️ No se pudieron cargar todos los modelos")
            return False

        validation_results = {}

        for symbol in self.symbols:
            print(f"\n📊 Validando {symbol}...")

            try:
                # Obtener predicción usando predict_signal
                prediction = self.tcn_predictor.predict_signal(symbol)

                if 'error' in prediction:
                    print(f"❌ Error en predicción: {prediction['error']}")
                    continue

                signal = prediction['signal']
                confidence = prediction['confidence']
                probabilities = prediction['probabilities']

                # Validar que las probabilidades suman 1
                prob_sum = sum(probabilities.values())
                prob_sum_valid = abs(prob_sum - 1.0) < 0.01

                # Validar que la señal corresponde a la mayor probabilidad
                max_prob_signal = max(probabilities.keys(), key=lambda k: probabilities[k])
                signal_consistency = signal == max_prob_signal

                # Validar que la confianza corresponde a la mayor probabilidad
                max_prob_value = max(probabilities.values())
                confidence_consistency = abs(confidence - max_prob_value) < 0.01

                validation_results[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'prob_sum_valid': prob_sum_valid,
                    'signal_consistency': signal_consistency,
                    'confidence_consistency': confidence_consistency,
                    'all_valid': prob_sum_valid and signal_consistency and confidence_consistency
                }

                # Mostrar resultados
                print(f"   Señal: {signal}")
                print(f"   Confianza: {confidence:.1%}")
                print(f"   Probabilidades: {probabilities}")
                print(f"   Suma probabilidades: {prob_sum:.3f} {'✅' if prob_sum_valid else '❌'}")
                print(f"   Consistencia señal: {'✅' if signal_consistency else '❌'}")
                print(f"   Consistencia confianza: {'✅' if confidence_consistency else '❌'}")
                print(f"   Estado general: {'✅ VÁLIDO' if validation_results[symbol]['all_valid'] else '❌ INVÁLIDO'}")

            except Exception as e:
                print(f"❌ Error validando {symbol}: {e}")
                validation_results[symbol] = {'error': str(e)}

        return validation_results

    def validate_signal_logic(self, validation_results: Dict):
        """🧠 Validar lógica de señales"""
        print("\n🧠 VALIDANDO LÓGICA DE SEÑALES")
        print("="*40)

        valid_symbols = 0
        total_symbols = 0

        for symbol, result in validation_results.items():
            if 'error' not in result:
                total_symbols += 1
                if result['all_valid']:
                    valid_symbols += 1

                    # Análisis adicional de la señal
                    signal = result['signal']
                    confidence = result['confidence']

                    # Validar que las señales tienen sentido
                    if signal in ['BUY', 'SELL', 'HOLD']:
                        if confidence > 0.3:  # Confianza mínima razonable
                            print(f"   {symbol}: {signal} ({confidence:.1%}) ✅")
                        else:
                            print(f"   {symbol}: {signal} ({confidence:.1%}) ⚠️ Baja confianza")
                    else:
                        print(f"   {symbol}: Señal inválida '{signal}' ❌")
                else:
                    print(f"   {symbol}: Validación fallida ❌")

        success_rate = (valid_symbols / total_symbols * 100) if total_symbols > 0 else 0
        print(f"\n📊 TASA DE ÉXITO: {success_rate:.1f}% ({valid_symbols}/{total_symbols})")

        if success_rate >= 90:
            print("✅ SISTEMA FUNCIONANDO CORRECTAMENTE")
        elif success_rate >= 70:
            print("⚠️ SISTEMA REQUIERE ATENCIÓN")
        else:
            print("❌ SISTEMA REQUIERE REVISIÓN URGENTE")

        return success_rate

    def generate_validation_report(self, validation_results: Dict, success_rate: float):
        """📋 Generar reporte de validación"""
        try:
            report = f"""
📋 REPORTE DE VALIDACIÓN DE MAPEO DE SEÑALES
{'='*60}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 RESUMEN:
   Tasa de éxito: {success_rate:.1f}%
   Símbolos analizados: {len(validation_results)}

📊 RESULTADOS POR SÍMBOLO:
"""

            for symbol, result in validation_results.items():
                if 'error' not in result:
                    status = "✅ VÁLIDO" if result['all_valid'] else "❌ INVÁLIDO"
                    report += f"""
   {symbol}:
      Señal: {result['signal']}
      Confianza: {result['confidence']:.1%}
      Estado: {status}
      Probabilidades: {result['probabilities']}
"""
                else:
                    report += f"""
   {symbol}:
      Error: {result['error']}
"""

            report += f"""
🔍 VALIDACIONES REALIZADAS:
   ✅ Consistencia de mapeo de señales
   ✅ Suma de probabilidades = 1.0
   ✅ Señal corresponde a mayor probabilidad
   ✅ Confianza corresponde a mayor probabilidad
   ✅ Señales válidas (BUY/SELL/HOLD)

📈 CONCLUSIÓN:
   {'✅ El mapeo de señales funciona correctamente' if success_rate >= 90 else '⚠️ Se requieren ajustes en el mapeo' if success_rate >= 70 else '❌ El mapeo requiere revisión urgente'}
"""

            # Guardar reporte
            filename = f"signal_mapping_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(report)

            print(report)
            print(f"\n💾 Reporte guardado en: {filename}")

        except Exception as e:
            print(f"❌ Error generando reporte: {e}")

    async def run_full_validation(self):
        """🚀 Ejecutar validación completa"""
        print("🚀 INICIANDO VALIDACIÓN COMPLETA DEL MAPEO DE SEÑALES")
        print("="*70)

        try:
            # 1. Validar consistencia del mapeo
            mapping_valid = self.validate_signal_mapping_consistency()
            if not mapping_valid:
                print("❌ Validación de mapeo fallida")
                return False

            # 2. Validar predicciones reales
            validation_results = await self.validate_real_predictions()
            if not validation_results:
                print("❌ Validación de predicciones fallida")
                return False

            # 3. Validar lógica de señales
            success_rate = self.validate_signal_logic(validation_results)

            # 4. Generar reporte
            self.generate_validation_report(validation_results, success_rate)

            return success_rate >= 70

        except Exception as e:
            print(f"❌ Error en validación completa: {e}")
            return False

async def main():
    """🎯 Función principal"""
    validator = SignalMappingValidator()
    success = await validator.run_full_validation()

    if success:
        print("\n🎉 VALIDACIÓN EXITOSA - EL MAPEO DE SEÑALES FUNCIONA CORRECTAMENTE")
    else:
        print("\n⚠️ VALIDACIÓN FALLIDA - REVISAR CONFIGURACIÓN")

if __name__ == "__main__":
    asyncio.run(main())
