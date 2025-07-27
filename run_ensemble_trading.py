#!/usr/bin/env python3
"""
🎯 DEMO COMPLETO DEL SISTEMA DE ENSAMBLE TCN
Script que muestra cómo entrenar y usar el sistema de ensamble de modelos 1m y 5m
"""

import asyncio
import sys
import os
from datetime import datetime

# Importar nuestros sistemas
from tcn_ensemble_trainer import TCNEnsembleTrainer
from tcn_ensemble_predictor import TCNEnsemblePredictor


class EnsembleTradingDemo:
    """🎯 Demo completo del sistema de ensamble"""

    def __init__(self):
        self.trainer = TCNEnsembleTrainer()
        self.predictor = TCNEnsemblePredictor()

    async def step1_train_ensemble_models(self):
        """🚀 Paso 1: Entrenar modelos de ensamble"""

        print("\n" + "=" * 80)
        print("🚀 PASO 1: ENTRENAMIENTO DE MODELOS DE ENSAMBLE")
        print("=" * 80)
        print("🎯 Se entrenarán modelos TCN para timeframes 1m y 5m")
        print("⏰ Tiempo estimado: ~2-3 horas por símbolo")
        print("💾 Los modelos se guardarán en models/ensemble_*_symbol/")

        # Verificar si ya existen modelos
        existing_models = self.check_existing_models()
        if existing_models:
            print(f"\n📦 Modelos existentes encontrados:")
            for symbol, timeframes in existing_models.items():
                print(f"   - {symbol}: {timeframes}")

            retrain = input("\n🤔 ¿Reentrenar modelos existentes? (y/n): ").lower().strip()
            if retrain != 'y':
                print("⏭️ Saltando al entrenamiento...")
                return True

        # Seleccionar símbolos para entrenar
        print(f"\n📊 Símbolos disponibles: {self.trainer.pairs}")
        train_symbol = input("🎯 Entrenar todos los símbolos? (y) o seleccionar uno (símbolo): ").upper().strip()

        if train_symbol == 'Y':
            # Entrenar todos los símbolos
            print("\n🚀 Entrenando modelos de ensamble para todos los símbolos...")
            results = {}

            for symbol in self.trainer.pairs:
                print(f"\n{'='*20} ENTRENANDO {symbol} {'='*20}")
                success = await self.trainer.train_ensemble_models(symbol)
                results[symbol] = success

                if success:
                    print(f"✅ {symbol}: Ensamble completado exitosamente")
                else:
                    print(f"❌ {symbol}: Error en el entrenamiento")

            # Resumen final
            print(f"\n🎯 RESUMEN FINAL DEL ENTRENAMIENTO:")
            print("=" * 60)
            successful = 0
            for symbol, success in results.items():
                status = "✅ ÉXITO" if success else "❌ FALLO"
                print(f"   {symbol}: {status}")
                if success:
                    successful += 1

            print(f"\n🏆 Modelos entrenados exitosamente: {successful}/{len(results)}")
            return successful > 0

        elif train_symbol in self.trainer.pairs:
            # Entrenar solo un símbolo
            print(f"\n🚀 Entrenando ensamble para {train_symbol}...")
            success = await self.trainer.train_ensemble_models(train_symbol)

            if success:
                print(f"✅ {train_symbol}: Ensamble completado exitosamente")
                return True
            else:
                print(f"❌ {train_symbol}: Error en el entrenamiento")
                return False
        else:
            print(f"❌ Símbolo no válido: {train_symbol}")
            return False

    def check_existing_models(self):
        """📦 Verificar modelos existentes"""
        existing = {}

        for symbol in self.trainer.pairs:
            symbol_models = []
            for timeframe in ['1m', '5m']:
                model_dir = f'models/ensemble_{timeframe}_{symbol.lower()}'
                if os.path.exists(model_dir) and os.path.exists(f'{model_dir}/best_model.h5'):
                    symbol_models.append(timeframe)

            if symbol_models:
                existing[symbol] = symbol_models

        return existing

    async def step2_test_ensemble_predictions(self):
        """🔮 Paso 2: Probar predicciones de ensamble"""

        print("\n" + "=" * 80)
        print("🔮 PASO 2: PRUEBA DE PREDICCIONES DE ENSAMBLE")
        print("=" * 80)

        # Cargar modelos
        if not self.predictor.load_ensemble_models():
            print("❌ No se pudieron cargar los modelos de ensamble")
            print("💡 Ejecuta primero el Paso 1 para entrenar los modelos")
            return False

        print("\n📊 Métodos de combinación disponibles:")
        print("   1. weighted_average - Promedio ponderado por accuracy y timeframe")
        print("   2. confidence_based - Selección basada en confianza")
        print("   3. consensus - Decisión por consenso/mayoría")

        # Probar con un símbolo
        test_symbol = input("\n🎯 Símbolo para prueba (BTCUSDT): ").upper().strip() or "BTCUSDT"

        if test_symbol not in self.predictor.symbols:
            print(f"❌ Símbolo no válido: {test_symbol}")
            return False

        print(f"\n🔮 Probando predicciones ensemble para {test_symbol}...")

        # Probar los tres métodos
        methods = ['weighted_average', 'confidence_based', 'consensus']

        for i, method in enumerate(methods, 1):
            print(f"\n{'-'*20} MÉTODO {i}: {method.upper()} {'-'*20}")

            try:
                result = await self.predictor.predict_ensemble(test_symbol, method)

                if result:
                    print(f"✅ Señal: {result['ensemble_signal']}")
                    print(f"✅ Confianza: {result['ensemble_confidence']:.3f}")
                    print(f"✅ Método usado: {result['combination_method']}")

                    # Mostrar predicciones individuales
                    if 'individual_predictions' in result:
                        print("📊 Predicciones individuales:")
                        for pred in result['individual_predictions']:
                            timeframe = pred['timeframe']
                            signal = pred['signal']
                            # ✅ CORRECCIÓN: Verificar que existe 'confidence' antes de acceder
                            confidence = pred.get('confidence', 0.5)
                            print(f"   - {timeframe}: {signal} ({confidence:.3f})")
                else:
                    print(f"❌ Error en predicción con método {method}")

            except Exception as e:
                print(f"❌ Error: {e}")

        return True

    async def step3_full_market_analysis(self):
        """📈 Paso 3: Análisis completo del mercado"""

        print("\n" + "=" * 80)
        print("📈 PASO 3: ANÁLISIS COMPLETO DEL MERCADO")
        print("=" * 80)

        # Cargar modelos si no están cargados
        if not hasattr(self.predictor, 'models') or not self.predictor.models:
            if not self.predictor.load_ensemble_models():
                print("❌ No se pudieron cargar los modelos")
                return False

        # Seleccionar método de combinación
        print("\n🎯 Selecciona método de combinación:")
        print("   1. weighted_average (recomendado)")
        print("   2. confidence_based")
        print("   3. consensus")

        method_choice = input("Método (1-3): ").strip() or "1"
        method_map = {
            "1": "weighted_average",
            "2": "confidence_based",
            "3": "consensus"
        }

        selected_method = method_map.get(method_choice, "weighted_average")
        print(f"🔄 Usando método: {selected_method}")

        # Generar predicciones para todos los símbolos
        print(f"\n🚀 Generando análisis completo del mercado...")
        results = await self.predictor.predict_all_symbols(selected_method)

        if results:
            print(f"\n🏆 ANÁLISIS COMPLETO DEL MERCADO")
            print("=" * 70)
            print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔄 Método: {selected_method}")
            print()

            # Mostrar resultados organizados
            buy_signals = []
            hold_signals = []
            sell_signals = []

            for symbol, result in results.items():
                signal = result['ensemble_signal']
                confidence = result['ensemble_confidence']

                if signal == 'BUY':
                    buy_signals.append((symbol, confidence))
                elif signal == 'SELL':
                    sell_signals.append((symbol, confidence))
                else:
                    hold_signals.append((symbol, confidence))

            # Ordenar por confianza
            buy_signals.sort(key=lambda x: x[1], reverse=True)
            sell_signals.sort(key=lambda x: x[1], reverse=True)
            hold_signals.sort(key=lambda x: x[1], reverse=True)

            print("🟢 SEÑALES DE COMPRA:")
            if buy_signals:
                for symbol, confidence in buy_signals:
                    print(f"   📈 {symbol}: BUY ({confidence:.3f})")
            else:
                print("   - Ninguna")

            print("\n🔴 SEÑALES DE VENTA:")
            if sell_signals:
                for symbol, confidence in sell_signals:
                    print(f"   📉 {symbol}: SELL ({confidence:.3f})")
            else:
                print("   - Ninguna")

            print("\n🟡 SEÑALES DE MANTENER:")
            if hold_signals:
                for symbol, confidence in hold_signals:
                    print(f"   ⏸️ {symbol}: HOLD ({confidence:.3f})")
            else:
                print("   - Ninguna")

            # Estadísticas
            total_symbols = len(results)
            print(f"\n📊 RESUMEN ESTADÍSTICO:")
            print(f"   - Total símbolos analizados: {total_symbols}")
            print(f"   - Señales BUY: {len(buy_signals)} ({len(buy_signals)/total_symbols*100:.1f}%)")
            print(f"   - Señales SELL: {len(sell_signals)} ({len(sell_signals)/total_symbols*100:.1f}%)")
            print(f"   - Señales HOLD: {len(hold_signals)} ({len(hold_signals)/total_symbols*100:.1f}%)")

            # Confianza promedio
            avg_confidence = sum(r['ensemble_confidence'] for r in results.values()) / len(results)
            print(f"   - Confianza promedio: {avg_confidence:.3f}")

            return True
        else:
            print("❌ No se pudieron generar predicciones")
            return False

    async def run_full_demo(self):
        """🎯 Ejecutar demo completo"""

        print("🎯 SISTEMA DE ENSAMBLE TCN - DEMO COMPLETO")
        print("=" * 80)
        print("🔄 Este demo te guiará a través del sistema completo:")
        print("   1. Entrenamiento de modelos de ensamble (1m + 5m)")
        print("   2. Prueba de predicciones individuales")
        print("   3. Análisis completo del mercado")
        print("=" * 80)

        # Verificar si saltar pasos
        skip_training = input("\n🤔 ¿Saltar entrenamiento si ya hay modelos? (y/n): ").lower().strip() == 'y'

        # Paso 1: Entrenamiento
        if skip_training and self.check_existing_models():
            print("⏭️ Saltando entrenamiento - usando modelos existentes")
            step1_success = True
        else:
            step1_success = await self.step1_train_ensemble_models()

        if not step1_success:
            print("❌ Error en el entrenamiento. Terminando demo.")
            return

        # Paso 2: Pruebas
        print("\n" + "⏯️ Continuando con pruebas...")
        step2_success = await self.step2_test_ensemble_predictions()

        if not step2_success:
            print("❌ Error en las pruebas. Saltando análisis completo.")

        # Paso 3: Análisis completo
        print("\n" + "⏯️ Continuando con análisis completo...")
        step3_success = await self.step3_full_market_analysis()

        # Resumen final
        print("\n" + "=" * 80)
        print("🏁 DEMO COMPLETADO")
        print("=" * 80)
        print(f"   ✅ Paso 1 - Entrenamiento: {'OK' if step1_success else 'FALLO'}")
        print(f"   ✅ Paso 2 - Pruebas: {'OK' if step2_success else 'FALLO'}")
        print(f"   ✅ Paso 3 - Análisis: {'OK' if step3_success else 'FALLO'}")

        if step1_success and step2_success and step3_success:
            print("\n🎉 ¡DEMO EXITOSO! El sistema de ensamble está listo para usar.")
            print("\n💡 Próximos pasos:")
            print("   - Integrar el predictor de ensamble en tu trading manager")
            print("   - Configurar alertas basadas en señales del ensamble")
            print("   - Monitorear el rendimiento en tiempo real")
        else:
            print("\n⚠️ Demo parcialmente exitoso. Revisa los errores arriba.")


async def main():
    """🚀 Función principal"""

    try:
        demo = EnsembleTradingDemo()
        await demo.run_full_demo()

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en el demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Verificar dependencias
    try:
        import tensorflow as tf
        import aiohttp
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import RobustScaler
        print("✅ Todas las dependencias están disponibles")
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
        sys.exit(1)

    # Ejecutar demo
    asyncio.run(main())
