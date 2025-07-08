#!/usr/bin/env python3
"""
🚀 EJECUTOR DE ANÁLISIS DE SEÑALES
================================

Script de uso rápido para ejecutar análisis completo de señales de trading:
1. Validación del mapeo de señales
2. Análisis de coherencia con indicadores técnicos
3. Generación de reportes detallados

Uso:
    python run_signal_analysis.py [--quick] [--validation-only] [--analysis-only]
"""

import asyncio
import argparse
import sys
from datetime import datetime

# Importar nuestros analizadores
from validate_signal_mapping import SignalMappingValidator
from analyze_trading_signals import TradingSignalAnalyzer

async def run_validation_only():
    """✅ Ejecutar solo validación de mapeo"""
    print("🔍 EJECUTANDO VALIDACIÓN DE MAPEO DE SEÑALES")
    print("="*60)

    validator = SignalMappingValidator()
    success = await validator.run_full_validation()

    return success

async def run_analysis_only():
    """📊 Ejecutar solo análisis de señales"""
    print("📊 EJECUTANDO ANÁLISIS DE SEÑALES DE TRADING")
    print("="*60)

    analyzer = TradingSignalAnalyzer()
    results = await analyzer.run_full_analysis()

    return results is not None

async def run_full_analysis():
    """🚀 Ejecutar análisis completo"""
    print("🚀 EJECUTANDO ANÁLISIS COMPLETO DE SEÑALES DE TRADING")
    print("="*80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    try:
        # 1. Validación del mapeo de señales
        print("\n" + "🔍 FASE 1: VALIDACIÓN DEL MAPEO DE SEÑALES")
        print("-"*50)

        validator = SignalMappingValidator()
        validation_success = await validator.run_full_validation()

        if not validation_success:
            print("❌ VALIDACIÓN FALLIDA - DETENIENDO ANÁLISIS")
            print("   El mapeo de señales no está funcionando correctamente.")
            print("   Revisar las correcciones en tcn_definitivo_predictor.py")
            return False

        print("✅ VALIDACIÓN EXITOSA - CONTINUANDO CON ANÁLISIS")

        # 2. Análisis de coherencia con indicadores técnicos
        print("\n" + "📊 FASE 2: ANÁLISIS DE COHERENCIA CON INDICADORES TÉCNICOS")
        print("-"*60)

        analyzer = TradingSignalAnalyzer()
        analysis_results = await analyzer.run_full_analysis()

        if analysis_results is None:
            print("❌ ANÁLISIS FALLIDO")
            return False

        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")

        # 3. Resumen final
        print("\n" + "🎯 RESUMEN FINAL")
        print("-"*30)
        print("✅ Validación de mapeo: EXITOSA")
        print("✅ Análisis de coherencia: COMPLETADO")
        print("💾 Reportes generados y guardados")
        print(f"⏰ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"❌ ERROR EN ANÁLISIS COMPLETO: {e}")
        return False

async def run_quick_check():
    """⚡ Ejecutar verificación rápida"""
    print("⚡ EJECUTANDO VERIFICACIÓN RÁPIDA")
    print("="*50)

    try:
        # Solo validar mapeo básico
        validator = SignalMappingValidator()

        # Validar consistencia del mapeo
        mapping_valid = validator.validate_signal_mapping_consistency()

        if mapping_valid:
            print("\n✅ VERIFICACIÓN RÁPIDA EXITOSA")
            print("   El mapeo de señales está configurado correctamente")
            return True
        else:
            print("\n❌ VERIFICACIÓN RÁPIDA FALLIDA")
            print("   Problemas detectados en el mapeo de señales")
            return False

    except Exception as e:
        print(f"❌ ERROR EN VERIFICACIÓN RÁPIDA: {e}")
        return False

def print_usage():
    """📋 Mostrar instrucciones de uso"""
    print("""
🚀 ANALIZADOR DE SEÑALES DE TRADING TCN
====================================

Opciones disponibles:

1. Análisis completo (recomendado):
   python run_signal_analysis.py

2. Solo validación de mapeo:
   python run_signal_analysis.py --validation-only

3. Solo análisis de coherencia:
   python run_signal_analysis.py --analysis-only

4. Verificación rápida:
   python run_signal_analysis.py --quick

📊 El análisis completo incluye:
   • Validación del mapeo de señales TCN
   • Análisis de coherencia con indicadores técnicos
   • Comparación de señales por símbolo
   • Generación de reportes detallados
   • Estadísticas de rendimiento

⏱️ Tiempo estimado: 2-5 minutos
""")

async def main():
    """🎯 Función principal"""
    parser = argparse.ArgumentParser(description='Analizador de señales de trading TCN')
    parser.add_argument('--quick', action='store_true', help='Verificación rápida')
    parser.add_argument('--validation-only', action='store_true', help='Solo validación de mapeo')
    parser.add_argument('--analysis-only', action='store_true', help='Solo análisis de coherencia')
    parser.add_argument('--help-usage', action='store_true', help='Mostrar instrucciones detalladas')

    args = parser.parse_args()

    if args.help_usage:
        print_usage()
        return

    try:
        if args.quick:
            success = await run_quick_check()
        elif args.validation_only:
            success = await run_validation_only()
        elif args.analysis_only:
            success = await run_analysis_only()
        else:
            success = await run_full_analysis()

        # Código de salida
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️ Análisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
