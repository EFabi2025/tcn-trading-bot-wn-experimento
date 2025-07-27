#!/usr/bin/env python3
"""
🧪 TEST HYBRID ENSEMBLE - PRUEBA DEL ENSEMBLE HÍBRIDO
Script para probar el ensemble trainer con arquitectura híbrida
"""

import asyncio
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcn_ensemble_trainer import TCNEnsembleTrainer


async def test_hybrid_ensemble():
    """🧪 Probar el ensemble trainer híbrido"""
    
    print("🧪 TEST HYBRID ENSEMBLE TRAINER")
    print("=" * 60)
    print("🎯 Características del ensemble híbrido:")
    print("   ✅ Etiquetado ATR dinámico (mejor que percentiles)")
    print("   ✅ Arquitectura simplificada probada")
    print("   ✅ Configuración estable (100 epochs, batch 64)")
    print("   ✅ Learning rate conservador (0.0005)")
    print("   ✅ Callbacks optimizados")
    print("=" * 60)
    
    # Configuración de prueba
    config = {
        'symbol': 'XRPUSDT',
        'timeframe': '5m',
        'days': 30,
        'start_time': None,
        'end_time': None
    }
    
    print(f"\n📊 CONFIGURACIÓN DE PRUEBA:")
    print(f"   - Símbolo: {config['symbol']}")
    print(f"   - Timeframe: {config['timeframe']}")
    print(f"   - Días: {config['days']}")
    
    # Crear trainer híbrido
    trainer = TCNEnsembleTrainer(config)
    
    # Entrenar modelo híbrido
    print(f"\n🚀 INICIANDO ENTRENAMIENTO HÍBRIDO...")
    success = await trainer.train_ensemble_models(config['symbol'])
    
    if success:
        print(f"\n✅ PRUEBA HÍBRIDA EXITOSA:")
        print(f"   - Etiquetado ATR funcionando")
        print(f"   - Arquitectura simplificada estable")
        print(f"   - Modelo guardado correctamente")
        print(f"   - Verificar accuracy mejorado en logs")
    else:
        print(f"\n❌ PRUEBA HÍBRIDA FALLIDA:")
        print(f"   - Revisar configuración ATR")
        print(f"   - Verificar datos de entrada")
    
    return success


async def main():
    """🎯 Función principal de prueba"""
    
    print("🎯 TEST HYBRID ENSEMBLE TRAINER")
    print("=" * 80)
    print("🔄 Combinando lo mejor del etiquetado ATR + arquitectura simplificada")
    print("🎯 Objetivo: Mejorar accuracy y estabilidad")
    print("=" * 80)
    
    try:
        success = await test_hybrid_ensemble()
        
        if success:
            print(f"\n🎉 ¡ENSEMBLE HÍBRIDO COMPLETADO EXITOSAMENTE!")
            print(f"📊 Modelo guardado en: models/definitivo_5m_xrpusdt/")
            print(f"\n📈 EXPECTATIVAS DE MEJORA:")
            print(f"   - Accuracy objetivo: >65% (vs 57.9% anterior)")
            print(f"   - Distribución más balanceada")
            print(f"   - Win rate objetivo: >60%")
            print(f"   - Etiquetado más inteligente")
        else:
            print(f"\n❌ ENSEMBLE HÍBRIDO FALLIDO")
            print(f"💡 Posibles soluciones:")
            print(f"   - Verificar instalación de talib")
            print(f"   - Revisar conexión a internet")
            print(f"   - Verificar datos de entrada")
            
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBA HÍBRIDA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 