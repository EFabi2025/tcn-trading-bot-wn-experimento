#!/usr/bin/env python3
"""
🧪 TEST IMPROVED ENSEMBLE - PRUEBA DE MEJORAS
Script para probar las mejoras implementadas en el ensemble trainer
"""

import asyncio
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tcn_ensemble_trainer import TCNEnsembleTrainer


async def test_improved_ensemble():
    """🧪 Probar las mejoras del ensemble trainer"""
    
    print("🧪 TEST IMPROVED ENSEMBLE TRAINER")
    print("=" * 60)
    print("🎯 Probando mejoras implementadas:")
    print("   ✅ Thresholds fijos más rentables")
    print("   ✅ Filtros técnicos mejorados")
    print("   ✅ Arquitectura TCN más robusta")
    print("   ✅ Callbacks más pacientes")
    print("   ✅ Configuración de entrenamiento mejorada")
    print("=" * 60)
    
    # Configuración de prueba optimizada
    config = {
        'symbol': 'XRPUSDT',
        'timeframe': '5m',
        'days': 30,  # Datos suficientes para prueba
        'start_time': None,
        'end_time': None
    }
    
    print(f"\n📊 CONFIGURACIÓN DE PRUEBA:")
    print(f"   - Símbolo: {config['symbol']}")
    print(f"   - Timeframe: {config['timeframe']}")
    print(f"   - Días: {config['days']}")
    
    # Crear trainer con configuración
    trainer = TCNEnsembleTrainer(config)
    
    # Entrenar modelo
    print(f"\n🚀 INICIANDO PRUEBA DE ENTRENAMIENTO...")
    success = await trainer.train_ensemble_models(config['symbol'])
    
    if success:
        print(f"\n✅ PRUEBA EXITOSA:")
        print(f"   - Modelo entrenado correctamente")
        print(f"   - Mejoras implementadas funcionando")
        print(f"   - Verificar accuracy en logs")
    else:
        print(f"\n❌ PRUEBA FALLIDA:")
        print(f"   - Revisar errores en el entrenamiento")
        print(f"   - Verificar configuración")
    
    return success


async def main():
    """🎯 Función principal de prueba"""
    
    print("🎯 TEST IMPROVED ENSEMBLE TRAINER")
    print("=" * 80)
    
    try:
        success = await test_improved_ensemble()
        
        if success:
            print(f"\n🎉 ¡PRUEBA COMPLETADA EXITOSAMENTE!")
            print(f"📊 Revisa los resultados en models/definitivo_5m_xrpusdt/")
        else:
            print(f"\n❌ PRUEBA FALLIDA - Revisar errores")
            
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 