#!/usr/bin/env python3
"""
🔧 FIX PARA MODELOS CON FEATURES FALTANTES
==========================================

Script para reentrenar los modelos que tienen 62 features en lugar de 66.
Esto solucionará el problema de los modelos "dormidos".

Modelos problemáticos:
- definitivo_v3_3m_dotusdt (62 features)
- definitivo_v3_5m_ethusdt (62 features) 
- definitivo_v3_dotusdt (62 features)
- refactored_v3_3m_ethusdt (62 features)

Features faltantes:
- higher_high
- lower_low
- resistance_touch
- support_touch
"""

import sys
import os
import shutil
from datetime import datetime

def fix_missing_features_models():
    """🔧 Reentrenar modelos con features faltantes"""
    
    print("🔧 FIX PARA MODELOS CON FEATURES FALTANTES")
    print("=" * 50)
    
    # Modelos problemáticos identificados
    problematic_models = [
        'definitivo_v3_3m_dotusdt',
        'definitivo_v3_5m_ethusdt', 
        'definitivo_v3_dotusdt',
        'refactored_v3_3m_ethusdt'
    ]
    
    print(f"\n📋 MODELOS QUE NECESITAN REENTRENAMIENTO:")
    for model in problematic_models:
        print(f"   - {model}")
    
    print(f"\n🔍 FEATURES FALTANTES:")
    missing_features = ['higher_high', 'lower_low', 'resistance_touch', 'support_touch']
    for feature in missing_features:
        print(f"   - {feature}")
    
    # Crear backup de modelos problemáticos
    backup_dir = f"backup_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"\n💾 CREANDO BACKUP EN {backup_dir}...")
    
    for model in problematic_models:
        model_path = f"models/{model}"
        if os.path.exists(model_path):
            backup_path = f"{backup_dir}/{model}"
            shutil.copytree(model_path, backup_path)
            print(f"   ✅ Backup creado: {model}")
        else:
            print(f"   ⚠️ Modelo no encontrado: {model}")
    
    print(f"\n📋 INSTRUCCIONES PARA SOLUCIONAR:")
    print(f"   1. Ejecutar reentrenamiento:")
    print(f"      python retrain_missing_features.py")
    print(f"   2. Verificar que se corrigió:")
    print(f"      python verify_retrained_models.py")
    print(f"   3. Probar el predictor:")
    print(f"      python simple_professional_manager_v2.py")
    print(f"")
    print(f"💾 Backup creado en: {backup_dir}")
    print(f"🔄 Los modelos problemáticos serán reentrenados con 66 features")
    print(f"🎯 Esto debería resolver el problema de modelos 'dormidos'")

if __name__ == "__main__":
    fix_missing_features_models() 