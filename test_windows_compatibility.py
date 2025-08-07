#!/usr/bin/env python3
"""
🧪 TEST DE COMPATIBILIDAD CON WINDOWS
=====================================
Verifica que los scripts principales funcionen correctamente en Windows
"""

import os
import sys
import importlib
from pathlib import Path

def test_imports():
    """🧪 Probar importaciones de módulos"""
    print("🔍 Probando importaciones...")
    
    modules_to_test = [
        'tcn_ensemble_predictor',
        'simple_professional_managerv_2', 
        'advanced_risk_manager',
        'centralized_features_engine2',
        'config.trading_config'
    ]
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: OK")
        except Exception as e:
            print(f"❌ {module_name}: ERROR - {e}")
    
    print()

def test_dependencies():
    """🧪 Probar dependencias críticas"""
    print("🔍 Probando dependencias...")
    
    dependencies = [
        'tensorflow',
        'talib',
        'aiohttp',
        'pandas',
        'numpy',
        'asyncio',
        'dotenv'
    ]
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}: OK")
        except Exception as e:
            print(f"❌ {dep}: ERROR - {e}")
    
    print()

def test_directories():
    """🧪 Verificar directorios necesarios"""
    print("🔍 Verificando directorios...")
    
    required_dirs = ['models', 'cache', 'config']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/: Existe")
        else:
            print(f"❌ {dir_name}/: No existe")
    
    print()

def test_config_files():
    """🧪 Verificar archivos de configuración"""
    print("🔍 Verificando archivos de configuración...")
    
    config_files = [
        '.env',
        'config/trading_config.py',
        'config/trading_config.json'
    ]
    
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Existe")
        else:
            print(f"❌ {file_path}: No existe")
    
    print()

def test_path_compatibility():
    """🧪 Probar compatibilidad de rutas"""
    print("🔍 Probando compatibilidad de rutas...")
    
    # Probar rutas que usan los scripts
    test_paths = [
        'models/',
        'cache/',
        'models/adaptive_test',
        'cache/test_data.pkl'
    ]
    
    for path in test_paths:
        try:
            # Crear directorio temporal para prueba
            if path.endswith('/'):
                os.makedirs(path, exist_ok=True)
                print(f"✅ {path}: Compatible")
                # Limpiar
                if 'test' in path:
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
            else:
                # Crear archivo temporal
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    f.write('test')
                print(f"✅ {path}: Compatible")
                # Limpiar
                if 'test' in path:
                    os.remove(path)
        except Exception as e:
            print(f"❌ {path}: ERROR - {e}")
    
    print()

def test_script_initialization():
    """🧪 Probar inicialización de scripts"""
    print("🔍 Probando inicialización de scripts...")
    
    try:
        # Probar importación de TCN Ensemble Predictor
        from tcn_ensemble_predictor import TCNEnsemblePredictor
        predictor = TCNEnsemblePredictor()
        print("✅ TCNEnsemblePredictor: Inicialización OK")
    except Exception as e:
        print(f"❌ TCNEnsemblePredictor: ERROR - {e}")
    
    try:
        # Probar importación de Risk Manager
        from advanced_risk_manager import AdvancedRiskManager
        print("✅ AdvancedRiskManager: Importación OK")
    except Exception as e:
        print(f"❌ AdvancedRiskManager: ERROR - {e}")
    
    try:
        # Probar importación de Trading Manager
        from simple_professional_managerv_2 import SimpleProfessionalTradingManager
        print("✅ SimpleProfessionalTradingManager: Importación OK")
    except Exception as e:
        print(f"❌ SimpleProfessionalTradingManager: ERROR - {e}")
    
    print()

def main():
    """🎯 Función principal de pruebas"""
    print("🧪 TEST DE COMPATIBILIDAD CON WINDOWS")
    print("=" * 50)
    print()
    
    test_imports()
    test_dependencies()
    test_directories()
    test_config_files()
    test_path_compatibility()
    test_script_initialization()
    
    print("🎯 RESUMEN:")
    print("✅ Si todos los tests pasan, los scripts son compatibles con Windows")
    print("⚠️  Si hay errores, revisa las dependencias faltantes")
    print("📝 Para ejecutar los scripts, usa: python <script_name>.py")

if __name__ == "__main__":
    main() 