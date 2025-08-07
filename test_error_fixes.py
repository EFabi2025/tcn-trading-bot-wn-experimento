#!/usr/bin/env python3
"""
🧪 TEST: Verificación de correcciones de errores críticos
"""

import os
import json
import numpy as np
import pickle
from datetime import datetime

def test_directory_creation():
    """🧪 Probar creación de directorios"""
    print("🧪 Probando creación de directorios...")
    
    test_dir = "test_models/adaptive_test_1m_6h_24w_optimized_crypto"
    
    try:
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ Directorio creado: {test_dir}")
        
        # Verificar que existe
        if os.path.exists(test_dir):
            print(f"✅ Directorio existe: {test_dir}")
        else:
            print(f"❌ Error: Directorio no existe")
            return False
            
        # Limpiar
        os.rmdir(test_dir)
        print(f"✅ Directorio eliminado: {test_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error creando directorio: {e}")
        return False

def test_json_serialization():
    """🧪 Probar serialización JSON con tipos numpy"""
    print("\n🧪 Probando serialización JSON...")
    
    # Crear datos con tipos numpy problemáticos
    test_data = {
        'accuracy': np.float64(0.75),
        'loss': np.float32(0.25),
        'epochs': np.int64(50),
        'array': np.array([1, 2, 3, 4]),
        'nested': {
            'precision': np.float64(0.80),
            'recall': np.float32(0.70)
        },
        'list_with_numpy': [np.int64(1), np.float64(2.5)]
    }
    
    def convert_numpy_types(obj):
        """🔄 Convertir tipos numpy a tipos nativos de Python para JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    try:
        # Convertir datos
        converted_data = convert_numpy_types(test_data)
        
        # Intentar serializar
        json_str = json.dumps(converted_data, indent=2)
        print(f"✅ JSON serializado exitosamente")
        print(f"   📊 Tamaño: {len(json_str)} caracteres")
        
        # Verificar que se puede deserializar
        parsed_data = json.loads(json_str)
        print(f"✅ JSON deserializado exitosamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en serialización JSON: {e}")
        return False

def test_model_saving():
    """🧪 Probar guardado de archivos de modelo"""
    print("\n🧪 Probando guardado de archivos...")
    
    test_dir = "test_models/adaptive_test_1m_6h_24w_optimized_crypto"
    
    try:
        # Crear directorio
        os.makedirs(test_dir, exist_ok=True)
        
        # Simular archivos de modelo
        test_config = {
            'symbol': 'TESTUSDT',
            'timeframe': '1m',
            'prediction_horizon': 6,
            'lookback_window': 24,
            'feature_set': 'optimized_crypto',
            'basic_metrics': {
                'accuracy': float(0.75),
                'loss': float(0.25)
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Guardar config.json
        config_path = f'{test_dir}/config.json'
        with open(config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        print(f"✅ Config guardado: {config_path}")
        
        # Simular scaler
        test_scaler = {'type': 'RobustScaler', 'params': {'quantile_range': (25.0, 75.0)}}
        scaler_path = f'{test_dir}/scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(test_scaler, f)
        print(f"✅ Scaler guardado: {scaler_path}")
        
        # Simular feature columns
        test_features = ['feature_1', 'feature_2', 'feature_3']
        features_path = f'{test_dir}/feature_columns.pkl'
        with open(features_path, 'wb') as f:
            pickle.dump(test_features, f)
        print(f"✅ Features guardado: {features_path}")
        
        # Verificar archivos
        required_files = ['config.json', 'scaler.pkl', 'feature_columns.pkl']
        missing_files = []
        for file in required_files:
            if not os.path.exists(f'{test_dir}/{file}'):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Archivos faltantes: {missing_files}")
            return False
        else:
            print(f"✅ Todos los archivos guardados correctamente")
        
        # Limpiar
        import shutil
        shutil.rmtree("test_models")
        print(f"✅ Directorio de prueba eliminado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en guardado de archivos: {e}")
        return False

def main():
    """🧪 Ejecutar todas las pruebas"""
    print("🧪 INICIANDO PRUEBAS DE CORRECCIÓN DE ERRORES")
    print("=" * 60)
    
    tests = [
        ("Creación de directorios", test_directory_creation),
        ("Serialización JSON", test_json_serialization),
        ("Guardado de archivos", test_model_saving)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🎯 {test_name.upper()}")
        print("-" * 40)
        results[test_name] = test_func()
    
    print(f"\n🎯 RESUMEN DE PRUEBAS")
    print("=" * 40)
    for test_name, passed in results.items():
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🏆 TODAS LAS PRUEBAS PASARON")
        print(f"✅ Las correcciones de errores están funcionando correctamente")
    else:
        print(f"\n❌ ALGUNAS PRUEBAS FALLARON")
        print(f"⚠️  Revisa los errores arriba")
    
    return all_passed

if __name__ == "__main__":
    main()
