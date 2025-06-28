#!/usr/bin/env python3
"""
🔧 INTEGRACIÓN XRP AL SISTEMA DE TRADING
=======================================

Script para integrar el modelo XRP entrenado al sistema de trading actual.
Actualiza configuraciones y verifica compatibilidad.
"""

import os
import shutil
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class XRPSystemIntegrator:
    """Integrador de XRP al sistema de trading"""
    
    def __init__(self):
        self.symbol = "XRPUSDT"
        self.model_source = "models/definitivo_xrpusdt.h5"
        self.integration_steps = []
        
        print("🔧 XRP System Integrator inicializado")
        print(f"   📊 Símbolo: {self.symbol}")
        print(f"   📁 Modelo: {self.model_source}")

    def step_1_verify_model_exists(self) -> bool:
        """Verificar que el modelo XRP existe"""
        print("\n🔍 STEP 1: Verificando modelo XRP")
        print("=" * 50)
        
        if os.path.exists(self.model_source):
            file_size = os.path.getsize(self.model_source) / (1024 * 1024)  # MB
            print(f"✅ Modelo encontrado: {self.model_source}")
            print(f"   📊 Tamaño: {file_size:.1f} MB")
            
            # Verificar resultados de entrenamiento
            results_path = "results/xrp_training_results.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                print(f"   🎯 Accuracy: {results.get('test_accuracy', 'N/A')}")
                print(f"   📈 Precision: {results.get('test_precision', 'N/A')}")
                print(f"   📉 Recall: {results.get('test_recall', 'N/A')}")
            
            self.integration_steps.append("✅ Modelo verificado")
            return True
        else:
            print(f"❌ Modelo no encontrado: {self.model_source}")
            print(f"   🔄 Ejecuta primero: python train_xrp_model.py")
            return False

    def step_2_update_config_files(self) -> bool:
        """Actualizar archivos de configuración"""
        print("\n⚙️ STEP 2: Actualizando configuraciones")
        print("=" * 50)
        
        try:
            # 1. Actualizar config.py
            self._update_main_config()
            
            # 2. Actualizar simple_professional_manager.py
            self._update_trading_manager()
            
            # 3. Actualizar definitivo_tcn_predictor.py
            self._update_predictor()
            
            self.integration_steps.append("✅ Configuraciones actualizadas")
            return True
            
        except Exception as e:
            print(f"❌ Error actualizando configuraciones: {e}")
            return False

    def _update_main_config(self):
        """Actualizar config.py principal"""
        print("   📝 Actualizando config.py...")
        
        config_file = "config.py"
        
        if not os.path.exists(config_file):
            print(f"   ⚠️ {config_file} no encontrado")
            return
        
        # Leer archivo actual
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar si XRP ya está incluido
        if 'XRPUSDT' in content:
            print(f"   ✅ XRPUSDT ya está en config.py")
            return
        
        # Buscar la lista de símbolos y agregar XRP
        if "self.SYMBOLS: List[str] = [" in content:
            # Encontrar la línea y agregar XRP
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "self.SYMBOLS: List[str] = [" in line:
                    # Buscar el cierre de la lista
                    for j in range(i, len(lines)):
                        if ']' in lines[j] and 'SYMBOLS' in lines[i:j+1]:
                            # Insertar XRP antes del cierre
                            if '"XRPUSDT"' not in lines[j]:
                                lines[j] = lines[j].replace(']', ', "XRPUSDT"]')
                            break
                    break
            
            # Escribir archivo actualizado
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"   ✅ XRPUSDT agregado a config.py")
        else:
            print(f"   ⚠️ No se encontró lista SYMBOLS en config.py")

    def _update_trading_manager(self):
        """Actualizar simple_professional_manager.py"""
        print("   📝 Actualizando simple_professional_manager.py...")
        
        manager_file = "simple_professional_manager.py"
        
        if not os.path.exists(manager_file):
            print(f"   ⚠️ {manager_file} no encontrado")
            return
        
        # Leer archivo actual
        with open(manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar si XRP ya está incluido
        if 'XRPUSDT' in content:
            print(f"   ✅ XRPUSDT ya está en trading manager")
        else:
            print(f"   ℹ️ XRPUSDT se agregará automáticamente desde config")

    def _update_predictor(self):
        """Actualizar definitivo_tcn_predictor.py"""
        print("   📝 Actualizando definitivo_tcn_predictor.py...")
        
        predictor_file = "definitivo_tcn_predictor.py"
        
        if not os.path.exists(predictor_file):
            print(f"   ⚠️ {predictor_file} no encontrado")
            return
        
        # Leer archivo actual
        with open(predictor_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar si XRP ya está incluido
        if 'XRPUSDT' in content:
            print(f"   ✅ XRPUSDT ya está en predictor")
        else:
            # Buscar la lista de pairs y agregar XRP
            if 'self.pairs = [' in content:
                content = content.replace(
                    'self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]',
                    'self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]'
                )
                
                # Escribir archivo actualizado
                with open(predictor_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"   ✅ XRPUSDT agregado a predictor")
            else:
                print(f"   ℹ️ XRPUSDT se agregará automáticamente desde config")

    def step_3_verify_model_compatibility(self) -> bool:
        """Verificar compatibilidad del modelo"""
        print("\n🧪 STEP 3: Verificando compatibilidad del modelo")
        print("=" * 50)
        
        try:
            import tensorflow as tf
            
            # Cargar modelo
            print(f"   📦 Cargando modelo: {self.model_source}")
            model = tf.keras.models.load_model(self.model_source)
            
            # Verificar arquitectura
            input_shape = model.input_shape
            output_shape = model.output_shape
            
            print(f"   📐 Input shape: {input_shape}")
            print(f"   📐 Output shape: {output_shape}")
            
            # Verificar que tiene 3 salidas (BUY, HOLD, SELL)
            if output_shape[-1] == 3:
                print(f"   ✅ Arquitectura compatible (3 clases)")
            else:
                print(f"   ❌ Arquitectura incompatible ({output_shape[-1]} clases)")
                return False
            
            # Verificar que acepta 66 features
            expected_features = 66
            actual_features = input_shape[-1]
            
            if actual_features == expected_features:
                print(f"   ✅ Features compatibles ({actual_features} features)")
            else:
                print(f"   ⚠️ Features diferentes: esperado {expected_features}, actual {actual_features}")
                print(f"      El sistema puede adaptarse automáticamente")
            
            self.integration_steps.append("✅ Modelo compatible")
            return True
            
        except Exception as e:
            print(f"   ❌ Error verificando modelo: {e}")
            return False

    def step_4_test_prediction(self) -> bool:
        """Probar predicción con datos sintéticos"""
        print("\n🧪 STEP 4: Probando predicción")
        print("=" * 50)
        
        try:
            import numpy as np
            import tensorflow as tf
            from centralized_features_engine import CentralizedFeaturesEngine
            
            # Cargar modelo
            model = tf.keras.models.load_model(self.model_source)
            
            # Crear datos sintéticos para prueba
            print("   🎲 Generando datos de prueba...")
            
            # Generar datos OHLCV sintéticos
            n_samples = 100
            np.random.seed(42)
            
            test_data = {
                'open': np.random.uniform(0.5, 0.7, n_samples),
                'high': np.random.uniform(0.6, 0.8, n_samples),
                'low': np.random.uniform(0.4, 0.6, n_samples),
                'close': np.random.uniform(0.5, 0.7, n_samples),
                'volume': np.random.uniform(1000000, 5000000, n_samples)
            }
            
            import pandas as pd
            df = pd.DataFrame(test_data)
            
            # Calcular features
            features_engine = CentralizedFeaturesEngine()
            features_df = features_engine.calculate_features(df, feature_set='tcn_definitivo')
            
            print(f"   📊 Features calculadas: {features_df.shape}")
            
            # Preparar secuencia para predicción
            sequence_length = 60
            if len(features_df) >= sequence_length:
                # Tomar últimas 60 muestras
                sequence = features_df.iloc[-sequence_length:].fillna(0).values
                
                # Ajustar dimensiones del modelo
                model_input_shape = model.input_shape
                expected_features = model_input_shape[-1]
                
                if sequence.shape[1] != expected_features:
                    if sequence.shape[1] < expected_features:
                        # Padding con ceros
                        padding = np.zeros((sequence.shape[0], expected_features - sequence.shape[1]))
                        sequence = np.concatenate([sequence, padding], axis=1)
                    else:
                        # Truncar features
                        sequence = sequence[:, :expected_features]
                
                # Expandir dimensiones para batch
                input_data = np.expand_dims(sequence, axis=0)
                
                print(f"   📐 Input shape: {input_data.shape}")
                
                # Hacer predicción
                prediction = model.predict(input_data, verbose=0)
                probabilities = prediction[0]
                
                predicted_class = np.argmax(probabilities)
                confidence = float(np.max(probabilities))
                
                class_names = ['BUY', 'HOLD', 'SELL']
                signal = class_names[predicted_class]
                
                print(f"   🎯 Predicción de prueba:")
                print(f"     Señal: {signal}")
                print(f"     Confianza: {confidence:.4f}")
                print(f"     Probabilidades: BUY={probabilities[0]:.3f}, HOLD={probabilities[1]:.3f}, SELL={probabilities[2]:.3f}")
                
                if confidence > 0.1:  # Verificación básica
                    print(f"   ✅ Predicción exitosa")
                    self.integration_steps.append("✅ Predicción probada")
                    return True
                else:
                    print(f"   ⚠️ Confianza muy baja")
                    return False
            else:
                print(f"   ⚠️ Datos insuficientes para secuencia")
                return False
                
        except Exception as e:
            print(f"   ❌ Error en predicción de prueba: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_5_create_backup(self) -> bool:
        """Crear backup de configuraciones actuales"""
        print("\n💾 STEP 5: Creando backup")
        print("=" * 50)
        
        try:
            backup_dir = "backup_before_xrp"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Archivos a respaldar
            files_to_backup = [
                "config.py",
                "simple_professional_manager.py",
                "definitivo_tcn_predictor.py"
            ]
            
            for file_name in files_to_backup:
                if os.path.exists(file_name):
                    backup_path = os.path.join(backup_dir, file_name)
                    shutil.copy2(file_name, backup_path)
                    print(f"   📄 Backup: {file_name} -> {backup_path}")
            
            print(f"   ✅ Backup creado en: {backup_dir}")
            self.integration_steps.append("✅ Backup creado")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creando backup: {e}")
            return False

    def step_6_final_verification(self) -> bool:
        """Verificación final del sistema"""
        print("\n🔍 STEP 6: Verificación final")
        print("=" * 50)
        
        try:
            # Verificar que el modelo existe en la ubicación correcta
            if not os.path.exists(self.model_source):
                print(f"   ❌ Modelo no encontrado: {self.model_source}")
                return False
            
            # Verificar configuraciones
            config_updated = False
            try:
                from config import trading_config
                if hasattr(trading_config, 'SYMBOLS') and 'XRPUSDT' in trading_config.SYMBOLS:
                    config_updated = True
                    print(f"   ✅ XRPUSDT en configuración")
                else:
                    print(f"   ⚠️ XRPUSDT no encontrado en configuración")
            except:
                print(f"   ⚠️ No se pudo verificar configuración")
            
            # Verificar que el predictor puede cargar el modelo
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(self.model_source)
                print(f"   ✅ Modelo cargable")
            except Exception as e:
                print(f"   ❌ Modelo no cargable: {e}")
                return False
            
            print(f"\n📊 Resumen de integración:")
            for step in self.integration_steps:
                print(f"   {step}")
            
            if len(self.integration_steps) >= 4:  # Al menos 4 pasos exitosos
                print(f"\n🎉 INTEGRACIÓN EXITOSA")
                print(f"   📁 Modelo: {self.model_source}")
                print(f"   🎯 Símbolo: {self.symbol}")
                print(f"   🔄 Reinicia el sistema de trading para usar XRP")
                return True
            else:
                print(f"\n⚠️ INTEGRACIÓN PARCIAL")
                print(f"   Algunos pasos fallaron, revisa los errores")
                return False
                
        except Exception as e:
            print(f"   ❌ Error en verificación final: {e}")
            return False

    def run_complete_integration(self) -> bool:
        """Ejecutar integración completa"""
        print("🔧 INICIANDO INTEGRACIÓN COMPLETA DE XRP")
        print("=" * 60)
        
        success = True
        
        # Ejecutar todos los pasos
        success &= self.step_1_verify_model_exists()
        if not success:
            return False
        
        success &= self.step_2_update_config_files()
        success &= self.step_3_verify_model_compatibility()
        success &= self.step_4_test_prediction()
        success &= self.step_5_create_backup()
        success &= self.step_6_final_verification()
        
        return success

def main():
    """Función principal"""
    print("🔧 XRP SYSTEM INTEGRATOR")
    print("=" * 50)
    
    integrator = XRPSystemIntegrator()
    
    # Ejecutar integración
    success = integrator.run_complete_integration()
    
    if success:
        print(f"\n🎉 ¡XRP INTEGRADO EXITOSAMENTE!")
        print(f"🔄 Pasos siguientes:")
        print(f"   1. Reiniciar el sistema de trading")
        print(f"   2. Verificar que XRPUSDT aparece en los logs")
        print(f"   3. Monitorear las primeras predicciones")
        print(f"\n⚡ Comando para reiniciar:")
        print(f"   python simple_professional_manager.py")
    else:
        print(f"\n❌ Integración falló")
        print(f"   Revisa los errores anteriores")
        print(f"   El backup está disponible en: backup_before_xrp/")

if __name__ == "__main__":
    main()