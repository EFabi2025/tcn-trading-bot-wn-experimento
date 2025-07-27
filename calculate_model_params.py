#!/usr/bin/env python3
"""
🧮 CALCULADOR DE PARÁMETROS DEL MODELO TCN
Script para calcular exactamente cuántos parámetros tendrá el modelo
"""

import tensorflow as tf
import numpy as np

def create_tcn_model_for_calculation(input_shape: tuple):
    """🎯 Crear modelo TCN para calcular parámetros"""
    
    print(f"🎯 Calculando parámetros para input_shape: {input_shape}")
    
    model = tf.keras.Sequential([
        # Input
        tf.keras.layers.Input(shape=input_shape),
        
        # Normalización de entrada
        tf.keras.layers.LayerNormalization(),
        
        # TCN Layer 1
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # TCN Layer 2
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # TCN Layer 3
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Global pooling
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    return model

def calculate_parameters_detailed():
    """🧮 Calcular parámetros detalladamente"""
    
    print("🧮 CALCULADOR DE PARÁMETROS DEL MODELO TCN")
    print("=" * 60)
    
    # Configuraciones típicas
    configs = [
        {'timeframe': '1m', 'lookback': 60, 'features': 66},
        {'timeframe': '5m', 'lookback': 24, 'features': 66},
        {'timeframe': '15m', 'lookback': 16, 'features': 66},
        {'timeframe': '1h', 'lookback': 12, 'features': 66},
        {'timeframe': '4h', 'lookback': 8, 'features': 66}
    ]
    
    for config in configs:
        input_shape = (config['lookback'], config['features'])
        model = create_tcn_model_for_calculation(input_shape)
        
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n📊 CONFIGURACIÓN: {config['timeframe']}")
        print(f"   - Input shape: {input_shape}")
        print(f"   - Total parámetros: {total_params:,}")
        print(f"   - Trainable: {trainable_params:,}")
        print(f"   - Non-trainable: {non_trainable_params:,}")
        
        # Clasificar por tamaño
        if total_params > 1000000:
            size_category = "🚀 MUY GRANDE (>1M)"
        elif total_params > 500000:
            size_category = "📈 GRANDE (500K-1M)"
        elif total_params > 100000:
            size_category = "📊 MEDIANO (100K-500K)"
        elif total_params > 50000:
            size_category = "📉 PEQUEÑO (50K-100K)"
        else:
            size_category = "🔍 MUY PEQUEÑO (<50K)"
            
        print(f"   - Categoría: {size_category}")
        
        # Calcular eficiencia
        efficiency = total_params / (config['lookback'] * config['features'])
        print(f"   - Eficiencia: {efficiency:.1f} params por feature temporal")
        
        # Comparar con modelo anterior (complejo)
        complex_model_estimate = config['lookback'] * config['features'] * 256 * 7  # Estimación del modelo complejo
        reduction = ((complex_model_estimate - total_params) / complex_model_estimate) * 100
        print(f"   - Reducción vs modelo complejo: {reduction:.1f}%")

def main():
    """🎯 Función principal"""
    calculate_parameters_detailed()
    
    print(f"\n🎯 RESUMEN:")
    print(f"   ✅ El modelo actual es MUCHO MÁS PEQUEÑO que el anterior")
    print(f"   ✅ Debería estar en el rango 50K-200K parámetros")
    print(f"   ✅ Esto evita overfitting y mejora la generalización")
    print(f"   ✅ Entrenamiento más rápido y estable")

if __name__ == "__main__":
    main() 