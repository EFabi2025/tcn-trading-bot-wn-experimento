#!/usr/bin/env python3
"""
🧮 CALCULADOR DE PARÁMETROS DEL MODELO TCN INTERMEDIO
Script para calcular parámetros de la versión balanceada
"""

import tensorflow as tf
import numpy as np

def create_intermediate_tcn_model(input_shape: tuple, timeframe: str):
    """🎯 Crear modelo TCN intermedio para calcular parámetros"""

    print(f"🎯 Calculando parámetros intermedios para {timeframe} - input_shape: {input_shape}")

    # ✅ ARQUITECTURA INTERMEDIA - BALANCEADA
    if timeframe == '1m':
        # Modelo intermedio para alta frecuencia
        filters = [32, 64, 128, 64, 32]
        dilations = [1, 2, 4, 8, 16]
    elif timeframe == '5m':
        # Modelo intermedio para frecuencia media
        filters = [48, 96, 192, 96, 48]
        dilations = [1, 2, 4, 8, 16]
    elif timeframe == '15m':
        # Modelo intermedio
        filters = [40, 80, 160, 80, 40]
        dilations = [1, 2, 4, 8, 16]
    elif timeframe == '1h':
        # Modelo intermedio para timeframes largos
        filters = [32, 64, 128, 64, 32]
        dilations = [1, 2, 4, 8, 16]
    else:  # 4h
        # Modelo intermedio simple
        filters = [24, 48, 96, 48, 24]
        dilations = [1, 2, 4, 8, 16]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LayerNormalization(),
    ])

    # ✅ BLOQUES TCN INTERMEDIOS - BALANCEADOS
    for i, (f, d) in enumerate(zip(filters, dilations)):
        # Bloque TCN intermedio
        conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                filters=f, kernel_size=3, dilation_rate=d,
                padding='causal', activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SpatialDropout1D(0.1 + i * 0.02),  # Dropout moderado
        ])

        model.add(conv_block)

    # ✅ CAPAS FINALES INTERMEDIAS
    model.add(tf.keras.layers.GlobalAveragePooling1D())

    # Dense layers intermedios
    model.add(tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))

    # Capa final
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    return model

def calculate_intermediate_parameters():
    """🧮 Calcular parámetros de la versión intermedia"""

    print("🧮 CALCULADOR DE PARÁMETROS DEL MODELO TCN INTERMEDIO")
    print("=" * 70)

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
        model = create_intermediate_tcn_model(input_shape, config['timeframe'])

        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        print(f"\n📊 CONFIGURACIÓN INTERMEDIA: {config['timeframe']}")
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

        # Comparar con modelos anteriores
        simple_model_params = 23943  # Del modelo simple
        robust_model_params = {
            '1m': 650759, '5m': 1391143, '15m': 984471,
            '1h': 650759, '4h': 390007
        }

        vs_simple = ((total_params - simple_model_params) / simple_model_params) * 100
        vs_robust = ((robust_model_params[config['timeframe']] - total_params) / robust_model_params[config['timeframe']]) * 100

        print(f"   - Incremento vs modelo simple: {vs_simple:+.1f}%")
        print(f"   - Reducción vs modelo robusto: {vs_robust:+.1f}%")

def main():
    """🎯 Función principal"""
    calculate_intermediate_parameters()

    print(f"\n🎯 RESUMEN:")
    print(f"   ✅ El modelo intermedio es BALANCEADO")
    print(f"   ✅ Debería estar en el rango 50K-200K parámetros")
    print(f"   ✅ Balance entre capacidad y eficiencia")
    print(f"   ✅ Ideal para trading algorítmico")

if __name__ == "__main__":
    main()
