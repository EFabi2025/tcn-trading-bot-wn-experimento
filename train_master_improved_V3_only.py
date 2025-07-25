#!/usr/bin/env python3
"""
🎯 ENTRENADOR BTCUSDT ADAPTATIVO
Entrena solo el modelo de BTCUSDT desde cero con técnicas anti-sesgo
"""

import asyncio
from tcn_trainer_v3_improved import ImprovedTCNTrainer

async def main():
    """🚀 Entrenar solo BTCUSDT desde cero"""

    print("🎯 ENTRENAMIENTO ADAPTATIVO - BTCUSDT DESDE CERO")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = ImprovedTCNTrainer()

        # Entrenar solo ETHUSDT
        print("🚀 Iniciando entrenamiento de BTCUSDT desde cero...")
        print("📊 Usando mismo proceso exitoso que BTCUSDT")
        print("⏱️ Tiempo estimado: ~1.5 horas")
        print("💾 Guardará: modelo + scaler + features + checkpoints + checkpoints")

        success = await trainer.train_improved_model("BNBUSDT")

        if success:
            print(f"\n✅ BTCUSDT entrenado exitosamente desde cero")
            print(f"🎯 Archivos guardados en: models/adaptive_v3_improved_btcusdt/")
            print(f"📁 Incluye: best_model.h5, scaler.pkl, feature_columns.pkl")
        else:
            print(f"\n❌ Error entrenando BTCUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
