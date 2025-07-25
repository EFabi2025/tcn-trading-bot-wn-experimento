#!/usr/bin/env python3
"""
🎯 ENTRENADOR BNBUSDT ADAPTATIVO
Entrena solo el modelo de BNBUSDT desde cero con técnicas anti-sesgo
"""

import asyncio
from tcn_adaptative_trainer_v2_fixed import AdaptiveTCNTrainer

async def main():
    """🚀 Entrenar solo BNBUSDT desde cero"""

    print("🎯 ENTRENAMIENTO ADAPTATIVO - BNBUSDT DESDE CERO")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = AdaptiveTCNTrainer()

        # Entrenar solo BNBUSDT
        print("🚀 Iniciando entrenamiento de BNBUSDT desde cero...")
        print("📊 Usando mismo proceso exitoso que ETHUSDT")
        print("⏱️ Tiempo estimado: ~1.5 horas")
        print("💾 Guardará: modelo + scaler + features + checkpoints")

        success = await trainer.train_adaptive_model("BNBUSDT")

        if success:
            print(f"\n✅ BNBUSDT entrenado exitosamente desde cero")
            print(f"🎯 Archivos guardados en: models/adaptativo_v2_bnbusdt/")
            print(f"📁 Incluye: best_model.h5, scaler.pkl, feature_columns.pkl")
        else:
            print(f"\n❌ Error entrenando BNBUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
