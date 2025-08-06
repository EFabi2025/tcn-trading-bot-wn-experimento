#!/usr/bin/env python3
"""
🎯 ENTRENADOR BTCUSDT DEFINITIVO
Entrena solo el modelo de BTCUSDT desde cero con técnicas anti-sesgo
"""

import asyncio
from tcn_adaptative_trainer import AdaptiveTCNTrainer

async def main():
    """🚀 Entrenar solo XRPUSDT desde cero"""

    print("🎯 ENTRENAMIENTO DEFINITIVO - XRPUSDT DESDE CERO")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = AdaptiveTCNTrainer()

        # Entrenar solo XRPUSDT
        print("🚀 Iniciando entrenamiento de XRPUSDT desde cero...")
        print("📊 Usando mismo proceso exitoso que ETHUSDT")
        print("⏱️ Tiempo estimado: ~1.5 horas")
        print("💾 Guardará: modelo + scaler + features + checkpoints")

        success = await trainer.train_adaptive_model("XRPUSDT")

        if success:
            print(f"\n✅ XRPUSDT entrenado exitosamente desde cero")
            print(f"🎯 Archivos guardados en: models/adaptativo_xrpusdt/")
            print(f"📁 Incluye: best_model.h5, scaler.pkl, feature_columns.pkl")
        else:
            print(f"\n❌ Error entrenando XRPUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()Cami2018

if __name__ == "__main__":
    asyncio.run(main())
