#!/usr/bin/env python3
"""
🎯 ENTRENADOR XRPUSDT DEFINITIVO
Entrena solo el modelo de XRPUSDT con técnicas anti-sesgo
"""

import asyncio
from tcn_definitivo_trainer import DefinitiveTCNTrainer

async def main():
    """🚀 Entrenar solo XRPUSDT"""

    print("🎯 ENTRENAMIENTO DEFINITIVO - XRPUSDT ÚNICAMENTE")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = DefinitiveTCNTrainer()

        # Entrenar solo XRPUSDT
        print("🚀 Iniciando entrenamiento de XRPUSDT...")
        success = await trainer.train_definitive_model("XRPUSDT")

        if success:
            print(f"\n✅ XRPUSDT entrenado exitosamente")
        else:
            print(f"\n❌ Error entrenando XRPUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
