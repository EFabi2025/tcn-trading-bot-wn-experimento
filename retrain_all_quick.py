#!/usr/bin/env python3
"""
REENTRENAMIENTO RÁPIDO DE TODOS LOS MODELOS
Reentrenar BTC, ETH y BNB con sequence_length = 24
"""

import asyncio
from tcn_definitivo_trainer import DefinitiveTCNTrainer

async def main():
    print("🚀 REENTRENAMIENTO RÁPIDO DE TODOS LOS MODELOS")
    print("=" * 60)

    trainer = DefinitiveTCNTrainer()

    # Lista de símbolos a reentrenar
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    for symbol in symbols:
        print(f"\n🎯 REENTRENANDO {symbol}...")
        print("-" * 40)

        try:
            # Entrenar el modelo
            success = await trainer.train_single_symbol(symbol)

            if success:
                print(f"✅ {symbol} reentrenado exitosamente")
            else:
                print(f"❌ Error reentrenando {symbol}")

        except Exception as e:
            print(f"❌ Error en {symbol}: {e}")

    print("\n🎉 REENTRENAMIENTO COMPLETADO")

if __name__ == "__main__":
    asyncio.run(main())
