#!/usr/bin/env python3
"""
🎯 MASTER HYBRID TRAINER - ENTRENADOR MAESTRO V2
Script maestro para entrenar símbolos específicos con híbrido definitivo_v2
Flexible para entrenar cualquier par individual o múltiples pares
"""

import asyncio
import sys
import argparse
from typing import List, Optional
from tcn_hybrid_trainer import TCNHybridTrainer


class MasterHybridTrainer:
    """🎯 Entrenador maestro para modelos híbridos V2"""
    
    def __init__(self):
        self.trainer = TCNHybridTrainer()
        self.available_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
        
    def display_banner(self):
        """🎨 Banner del entrenador maestro"""
        print("=" * 80)
        print("🎯 MASTER HYBRID TRAINER V2")
        print("=" * 80)
        print("🔄 Etiquetado: Definitivo balanceado")
        print("🏗️ Arquitectura: TCN V3 mejorada (piramidal)")
        print("💾 Guardado como: definitivo_v2_5m_[symbol]")
        print("🎯 Compatible con: Predictor definitivo")
        print("=" * 80)
        
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """✅ Validar símbolos disponibles"""
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            if symbol_upper in self.available_pairs:
                valid_symbols.append(symbol_upper)
            else:
                invalid_symbols.append(symbol)
                
        if invalid_symbols:
            print(f"⚠️ Símbolos no soportados: {invalid_symbols}")
            print(f"✅ Símbolos disponibles: {self.available_pairs}")
            
        return valid_symbols
    
    async def train_single_symbol(self, symbol: str) -> bool:
        """🎯 Entrenar un símbolo específico"""
        print(f"\n🚀 ENTRENANDO {symbol} CON HÍBRIDO V2")
        print("=" * 60)
        
        try:
            success = await self.trainer.train_hybrid_model(symbol)
            
            if success:
                print(f"\n✅ {symbol}: ENTRENAMIENTO HÍBRIDO V2 EXITOSO")
                print(f"📁 Modelo guardado en: models/definitivo_v2_{symbol.lower()}/")
                print(f"🎯 Listo para usar con predictor")
            else:
                print(f"\n❌ {symbol}: ERROR EN ENTRENAMIENTO V2")
                
            return success
            
        except Exception as e:
            print(f"\n❌ Error entrenando {symbol}: {e}")
            return False
    
    async def train_multiple_symbols(self, symbols: List[str]) -> dict:
        """🎯 Entrenar múltiples símbolos"""
        print(f"\n🚀 ENTRENANDO MÚLTIPLES SÍMBOLOS: {symbols}")
        print("=" * 60)
        
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 PROGRESO: {i}/{len(symbols)} - {symbol}")
            success = await self.train_single_symbol(symbol)
            results[symbol] = success
            
            if i < len(symbols):
                print(f"\n⏸️ Pausa entre entrenamientos...")
                await asyncio.sleep(2)  # Pequeña pausa entre entrenamientos
                
        return results
    
    def display_results(self, results: dict):
        """📊 Mostrar resumen de resultados"""
        print(f"\n🎯 RESUMEN FINAL DE ENTRENAMIENTO")
        print("=" * 60)
        
        successful = 0
        failed = 0
        
        for symbol, success in results.items():
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"   {symbol}: {status}")
            
            if success:
                successful += 1
                print(f"      📁 models/definitivo_v2_{symbol.lower()}/")
            else:
                failed += 1
                
        print("=" * 60)
        print(f"✅ Exitosos: {successful}")
        print(f"❌ Fallidos: {failed}")
        print(f"📊 Total: {successful + failed}")
        
        if successful > 0:
            print(f"\n🎯 MODELOS HÍBRIDOS V2 LISTOS PARA USAR")
            print(f"🔧 Compatible con predictor definitivo existente")
    
    async def interactive_training(self):
        """🎮 Modo interactivo para entrenar"""
        self.display_banner()
        
        print(f"\n🎯 SÍMBOLOS DISPONIBLES:")
        for i, symbol in enumerate(self.available_pairs, 1):
            print(f"   {i}. {symbol}")
            
        print(f"\n🎯 OPCIONES:")
        print(f"   • Símbolo específico: xrp, btc, eth, bnb")
        print(f"   • Múltiples: btc,eth,xrp")
        print(f"   • Todos: all")
        print(f"   • Salir: exit")
        
        choice = input(f"\n🤔 ¿Qué quieres entrenar? ").strip().lower()
        
        if choice in ['exit', 'q', 'quit']:
            print("👋 ¡Hasta luego!")
            return
            
        if choice == 'all':
            symbols = self.available_pairs
        elif ',' in choice:
            # Múltiples símbolos separados por coma
            symbols = [s.strip().upper() + 'USDT' if not s.strip().upper().endswith('USDT') else s.strip().upper() 
                      for s in choice.split(',')]
        else:
            # Símbolo individual
            if choice.upper().endswith('USDT'):
                symbols = [choice.upper()]
            else:
                symbols = [choice.upper() + 'USDT']
                
        # Validar símbolos
        valid_symbols = self.validate_symbols(symbols)
        
        if not valid_symbols:
            print("❌ No hay símbolos válidos para entrenar")
            return
            
        # Confirmar entrenamiento
        print(f"\n🎯 SÍMBOLOS A ENTRENAR: {valid_symbols}")
        confirm = input(f"¿Proceder con el entrenamiento? (y/n): ").lower().strip()
        
        if confirm != 'y':
            print("❌ Entrenamiento cancelado")
            return
            
        # Entrenar
        if len(valid_symbols) == 1:
            symbol = valid_symbols[0]
            success = await self.train_single_symbol(symbol)
            results = {symbol: success}
        else:
            results = await self.train_multiple_symbols(valid_symbols)
            
        # Mostrar resultados
        self.display_results(results)

async def main():
    """🎯 Función principal del entrenador maestro"""
    
    master_trainer = MasterHybridTrainer()
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        # Modo comando directo
        symbol = sys.argv[1].upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
            
        master_trainer.display_banner()
        
        valid_symbols = master_trainer.validate_symbols([symbol])
        if valid_symbols:
            print(f"\n🎯 ENTRENAMIENTO DIRECTO: {valid_symbols[0]}")
            success = await master_trainer.train_single_symbol(valid_symbols[0])
            results = {valid_symbols[0]: success}
            master_trainer.display_results(results)
        else:
            print(f"❌ Símbolo no válido: {symbol}")
            print(f"✅ Símbolos disponibles: {master_trainer.available_pairs}")
    else:
        # Modo interactivo
        await master_trainer.interactive_training()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n👋 Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        print(f"🔍 Traceback: {traceback.format_exc()}")


# ========================================
# 🎯 EJEMPLOS DE USO:
# ========================================
#
# 1. ENTRENAR XRP (directo):
#    python train_master_hybrid_trainer.py xrp
#
# 2. ENTRENAR BTC (directo):
#    python train_master_hybrid_trainer.py btc
#
# 3. MODO INTERACTIVO:
#    python train_master_hybrid_trainer.py
#    > Opciones: xrp, btc, eth, bnb, all, btc,eth,xrp
#
# 4. ENTRENAR MÚLTIPLES:
#    python train_master_hybrid_trainer.py
#    > btc,xrp,eth
#
# 5. ENTRENAR TODOS:
#    python train_master_hybrid_trainer.py
#    > all
#
# ========================================
