#!/usr/bin/env python3
"""
🚀 Punto de entrada para iniciar el Trading Manager Profesional
"""
import asyncio
import os
from dotenv import load_dotenv
import platform
import sys

# Cargar variables de entorno desde .env
load_dotenv()

# Añadir el directorio raíz al path para importaciones limpias
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Importar el manager principal ya refactorizado
from simple_professional_manager import TradingManager

async def main():
    """Función principal para inicializar y correr el bot"""
    print("+" + "-"*60 + "+")
    print("| 🚀 INICIANDO BOT DE TRADING TCN PROFESIONAL 🚀 |")
    print("+" + "-"*60 + "+")
    print(f"🐍 Versión de Python: {platform.python_version()}")
    print(f"🏗️  Plataforma: {platform.system()} {platform.release()}")
    
    # Verificar que las API Keys están configuradas
    if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_SECRET_KEY'):
        print("\n❌ CRÍTICO: Las variables de entorno BINANCE_API_KEY y BINANCE_SECRET_KEY no están configuradas.")
        print("   Por favor, copie `env_example.txt` a `.env` y añada sus credenciales.")
        print("   El bot no puede iniciar sin credenciales.")
        return

    print("\n✅ Credenciales de Binance encontradas.")

    manager = TradingManager()
    
    try:
        # Inicializar todos los componentes (DB, Risk Manager, Modelos, etc.)
        await manager.initialize()
        
        # Iniciar el loop principal de trading
        await manager.run()
        
    except Exception as e:
        print(f"\n🚨 ERROR CRÍTICO INESPERADO EN EL NIVEL SUPERIOR: {e}")
        if manager and manager.logger:
            manager.logger.critical(f"Top-level unhandled exception: {e}", exc_info=True)
        print("   El sistema se detendrá.")
        
    finally:
        if manager:
            await manager.shutdown()
        print("\n👋 El bot de trading se ha detenido. ¡Hasta la próxima!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Detención manual solicitada por el usuario (Ctrl+C).") 