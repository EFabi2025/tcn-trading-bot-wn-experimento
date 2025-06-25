#!/usr/bin/env python3
"""
🚀 INICIO DEL BOT DE TRADING CON MOTOR HÍBRIDO
Script principal para ejecutar el bot con features híbridas optimizadas
"""

import asyncio
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv

# Cargar configuración
load_dotenv()

# Importar el manager actualizado con motor híbrido
from simple_professional_manager import TradingManager, TradingManagerStatus

class HybridTradingBot:
    """🤖 Bot de Trading con Motor Híbrido"""
    
    def __init__(self):
        self.manager = None
        self.running = False
        
    async def start(self):
        """🚀 Iniciar el bot de trading"""
        print("🤖 INICIANDO BOT DE TRADING CON MOTOR HÍBRIDO")
        print("=" * 60)
        print(f"🕐 Hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # 1. Crear y configurar el manager
            print("\n⚙️ Configurando Trading Manager...")
            self.manager = TradingManager()
            
            # 2. Inicializar todos los componentes
            print("🔧 Inicializando componentes del sistema...")
            await self.manager.initialize()
            
            if self.manager.status != TradingManagerStatus.RUNNING:
                raise Exception("El Trading Manager no pudo inicializarse correctamente")
            
            print("\n🎉 ¡BOT INICIADO EXITOSAMENTE!")
            print("📊 Características del bot:")
            print("   ✅ Motor de Features Híbridas activado")
            print("   ✅ Fallback automático al motor original")
            print("   ✅ Umbrales de confianza optimizados")
            print("   ✅ Gestión avanzada de riesgo")
            print("   ✅ Notificaciones Discord inteligentes")
            print("   ✅ Filtro de régimen de mercado")
            print("   ✅ Portfolio manager profesional")
            
            # 3. Configurar manejadores de señales
            self._setup_signal_handlers()
            
            # 4. Ejecutar bucle principal
            self.running = True
            print(f"\n🔄 Iniciando bucle principal de trading...")
            print("💡 Presiona Ctrl+C para detener el bot de forma segura")
            print("-" * 60)
            
            await self.manager.run()
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción detectada por el usuario...")
            await self.stop()
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            await self.stop()
            sys.exit(1)
    
    async def stop(self):
        """🛑 Detener el bot de forma segura"""
        if not self.running:
            return
            
        print("\n🛑 DETENIENDO BOT DE TRADING...")
        self.running = False
        
        if self.manager:
            try:
                await self.manager.shutdown()
                print("✅ Trading Manager detenido correctamente")
            except Exception as e:
                print(f"⚠️ Error durante el apagado: {e}")
        
        print("🏁 Bot detenido exitosamente")
        print(f"🕐 Hora de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _setup_signal_handlers(self):
        """⚙️ Configurar manejadores de señales del sistema"""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Señal {signum} recibida, iniciando apagado seguro...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """🎯 Función principal"""
    bot = HybridTradingBot()
    
    try:
        await bot.start()
    except Exception as e:
        print(f"❌ Error en función principal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 BINANCE TCN TRADING BOT - VERSIÓN HÍBRIDA")
    print("=" * 60)
    print("🔧 Motor de Features: HÍBRIDO OPTIMIZADO")
    print("🎯 Umbrales: CONFIGURABLES VÍA .env")
    print("🛡️ Seguridad: FALLBACK AUTOMÁTICO")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"\n💥 Error fatal: {e}")
        sys.exit(1) 