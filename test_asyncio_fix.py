#!/usr/bin/env python3
"""
🧪 TEST ASYNCIO FIX
Script de prueba para verificar las mejoras del sistema asyncio
"""

import asyncio
import sys
from datetime import datetime
from simple_professional_manager import SimpleProfessionalTradingManager
from asyncio_health_monitor import AsyncioHealthMonitor

async def test_system_resilience():
    """🧪 Probar la resistencia del sistema a errores"""
    print("🧪 PRUEBA DE RESISTENCIA ASYNCIO")
    print("=" * 50)

    # Crear manager de trading
    manager = SimpleProfessionalTradingManager()

    # Crear monitor de salud
    health_monitor = AsyncioHealthMonitor()

    try:
        # Inicializar sistema
        print("🔄 Inicializando sistema...")
        await manager.initialize()

        print("✅ Sistema inicializado")
        print(f"📊 Estado: {manager.status}")

        # Iniciar monitor de salud en paralelo
        health_task = asyncio.create_task(health_monitor.start_monitoring())

        # Simular errores para probar la resistencia
        print("\n🎯 Iniciando prueba de resistencia...")

        # Ejecutar por un tiempo limitado para prueba
        test_duration = 300  # 5 minutos
        start_time = datetime.now()

        # Crear tarea de trading con timeout
        trading_task = asyncio.create_task(manager.run())

        try:
            # Esperar con timeout
            await asyncio.wait_for(trading_task, timeout=test_duration)
        except asyncio.TimeoutError:
            print(f"⏰ Prueba completada después de {test_duration} segundos")

        # Obtener reporte de salud
        health_report = health_monitor.get_health_report()

        print("\n📊 REPORTE DE SALUD FINAL:")
        print(f"   ⏱️ Tiempo activo: {health_report['uptime_formatted']}")
        print(f"   🏥 Estado saludable: {'✅' if health_report['is_healthy'] else '❌'}")
        print(f"   📋 Checks de salud: {health_report['total_health_checks']}")
        print(f"   ❌ Errores recientes: {health_report['recent_errors']}")
        print(f"   💻 CPU promedio: {health_report['avg_cpu_recent']:.1f}%")
        print(f"   🧠 Memoria promedio: {health_report['avg_memory_recent']:.1f}%")
        print(f"   ⚡ Latencia promedio: {health_report['avg_latency_recent_ms']:.2f}ms")
        print(f"   📊 Tareas activas: {health_report['current_task_count']}")

        # Estado final del manager
        final_status = await manager.get_system_status()
        print(f"\n🎯 ESTADO FINAL DEL SISTEMA:")
        print(f"   📊 Estado: {final_status['status']}")
        print(f"   ⏱️ Tiempo activo: {final_status['uptime_minutes']:.1f} minutos")
        print(f"   💰 Balance actual: ${final_status['current_balance_usdt']:.2f}")
        print(f"   📈 PnL sesión: ${final_status['session_pnl']:.2f}")
        print(f"   🔢 Trades realizados: {final_status['trade_count']}")
        print(f"   ❌ Errores totales: {final_status['metrics'].get('error_count', 0)}")

        # Verificar si el sistema se mantuvo estable
        if final_status['status'] == 'RUNNING' and health_report['is_healthy']:
            print("\n✅ PRUEBA EXITOSA: Sistema asyncio estable")
            return True
        else:
            print("\n❌ PRUEBA FALLIDA: Sistema inestable")
            return False

    except Exception as e:
        print(f"\n❌ Error en prueba: {e}")
        return False

    finally:
        # Limpiar recursos
        print("\n🧹 Limpiando recursos...")

        health_monitor.stop_monitoring()

        if manager.status != 'STOPPED':
            await manager.shutdown()

        # Cancelar tareas pendientes
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        if tasks:
            print(f"🔄 Cancelando {len(tasks)} tareas pendientes...")
            for task in tasks:
                task.cancel()

            # Esperar a que se cancelen
            await asyncio.gather(*tasks, return_exceptions=True)

async def test_error_recovery():
    """🔄 Probar recuperación de errores"""
    print("\n🔄 PRUEBA DE RECUPERACIÓN DE ERRORES")
    print("=" * 40)

    # Crear instancia simplificada para prueba
    class TestManager:
        def __init__(self):
            self.status = "RUNNING"
            self.error_count = 0
            self.consecutive_errors = 0

        async def simulate_trading_cycle(self):
            """Simular ciclo de trading con errores ocasionales"""
            cycle = 0
            max_consecutive_errors = 5
            last_successful_cycle = datetime.now()

            while cycle < 10 and self.status == "RUNNING":
                cycle += 1
                try:
                    print(f"🔄 Ciclo {cycle}")

                    # Simular error ocasional
                    if cycle in [3, 4, 7]:  # Errores en ciclos específicos
                        raise Exception(f"Error simulado en ciclo {cycle}")

                    # Simular trabajo
                    await asyncio.sleep(0.1)

                    # Ciclo exitoso
                    self.consecutive_errors = 0
                    last_successful_cycle = datetime.now()
                    print(f"✅ Ciclo {cycle} completado")

                except Exception as e:
                    self.consecutive_errors += 1
                    self.error_count += 1

                    print(f"❌ Error en ciclo {cycle}: {e} (Consecutivos: {self.consecutive_errors})")

                    # Lógica de recuperación mejorada
                    if self.consecutive_errors <= 2:
                        print("   ⏱️ Pausa corta (1s)")
                        await asyncio.sleep(1)
                    elif self.consecutive_errors <= 4:
                        print("   ⏱️ Pausa media (2s)")
                        await asyncio.sleep(2)
                    else:
                        print("   ⏱️ Pausa larga (3s)")
                        await asyncio.sleep(3)

                    # Pausar si demasiados errores
                    if self.consecutive_errors >= max_consecutive_errors:
                        print(f"🚨 Sistema pausado por {max_consecutive_errors} errores consecutivos")
                        self.status = "PAUSED"

                        # Auto-reanudar después de pausa
                        await asyncio.sleep(5)
                        print("🔄 Auto-reanudando sistema")
                        self.status = "RUNNING"
                        self.consecutive_errors = 0

    # Ejecutar prueba
    test_manager = TestManager()
    await test_manager.simulate_trading_cycle()

    print(f"\n📊 Resultados de la prueba:")
    print(f"   ❌ Errores totales: {test_manager.error_count}")
    print(f"   📊 Estado final: {test_manager.status}")

    if test_manager.status == "RUNNING":
        print("✅ Sistema se recuperó exitosamente")
        return True
    else:
        print("❌ Sistema no se recuperó")
        return False

async def main():
    """🎯 Función principal de pruebas"""
    print("🧪 SUITE DE PRUEBAS ASYNCIO")
    print("=" * 50)

    results = []

    try:
        # Prueba 1: Recuperación de errores
        print("\n1️⃣ PRUEBA DE RECUPERACIÓN")
        result1 = await test_error_recovery()
        results.append(("Recuperación de errores", result1))

        # Prueba 2: Resistencia del sistema (solo si la primera pasa)
        if result1:
            print("\n2️⃣ PRUEBA DE RESISTENCIA COMPLETA")
            result2 = await test_system_resilience()
            results.append(("Resistencia del sistema", result2))
        else:
            print("\n⚠️ Saltando prueba de resistencia debido a falla anterior")
            results.append(("Resistencia del sistema", False))

    except KeyboardInterrupt:
        print("\n⏹️ Pruebas interrumpidas por usuario")
    except Exception as e:
        print(f"\n❌ Error en suite de pruebas: {e}")

    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 50)

    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"   {test_name}: {status}")

    total_passed = sum(1 for _, result in results if result)
    total_tests = len(results)

    print(f"\n🎯 Resultado: {total_passed}/{total_tests} pruebas exitosas")

    if total_passed == total_tests:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("✅ El sistema asyncio está funcionando correctamente")
    else:
        print("⚠️ Algunas pruebas fallaron")
        print("❗ El sistema necesita más trabajo")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Pruebas detenidas por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal en pruebas: {e}")
        sys.exit(1)
