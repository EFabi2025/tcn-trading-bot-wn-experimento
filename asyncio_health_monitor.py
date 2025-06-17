#!/usr/bin/env python3
"""
🔍 ASYNCIO HEALTH MONITOR
Monitor de salud para el sistema de trading asyncio
Detecta y corrige problemas de permanencia del ciclo
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asyncio_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AsyncioHealthMonitor:
    """🔍 Monitor de salud para sistema asyncio"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.health_checks = []
        self.error_history = []
        self.performance_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'task_count': [],
            'loop_latency': []
        }
        self.is_monitoring = False
        
    async def start_monitoring(self, check_interval: int = 30):
        """🔄 Iniciar monitoreo de salud"""
        logger.info("🔍 Iniciando monitor de salud asyncio")
        self.is_monitoring = True
        
        # Tareas de monitoreo en paralelo
        monitoring_tasks = [
            self._monitor_loop_health(),
            self._monitor_task_count(),
            self._monitor_system_resources(),
            self._monitor_event_loop_latency(),
            self._cleanup_old_data()
        ]
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except Exception as e:
            logger.error(f"❌ Error en monitoreo: {e}")
            self.is_monitoring = False
    
    async def _monitor_loop_health(self):
        """🔄 Monitorear salud del event loop"""
        while self.is_monitoring:
            try:
                loop = asyncio.get_running_loop()
                
                # Verificar si el loop está funcionando
                start_time = time.time()
                await asyncio.sleep(0.1)
                actual_delay = time.time() - start_time
                
                health_status = {
                    'timestamp': datetime.now(),
                    'loop_responsive': actual_delay < 0.2,  # Menos de 200ms de latencia
                    'actual_delay': actual_delay,
                    'task_count': len(asyncio.all_tasks()),
                    'loop_running': loop.is_running(),
                    'loop_closed': loop.is_closed()
                }
                
                self.health_checks.append(health_status)
                
                # Mantener solo últimos 100 checks
                if len(self.health_checks) > 100:
                    self.health_checks = self.health_checks[-100:]
                
                # Advertir si hay problemas
                if not health_status['loop_responsive']:
                    logger.warning(f"⚠️ Event loop lento: {actual_delay:.3f}s")
                
                if health_status['task_count'] > 50:
                    logger.warning(f"⚠️ Muchas tareas activas: {health_status['task_count']}")
                
                await asyncio.sleep(30)  # Check cada 30 segundos
                
            except Exception as e:
                logger.error(f"❌ Error monitoreando loop: {e}")
                self.error_history.append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'type': 'loop_monitoring'
                })
                await asyncio.sleep(10)
    
    async def _monitor_task_count(self):
        """📊 Monitorear cantidad de tareas"""
        while self.is_monitoring:
            try:
                all_tasks = asyncio.all_tasks()
                task_count = len(all_tasks)
                
                self.performance_metrics['task_count'].append({
                    'timestamp': datetime.now(),
                    'count': task_count
                })
                
                # Detectar tareas colgadas
                running_tasks = [t for t in all_tasks if not t.done()]
                if len(running_tasks) > 20:
                    logger.warning(f"⚠️ {len(running_tasks)} tareas corriendo simultáneamente")
                    
                    # Log de tareas activas
                    for i, task in enumerate(running_tasks[:5]):  # Solo primeras 5
                        logger.info(f"   📋 Tarea {i+1}: {task.get_name()}")
                
                await asyncio.sleep(60)  # Check cada minuto
                
            except Exception as e:
                logger.error(f"❌ Error monitoreando tareas: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_system_resources(self):
        """💻 Monitorear recursos del sistema"""
        while self.is_monitoring:
            try:
                # CPU y memoria
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                self.performance_metrics['cpu_usage'].append({
                    'timestamp': datetime.now(),
                    'value': cpu_percent
                })
                
                self.performance_metrics['memory_usage'].append({
                    'timestamp': datetime.now(),
                    'value': memory_percent
                })
                
                # Advertencias de recursos
                if cpu_percent > 80:
                    logger.warning(f"⚠️ CPU alta: {cpu_percent:.1f}%")
                
                if memory_percent > 85:
                    logger.warning(f"⚠️ Memoria alta: {memory_percent:.1f}%")
                
                await asyncio.sleep(60)  # Check cada minuto
                
            except Exception as e:
                logger.error(f"❌ Error monitoreando recursos: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_event_loop_latency(self):
        """⏱️ Monitorear latencia del event loop"""
        while self.is_monitoring:
            try:
                # Medir latencia del event loop
                measurements = []
                
                for _ in range(5):  # 5 mediciones
                    start = time.perf_counter()
                    await asyncio.sleep(0)  # Yield to event loop
                    latency = time.perf_counter() - start
                    measurements.append(latency)
                
                avg_latency = sum(measurements) / len(measurements)
                max_latency = max(measurements)
                
                self.performance_metrics['loop_latency'].append({
                    'timestamp': datetime.now(),
                    'avg_latency': avg_latency,
                    'max_latency': max_latency
                })
                
                # Advertir si latencia es muy alta
                if avg_latency > 0.001:  # Más de 1ms promedio
                    logger.warning(f"⚠️ Latencia alta del event loop: {avg_latency*1000:.2f}ms")
                
                await asyncio.sleep(120)  # Check cada 2 minutos
                
            except Exception as e:
                logger.error(f"❌ Error midiendo latencia: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """🧹 Limpiar datos antiguos"""
        while self.is_monitoring:
            try:
                cutoff_time = datetime.now() - timedelta(hours=2)
                
                # Limpiar métricas antiguas
                for metric_name, data_list in self.performance_metrics.items():
                    self.performance_metrics[metric_name] = [
                        item for item in data_list 
                        if item['timestamp'] > cutoff_time
                    ]
                
                # Limpiar errores antiguos
                self.error_history = [
                    error for error in self.error_history 
                    if error['timestamp'] > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Limpiar cada hora
                
            except Exception as e:
                logger.error(f"❌ Error en limpieza: {e}")
                await asyncio.sleep(600)
    
    def get_health_report(self) -> Dict:
        """📊 Obtener reporte de salud"""
        now = datetime.now()
        uptime = now - self.start_time
        
        # Calcular promedios recientes
        recent_checks = [
            check for check in self.health_checks 
            if (now - check['timestamp']).total_seconds() < 300  # Últimos 5 minutos
        ]
        
        recent_cpu = [
            metric['value'] for metric in self.performance_metrics['cpu_usage']
            if (now - metric['timestamp']).total_seconds() < 300
        ]
        
        recent_memory = [
            metric['value'] for metric in self.performance_metrics['memory_usage']
            if (now - metric['timestamp']).total_seconds() < 300
        ]
        
        recent_latency = [
            metric['avg_latency'] for metric in self.performance_metrics['loop_latency']
            if (now - metric['timestamp']).total_seconds() < 300
        ]
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime).split('.')[0],
            'is_healthy': len(recent_checks) > 0 and all(check['loop_responsive'] for check in recent_checks),
            'total_health_checks': len(self.health_checks),
            'recent_health_checks': len(recent_checks),
            'recent_errors': len([e for e in self.error_history if (now - e['timestamp']).total_seconds() < 3600]),
            'avg_cpu_recent': sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0,
            'avg_memory_recent': sum(recent_memory) / len(recent_memory) if recent_memory else 0,
            'avg_latency_recent_ms': (sum(recent_latency) / len(recent_latency) * 1000) if recent_latency else 0,
            'current_task_count': len(asyncio.all_tasks()) if asyncio.current_task() else 0,
            'last_check': recent_checks[-1]['timestamp'].isoformat() if recent_checks else None
        }
    
    def stop_monitoring(self):
        """🛑 Detener monitoreo"""
        logger.info("🛑 Deteniendo monitor de salud")
        self.is_monitoring = False

async def main():
    """🎯 Función principal para testing"""
    monitor = AsyncioHealthMonitor()
    
    try:
        # Ejecutar monitoreo por 5 minutos
        await asyncio.wait_for(monitor.start_monitoring(), timeout=300)
    except asyncio.TimeoutError:
        logger.info("⏰ Timeout de 5 minutos alcanzado")
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupción manual")
    finally:
        monitor.stop_monitoring()
        
        # Mostrar reporte final
        report = monitor.get_health_report()
        logger.info("📊 REPORTE FINAL DE SALUD:")
        for key, value in report.items():
            logger.info(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main()) 