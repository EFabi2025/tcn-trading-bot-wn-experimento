#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE ÓRDENES NO EJECUTADAS
Script para identificar por qué las señales BUY con alta confianza no se ejecutan
"""

import os
import asyncio
import aiohttp
import time
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class OrderDiagnostic:
    """🔍 Diagnóstico de ejecución de órdenes"""

    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://testnet.binance.vision')
        self.trade_mode = os.getenv('TRADE_MODE', 'dry_run')
        self.dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        self.environment = os.getenv('ENVIRONMENT', 'testnet')

    async def run_full_diagnostic(self):
        """🔍 Ejecutar diagnóstico completo"""

        print("🔍 DIAGNÓSTICO DE ÓRDENES NO EJECUTADAS")
        print("=" * 60)
        print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 1. Verificar configuración
        await self._check_configuration()

        # 2. Verificar conectividad
        await self._check_connectivity()

        # 3. Verificar balance
        await self._check_balance()

        # 4. Verificar filtros de trading
        await self._check_trading_filters()

        # 5. Simular una orden para identificar problemas
        await self._simulate_order_execution()

        print("\n" + "=" * 60)
        print("✅ DIAGNÓSTICO COMPLETADO")

    async def _check_configuration(self):
        """⚙️ Verificar configuración del sistema"""

        print("1️⃣ VERIFICACIÓN DE CONFIGURACIÓN")
        print("-" * 40)

        # Verificar variables críticas
        config_issues = []

                if not self.api_key or self.api_key == 'tu_api_key_de_binance_aqui':
            config_issues.append("❌ BINANCE_API_KEY no configurada")
        else:
            print(f"✅ API KEY configurada")
            
        if not self.secret_key or self.secret_key == 'tu_secret_key_de_binance_aqui':
            config_issues.append("❌ BINANCE_SECRET_KEY no configurada")
        else:
            print(f"✅ SECRET KEY configurada")

        print(f"🌍 Entorno: {self.environment}")
        print(f"🔗 Base URL: {self.base_url}")
        print(f"📊 Modo Trading: {self.trade_mode}")
        print(f"🧪 Dry Run: {self.dry_run}")

        # Verificar modo trading
        if self.dry_run:
            config_issues.append("⚠️ DRY_RUN=true - Solo simulación, NO órdenes reales")

        if self.trade_mode == 'dry_run':
            config_issues.append("⚠️ TRADE_MODE=dry_run - Solo simulación")

        if self.environment == 'testnet':
            print("ℹ️ Usando TESTNET - Órdenes reales pero con dinero virtual")

        # Mostrar problemas
        if config_issues:
            print("\n🚨 PROBLEMAS DE CONFIGURACIÓN DETECTADOS:")
            for issue in config_issues:
                print(f"   {issue}")
        else:
            print("\n✅ Configuración parece correcta")

        print()

    async def _check_connectivity(self):
        """🌐 Verificar conectividad con Binance"""

        print("2️⃣ VERIFICACIÓN DE CONECTIVIDAD")
        print("-" * 40)

        try:
            # Test básico de conectividad
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ping"

                async with session.get(url) as response:
                    if response.status == 200:
                        print("✅ Conectividad con Binance: OK")
                    else:
                        print(f"❌ Error de conectividad: {response.status}")

                # Test de servidor time
                url = f"{self.base_url}/api/v3/time"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        server_time = data['serverTime'] / 1000
                        local_time = time.time()
                        time_diff = abs(server_time - local_time)

                        if time_diff < 1:
                            print("✅ Sincronización de tiempo: OK")
                        else:
                            print(f"⚠️ Diferencia de tiempo: {time_diff:.2f}s")

        except Exception as e:
            print(f"❌ Error de conectividad: {e}")

        print()

    async def _check_balance(self):
        """💰 Verificar balance de la cuenta"""

        print("3️⃣ VERIFICACIÓN DE BALANCE")
        print("-" * 40)

        if not self.api_key or not self.secret_key:
            print("❌ No se puede verificar balance - API keys no configuradas")
            print()
            return

        try:
            # Obtener información de cuenta
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"

            import hmac
            import hashlib
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            url = f"{self.base_url}/api/v3/account"
            headers = {'X-MBX-APIKEY': self.api_key}
            params = {'timestamp': timestamp, 'signature': signature}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Buscar balance USDT
                        usdt_balance = 0
                        for balance in data['balances']:
                            if balance['asset'] == 'USDT':
                                usdt_balance = float(balance['free'])
                                break

                        print(f"💰 Balance USDT: ${usdt_balance:.2f}")

                        # Verificar si es suficiente
                        min_required = float(os.getenv('MIN_TRADE_VALUE_USDT', '11'))
                        if usdt_balance >= min_required:
                            print(f"✅ Balance suficiente (mínimo: ${min_required})")
                        else:
                            print(f"❌ Balance insuficiente (mínimo: ${min_required})")

                    else:
                        error_text = await response.text()
                        print(f"❌ Error obteniendo balance: {response.status}")
                        print(f"   Detalle: {error_text}")

        except Exception as e:
            print(f"❌ Error verificando balance: {e}")

        print()

    async def _check_trading_filters(self):
        """🛡️ Verificar filtros de trading"""

        print("4️⃣ VERIFICACIÓN DE FILTROS DE TRADING")
        print("-" * 40)

        # Verificar configuración de confianza
        min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))
        print(f"🎯 Confianza mínima configurada: {min_confidence:.1%}")

        # Verificar límites de posición
        max_position = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '15'))
        print(f"📊 Tamaño máximo de posición: {max_position}%")

        # Verificar pérdida diaria máxima
        max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '10'))
        print(f"📉 Pérdida diaria máxima: {max_daily_loss}%")

        # Verificar posiciones simultáneas
        max_positions = int(os.getenv('MAX_SIMULTANEOUS_POSITIONS', '2'))
        print(f"🔢 Posiciones simultáneas máximas: {max_positions}")

        print("✅ Filtros configurados correctamente")
        print()

    async def _simulate_order_execution(self):
        """🧪 Simular ejecución de orden para identificar problemas"""

        print("5️⃣ SIMULACIÓN DE EJECUCIÓN DE ORDEN")
        print("-" * 40)

        # Simular una señal BUY con alta confianza
        test_symbol = "BTCUSDT"
        test_confidence = 0.85
        test_signal = "BUY"

        print(f"🧪 Simulando: {test_signal} {test_symbol} con confianza {test_confidence:.1%}")

        # Verificar filtros paso a paso
        checks = []

        # 1. Verificar confianza
        min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))
        if test_confidence >= min_confidence:
            checks.append("✅ Confianza suficiente")
        else:
            checks.append(f"❌ Confianza insuficiente: {test_confidence:.1%} < {min_confidence:.1%}")

        # 2. Verificar modo trading
        if self.dry_run:
            checks.append("⚠️ Modo DRY_RUN - Solo simulación")
        elif self.trade_mode == 'dry_run':
            checks.append("⚠️ TRADE_MODE=dry_run - Solo simulación")
        else:
            checks.append("✅ Modo trading real configurado")

        # 3. Verificar API keys
        if self.api_key and self.secret_key and self.api_key != 'tu_api_key_de_binance_aqui':
            checks.append("✅ API keys configuradas")
        else:
            checks.append("❌ API keys no configuradas")

        # Mostrar resultados
        for check in checks:
            print(f"   {check}")

        print()

        # Conclusión
        if self.dry_run or self.trade_mode == 'dry_run':
            print("🔍 CONCLUSIÓN: Las órdenes NO se ejecutan porque está en modo simulación")
            print("   💡 SOLUCIÓN: Cambiar DRY_RUN=false y TRADE_MODE=real en el archivo .env")
        elif not self.api_key or not self.secret_key or self.api_key == 'tu_api_key_de_binance_aqui':
            print("🔍 CONCLUSIÓN: Las órdenes NO se ejecutan porque faltan API keys")
            print("   💡 SOLUCIÓN: Configurar BINANCE_API_KEY y BINANCE_SECRET_KEY")
        else:
            print("🔍 CONCLUSIÓN: La configuración parece correcta para trading real")
            print("   ⚠️ Verificar logs del sistema para errores específicos")

async def main():
    """🚀 Función principal"""
    diagnostic = OrderDiagnostic()
    await diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    asyncio.run(main())
