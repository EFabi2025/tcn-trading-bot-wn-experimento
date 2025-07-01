#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO: PROBLEMA DEL RÉGIMEN DE MERCADO
Analizar por qué el bot detecta NEUTRAL cuando debería ser BEARISH
"""

import asyncio
import aiohttp
import numpy as np
from datetime import datetime

async def diagnose_market_regime():
    """🔍 Diagnosticar el cálculo del régimen de mercado"""
    print("🔍 DIAGNÓSTICO DEL RÉGIMEN DE MERCADO")
    print("=" * 60)

    try:
        # 1. Obtener datos actuales como lo hace el bot
        print("1️⃣ Obteniendo datos de mercado...")
        market_data = {}

        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            try:
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': '1h',
                    'limit': 100
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            klines = await response.json()
                            market_data[symbol] = [float(k[4]) for k in klines]  # Precios de cierre
                            print(f"   ✅ {symbol}: {len(market_data[symbol])} velas obtenidas")
            except Exception as e:
                print(f"   ❌ Error obteniendo {symbol}: {e}")

        # 2. Analizar tendencia BTC (problema principal)
        print("\n2️⃣ Analizando tendencia BTC...")
        btc_prices = market_data.get('BTCUSDT', [])

        if len(btc_prices) >= 50:
            current_btc = btc_prices[-1]
            short_avg = sum(btc_prices[-10:]) / 10   # 10h
            medium_avg = sum(btc_prices[-24:]) / 24  # 24h
            long_avg = sum(btc_prices[-50:]) / 50    # 50h

            print(f"   💰 Precio actual BTC: ${current_btc:,.2f}")
            print(f"   📊 Promedio 10h: ${short_avg:,.2f}")
            print(f"   📊 Promedio 24h: ${medium_avg:,.2f}")
            print(f"   📊 Promedio 50h: ${long_avg:,.2f}")

            # Calcular diferencias porcentuales
            short_diff = ((current_btc - short_avg) / short_avg) * 100
            medium_diff = ((current_btc - medium_avg) / medium_avg) * 100
            long_diff = ((current_btc - long_avg) / long_avg) * 100

            print(f"   📈 Vs promedio 10h: {short_diff:+.2f}%")
            print(f"   📈 Vs promedio 24h: {medium_diff:+.2f}%")
            print(f"   📈 Vs promedio 50h: {long_diff:+.2f}%")

            # Cálculo actual del bot (PROBLEMÁTICO)
            trend_score_current = 0
            trend_score_current += 0.4 * ((current_btc - short_avg) / short_avg)   # 40%
            trend_score_current += 0.35 * ((current_btc - medium_avg) / medium_avg) # 35%
            trend_score_current += 0.25 * ((current_btc - long_avg) / long_avg)     # 25%

            # ❌ PROBLEMA: Multiplicar por 10
            trend_score_normalized = max(-1, min(1, trend_score_current * 10))

            print(f"\n   🔍 ANÁLISIS DEL PROBLEMA:")
            print(f"   📊 Trend score raw: {trend_score_current:.6f}")
            print(f"   📊 Trend score x10: {trend_score_current * 10:.6f}")
            print(f"   📊 Trend score final (clamped): {trend_score_normalized:.6f}")

            # Cálculo mejorado (SIN multiplicar por 10)
            trend_score_fixed = max(-1, min(1, trend_score_current))
            print(f"   ✅ Trend score corregido: {trend_score_fixed:.6f}")

        # 3. Calcular volatilidad (otro problema)
        print("\n3️⃣ Analizando factor de miedo/volatilidad...")

        if len(btc_prices) >= 25:
            # Cálculo actual (PROBLEMÁTICO)
            btc_returns = np.diff(btc_prices[-24:]) / btc_prices[-25:-1]
            btc_volatility = np.std(btc_returns)

            # ❌ PROBLEMA: Multiplicar por 100
            fear_factor_current = min(1, btc_volatility * 100)

            print(f"   📊 Volatilidad real: {btc_volatility:.6f}")
            print(f"   📊 Fear factor x100: {btc_volatility * 100:.6f}")
            print(f"   📊 Fear factor final (clamped): {fear_factor_current:.6f}")

            # Cálculo mejorado
            fear_factor_fixed = min(1, btc_volatility * 20)  # Usar 20 en lugar de 100
            print(f"   ✅ Fear factor corregido (x20): {fear_factor_fixed:.6f}")

        # 4. Simular cálculo completo
        print("\n4️⃣ Simulando cálculo completo...")

        # Con valores actuales (problemáticos)
        if len(btc_prices) >= 50:
            # Correlación simplificada (asumimos 0.7)
            correlation_strength = 0.7

            # Altcoin strength simplificado (asumimos -0.1)
            altcoin_strength = -0.1

            # Composite score actual (problemático)
            composite_score_current = (
                0.50 * trend_score_normalized +
                0.20 * (correlation_strength - 0.5) * 2 +
                0.15 * (0.5 - fear_factor_current) * 2 +
                0.15 * altcoin_strength * 2
            )

            # Composite score corregido
            composite_score_fixed = (
                0.50 * trend_score_fixed +
                0.20 * (correlation_strength - 0.5) * 2 +
                0.15 * (0.5 - fear_factor_fixed) * 2 +
                0.15 * altcoin_strength * 2
            )

            print(f"   📊 Composite score ACTUAL: {composite_score_current:.6f}")
            print(f"   ✅ Composite score CORREGIDO: {composite_score_fixed:.6f}")

            # Determinar régimen actual
            if composite_score_current > 0.15:
                regime_current = 'BULLISH'
            elif composite_score_current < -0.15:
                regime_current = 'BEARISH'
            else:
                regime_current = 'NEUTRAL'

            # Determinar régimen corregido
            if composite_score_fixed > 0.10:  # Umbrales más sensibles
                regime_fixed = 'BULLISH'
            elif composite_score_fixed < -0.10:
                regime_fixed = 'BEARISH'
            else:
                regime_fixed = 'NEUTRAL'

            print(f"\n   🔍 RESULTADOS:")
            print(f"   ❌ Régimen ACTUAL: {regime_current}")
            print(f"   ✅ Régimen CORREGIDO: {regime_fixed}")

        # 5. Análisis de pérdidas recientes
        print("\n5️⃣ Análisis de pérdidas recientes...")

        if len(btc_prices) >= 24:
            # Cambio en las últimas 24h
            change_24h = ((btc_prices[-1] - btc_prices[-24]) / btc_prices[-24]) * 100

            # Cambio en las últimas 12h
            change_12h = ((btc_prices[-1] - btc_prices[-12]) / btc_prices[-12]) * 100

            # Cambio en las últimas 6h
            change_6h = ((btc_prices[-1] - btc_prices[-6]) / btc_prices[-6]) * 100

            print(f"   📉 Cambio 24h: {change_24h:+.2f}%")
            print(f"   📉 Cambio 12h: {change_12h:+.2f}%")
            print(f"   📉 Cambio 6h: {change_6h:+.2f}%")

            # Si hay caídas consistentes, debería ser BEARISH
            if change_24h < -2 and change_12h < -1 and change_6h < -0.5:
                print(f"   🚨 PATRÓN BEARISH DETECTADO: Caídas consistentes")
                print(f"   ❌ El algoritmo actual NO lo detecta correctamente")

        print("\n" + "=" * 60)
        print("🎯 CONCLUSIONES:")
        print("1. El factor de multiplicación x10 en trend_score causa saturación")
        print("2. El factor de miedo x100 es excesivo")
        print("3. Los umbrales ±0.15 son muy conservadores")
        print("4. Se requiere corrección urgente del algoritmo")

        return True

    except Exception as e:
        print(f"\n❌ ERROR EN DIAGNÓSTICO: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """🚀 Función principal"""
    print("🔍 DIAGNÓSTICO DEL RÉGIMEN DE MERCADO")
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    success = await diagnose_market_regime()

    if success:
        print("\n✅ DIAGNÓSTICO COMPLETADO")
        print("📋 Revisar las conclusiones para implementar correcciones")
    else:
        print("\n❌ DIAGNÓSTICO FALLÓ")

if __name__ == "__main__":
    asyncio.run(main())
