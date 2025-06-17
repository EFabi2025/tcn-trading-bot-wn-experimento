#!/usr/bin/env python3
"""
🔍 INVESTIGACIÓN ESPECÍFICA - ETHEREUM 10:25 AM
Análisis detallado de por qué no se ejecutó la orden de ETH
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class EthereumProblemInvestigator:
    """🔍 Investigador específico del problema de Ethereum"""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.base_url = os.getenv('BINANCE_BASE_URL', 'https://api.binance.com')
        self.symbol = "ETHUSDT"
        
        print("🔍 INVESTIGANDO PROBLEMA DE ETHEREUM (10:25 AM)")
        print("=" * 60)
    
    def _generate_signature(self, params: str = "") -> str:
        """🔐 Generar firma para API"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _get_current_price(self) -> float:
        """💲 Obtener precio actual de ETH"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {'symbol': self.symbol}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except Exception as e:
            print(f"❌ Error obteniendo precio: {e}")
        return 0.0
    
    async def _get_account_balance(self) -> float:
        """💰 Obtener balance USDT actual"""
        try:
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = self._generate_signature(query_string)
            
            headers = {'X-MBX-APIKEY': self.api_key}
            params = {'timestamp': timestamp, 'signature': signature}
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/api/v3/account"
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for balance in data['balances']:
                            if balance['asset'] == 'USDT':
                                return float(balance['free'])
        except Exception as e:
            print(f"❌ Error obteniendo balance: {e}")
        return 0.0
    
    async def _simulate_tcn_signal(self) -> dict:
        """🤖 Simular señal TCN (usando lógica simplificada)"""
        
        # En un sistema real, esto vendría del modelo TCN
        # Para la investigación, vamos a simular una señal alta
        
        current_price = await self._get_current_price()
        
        # Simular confianza alta como la que reportas
        confidence = 0.82  # 82% - por encima del 70%
        signal = "BUY"
        
        return {
            'symbol': self.symbol,
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'timestamp': datetime.now(),
            'reason': 'Simulación de señal alta confianza'
        }
    
    async def _check_risk_filters(self, signal_data: dict) -> tuple:
        """🛡️ Verificar todos los filtros de riesgo"""
        
        print(f"\n🛡️ VERIFICANDO FILTROS DE RIESGO PARA {signal_data['symbol']}")
        print("-" * 50)
        
        current_price = signal_data['current_price']
        confidence = signal_data['confidence']
        
        # 1. Verificar confianza mínima
        min_confidence = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.70'))
        print(f"1️⃣ Confianza: {confidence:.1%} vs mínimo {min_confidence:.1%}")
        if confidence < min_confidence:
            return False, f"Confianza insuficiente: {confidence:.1%} < {min_confidence:.1%}"
        print("   ✅ PASADO")
        
        # 2. Verificar balance
        balance = await self._get_account_balance()
        min_trade = float(os.getenv('MIN_TRADE_VALUE_USDT', '11'))
        print(f"2️⃣ Balance: ${balance:.2f} vs mínimo ${min_trade}")
        if balance < min_trade:
            return False, f"Balance insuficiente: ${balance:.2f} < ${min_trade}"
        print("   ✅ PASADO")
        
        # 3. Verificar tamaño de posición
        max_position_percent = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '40'))
        position_value = balance * (max_position_percent / 100)
        print(f"3️⃣ Valor máximo posición: ${position_value:.2f} ({max_position_percent}%)")
        if position_value < min_trade:
            return False, f"Posición calculada muy pequeña: ${position_value:.2f}"
        print("   ✅ PASADO")
        
        # 4. Verificar cantidad mínima de Binance
        quantity = position_value / current_price
        print(f"4️⃣ Cantidad ETH calculada: {quantity:.6f}")
        # ETH tiene un mínimo de 0.0001
        if quantity < 0.0001:
            return False, f"Cantidad menor al mínimo Binance: {quantity:.6f} < 0.0001"
        print("   ✅ PASADO")
        
        # 5. Verificar pérdida diaria máxima
        max_daily_loss = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '5'))
        print(f"5️⃣ Límite pérdida diaria: {max_daily_loss}%")
        # Simulamos que no hay pérdidas
        current_daily_loss = 0.0
        if current_daily_loss >= max_daily_loss:
            return False, f"Pérdida diaria excedida: {current_daily_loss:.1f}% >= {max_daily_loss}%"
        print("   ✅ PASADO")
        
        # 6. Verificar posiciones máximas
        max_positions = int(os.getenv('MAX_CONCURRENT_POSITIONS', '2'))
        print(f"6️⃣ Posiciones máximas permitidas: {max_positions}")
        # Necesitaríamos verificar posiciones activas reales
        current_positions = 0  # Simular
        if current_positions >= max_positions:
            return False, f"Posiciones máximas alcanzadas: {current_positions}/{max_positions}"
        print("   ✅ PASADO")
        
        # 7. Verificar modo trading
        dry_run = os.getenv('DRY_RUN', 'true').lower() == 'true'
        trade_mode = os.getenv('TRADE_MODE', 'dry_run')
        print(f"7️⃣ Modo trading: DRY_RUN={dry_run}, TRADE_MODE={trade_mode}")
        
        if dry_run:
            return False, "❌ DRY_RUN=true - Solo simulación activada"
        if trade_mode == 'dry_run':
            return False, "❌ TRADE_MODE=dry_run - Modo simulación"
        print("   ✅ PASADO")
        
        return True, "Todos los filtros pasados correctamente"
    
    async def _simulate_order_execution(self, signal_data: dict) -> dict:
        """🔥 Simular ejecución de orden (SIN ejecutar realmente)"""
        
        print(f"\n🔥 SIMULANDO EJECUCIÓN DE ORDEN")
        print("-" * 50)
        
        current_price = signal_data['current_price']
        balance = await self._get_account_balance()
        max_position_percent = float(os.getenv('MAX_POSITION_SIZE_PERCENT', '40'))
        
        position_value = balance * (max_position_percent / 100)
        quantity = position_value / current_price
        
        print(f"📊 Detalles de la orden que DEBERÍA ejecutarse:")
        print(f"   🎯 Símbolo: {signal_data['symbol']}")
        print(f"   📈 Lado: {signal_data['signal']}")
        print(f"   💰 Precio actual: ${current_price:.4f}")
        print(f"   🔢 Cantidad: {quantity:.6f} ETH")
        print(f"   💵 Valor: ${position_value:.2f}")
        print(f"   🎲 Confianza: {signal_data['confidence']:.1%}")
        
        # Simular parámetros de orden
        timestamp = int(time.time() * 1000)
        order_params = {
            'symbol': signal_data['symbol'],
            'side': signal_data['signal'],
            'type': 'MARKET',
            'quantity': f"{quantity:.6f}",
            'timestamp': timestamp
        }
        
        print(f"\n📝 Parámetros de orden que se enviarían:")
        for key, value in order_params.items():
            print(f"   {key}: {value}")
        
        return {
            'would_execute': True,
            'order_params': order_params,
            'estimated_cost': position_value,
            'quantity': quantity
        }
    
    async def run_investigation(self):
        """🔍 Ejecutar investigación completa"""
        
        print(f"⏰ Investigando problema reportado a las 10:25 AM")
        print(f"🎯 Símbolo: {self.symbol}")
        print(f"📍 Hora actual: {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Generar señal simulada
        print(f"\n1️⃣ GENERANDO SEÑAL SIMULADA (como la de 10:25 AM)")
        signal_data = await self._simulate_tcn_signal()
        
        print(f"🤖 Señal generada:")
        print(f"   📊 {signal_data['symbol']}: {signal_data['signal']}")
        print(f"   🎯 Confianza: {signal_data['confidence']:.1%}")
        print(f"   💲 Precio: ${signal_data['current_price']:.4f}")
        
        # 2. Verificar filtros
        print(f"\n2️⃣ VERIFICANDO FILTROS DE RIESGO")
        can_execute, reason = await self._check_risk_filters(signal_data)
        
        if not can_execute:
            print(f"\n❌ PROBLEMA IDENTIFICADO:")
            print(f"   🚫 Razón: {reason}")
            print(f"   💡 Esta es probablemente la causa por la que no se ejecutó la orden a las 10:25 AM")
            return
        
        print(f"\n✅ TODOS LOS FILTROS PASADOS")
        
        # 3. Simular ejecución
        print(f"\n3️⃣ SIMULANDO EJECUCIÓN DE ORDEN")
        execution_result = await self._simulate_order_execution(signal_data)
        
        if execution_result['would_execute']:
            print(f"\n🎉 CONCLUSIÓN: La orden DEBERÍA haberse ejecutado")
            print(f"   💰 Costo estimado: ${execution_result['estimated_cost']:.2f}")
            print(f"   🔢 Cantidad ETH: {execution_result['quantity']:.6f}")
            print(f"\n🔍 POSIBLE CAUSA DEL PROBLEMA:")
            print(f"   • Error en el código de ejecución")
            print(f"   • Problema de conectividad con Binance")
            print(f"   • Error silencioso no detectado")
            print(f"   • Filtro adicional no documentado")
        else:
            print(f"\n❌ La orden NO se habría ejecutado por problemas técnicos")

async def main():
    """🎯 Función principal"""
    investigator = EthereumProblemInvestigator()
    await investigator.run_investigation()

if __name__ == "__main__":
    asyncio.run(main()) 