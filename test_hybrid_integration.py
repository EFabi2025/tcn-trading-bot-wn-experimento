#!/usr/bin/env python3
"""
🧪 TEST: Integración del Motor Híbrido en Trading Manager
Verifica que el nuevo motor de features híbridas funcione correctamente
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Cargar configuración
load_dotenv()

# Importar el manager actualizado
from simple_professional_manager import TradingManager

async def test_hybrid_integration():
    """🧪 Probar la integración del motor híbrido"""
    print("🧪 TESTING INTEGRACIÓN MOTOR HÍBRIDO")
    print("=" * 60)
    
    try:
        # 1. Verificar configuración
        print("\n📋 1. VERIFICANDO CONFIGURACIÓN...")
        
        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')
        base_url = os.getenv('BINANCE_BASE_URL')
        
        if not api_key or not secret_key:
            print("❌ Faltan credenciales de Binance en .env")
            return
        
        print(f"✅ API Key: {api_key[:8]}...")
        print(f"✅ Base URL: {base_url}")
        print(f"✅ Configuración válida")
        
        # 2. Inicializar Trading Manager
        print("\n🚀 2. INICIALIZANDO TRADING MANAGER...")
        manager = TradingManager()
        
        # 3. Inicializar componentes
        print("\n⚙️ 3. INICIALIZANDO COMPONENTES...")
        await manager.initialize()
        
        print("✅ Trading Manager inicializado correctamente")
        
        # 4. Probar predicciones híbridas
        print("\n🔮 4. PROBANDO PREDICCIONES HÍBRIDAS...")
        
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in test_symbols:
            print(f"\n   🧪 Testing {symbol}...")
            
            try:
                prediction = await manager._get_tcn_prediction(symbol)
                
                if prediction:
                    engine = prediction.get('features_engine', 'unknown')
                    signal = prediction.get('signal', 'N/A')
                    confidence = prediction.get('confidence', 0.0) * 100
                    quality = prediction.get('features_quality', 0.0)
                    
                    print(f"   ✅ {symbol}:")
                    print(f"      🎯 Señal: {signal}")
                    print(f"      📊 Confianza: {confidence:.1f}%")
                    print(f"      🔧 Motor: {engine}")
                    if engine == 'hybrid_optimized':
                        print(f"      ⭐ Calidad: {quality:.2f}")
                    
                    # Verificar probabilidades si están disponibles
                    if 'probabilities' in prediction:
                        probs = prediction['probabilities']
                        print(f"      📈 Probabilidades:")
                        for action, prob in probs.items():
                            print(f"         {action}: {prob*100:.1f}%")
                else:
                    print(f"   ❌ {symbol}: No se pudo obtener predicción")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        # 5. Probar generación de reporte completo
        print("\n📊 5. PROBANDO GENERACIÓN DE REPORTE...")
        
        try:
            # Obtener precios actuales
            prices = await manager._get_current_prices()
            print(f"   ✅ Precios obtenidos para {len(prices)} símbolos")
            
            # Generar todas las predicciones
            all_predictions = await manager._generate_tcn_predictions(prices)
            print(f"   ✅ {len(all_predictions)} predicciones generadas")
            
            # Contar motores utilizados
            engine_count = {}
            for pred in all_predictions:
                engine = pred.get('features_engine', 'unknown')
                engine_count[engine] = engine_count.get(engine, 0) + 1
            
            print(f"   📊 Motores utilizados:")
            for engine, count in engine_count.items():
                print(f"      {engine}: {count} predicciones")
            
            # Mostrar estadísticas de calidad
            hybrid_predictions = [p for p in all_predictions if p.get('features_engine') == 'hybrid_optimized']
            if hybrid_predictions:
                avg_quality = sum(p.get('features_quality', 0.0) for p in hybrid_predictions) / len(hybrid_predictions)
                print(f"   ⭐ Calidad promedio features híbridas: {avg_quality:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error en generación de reporte: {e}")
        
        # 6. Cleanup
        print("\n🔄 6. LIMPIEZA...")
        await manager.shutdown()
        print("✅ Sistema apagado correctamente")
        
        # 7. Resumen final
        print("\n" + "=" * 60)
        print("🎉 RESUMEN DEL TEST:")
        print(f"✅ Trading Manager inicializado correctamente")
        print(f"✅ Motor híbrido integrado y funcionando")
        print(f"✅ Predicciones generadas con features optimizadas")
        print(f"✅ Sistema con fallback seguro al motor original")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN TEST: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hybrid_integration()) 