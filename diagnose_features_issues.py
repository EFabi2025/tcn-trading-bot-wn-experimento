#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE FEATURES - Detector de Problemas
==================================================

Script para diagnosticar problemas en el cálculo de features técnicas
que pueden estar causando las bajas confianzas en Windows vs macOS.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports locales
from centralized_features_engine import CentralizedFeaturesEngine
from definitivo_tcn_predictor import BinanceDataProvider

class FeaturesIssueDiagnostics:
    """🔍 Diagnóstico de problemas en features"""
    
    def __init__(self):
        self.engine = CentralizedFeaturesEngine()
        self.issues_found = []
        
    def check_talib_availability(self):
        """Verificar disponibilidad de TA-Lib"""
        print("🔍 Verificando disponibilidad de TA-Lib...")
        
        try:
            import talib
            print("✅ TA-Lib está disponible")
            
            # Test básico
            test_data = np.random.random(50) * 100 + 50000
            rsi = talib.RSI(test_data)
            print(f"   📊 Test RSI: {rsi[-1]:.2f}")
            return True
            
        except ImportError:
            print("❌ TA-Lib NO está disponible - usando implementaciones manuales")
            self.issues_found.append("TA-Lib no disponible - cálculos menos precisos")
            return False
            
        except Exception as e:
            print(f"❌ Error en TA-Lib: {e}")
            self.issues_found.append(f"Error en TA-Lib: {e}")
            return False
    
    async def analyze_real_data_features(self, symbol: str = "BTCUSDT"):
        """Analizar features calculadas desde datos reales"""
        print(f"\n🔍 Analizando features para {symbol}...")
        
        try:
            async with BinanceDataProvider() as provider:
                # Obtener datos reales
                klines = await provider.get_klines(symbol, "1m", 200)
                
                if not klines:
                    print(f"❌ No se pudieron obtener datos para {symbol}")
                    return None
                
                print(f"✅ Obtenidos {len(klines)} klines")
                
                # Calcular features
                features_array = await self.engine.compute_features(symbol, klines, 'tcn_definitivo')
                
                if features_array is None:
                    print(f"❌ Error calculando features")
                    return None
                
                # Análisis estadístico
                self.analyze_features_statistics(features_array, symbol)
                
                # Detectar problemas específicos
                self.detect_feature_issues(features_array, symbol)
                
                return features_array
                
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            self.issues_found.append(f"Error análisis {symbol}: {e}")
            return None
    
    def analyze_features_statistics(self, features_array: np.ndarray, symbol: str):
        """Análisis estadístico detallado de features"""
        print(f"\n📊 ANÁLISIS ESTADÍSTICO - {symbol}")
        print("=" * 50)
        
        # Estadísticas básicas
        print(f"🔢 Shape: {features_array.shape}")
        print(f"🔢 Dtype: {features_array.dtype}")
        
        # Verificar NaN
        nan_count = np.isnan(features_array).sum()
        if nan_count > 0:
            print(f"❌ NaN encontrados: {nan_count}")
            self.issues_found.append(f"{symbol}: {nan_count} valores NaN")
        else:
            print(f"✅ Sin valores NaN")
        
        # Verificar infinitos
        inf_count = np.isinf(features_array).sum()
        if inf_count > 0:
            print(f"❌ Infinitos encontrados: {inf_count}")
            self.issues_found.append(f"{symbol}: {inf_count} valores infinitos")
        else:
            print(f"✅ Sin valores infinitos")
        
        # Estadísticas por feature
        print(f"\n📈 ESTADÍSTICAS POR FEATURE:")
        for i in range(min(10, features_array.shape[1])):  # Primeras 10 features
            col_data = features_array[:, i]
            
            mean_val = np.nanmean(col_data)
            std_val = np.nanstd(col_data)
            min_val = np.nanmin(col_data)
            max_val = np.nanmax(col_data)
            
            print(f"   Feature {i:2d}: μ={mean_val:8.4f}, σ={std_val:8.4f}, range=[{min_val:8.4f}, {max_val:8.4f}]")
            
            # Detectar outliers extremos
            if abs(mean_val) > 1e6:
                self.issues_found.append(f"{symbol}: Feature {i} tiene valores muy grandes (μ={mean_val:.2e})")
            
            if std_val > 1e6:
                self.issues_found.append(f"{symbol}: Feature {i} tiene varianza muy alta (σ={std_val:.2e})")
    
    def detect_feature_issues(self, features_array: np.ndarray, symbol: str):
        """Detectar problemas específicos en features"""
        print(f"\n🔍 DETECCIÓN DE PROBLEMAS ESPECÍFICOS - {symbol}")
        print("=" * 50)
        
        issues_detected = 0
        
        # 1. Features con varianza cero (constantes)
        variances = np.var(features_array, axis=0)
        zero_var_features = np.where(variances < 1e-10)[0]
        
        if len(zero_var_features) > 0:
            print(f"❌ Features constantes detectadas: {len(zero_var_features)}")
            print(f"   Índices: {zero_var_features.tolist()}")
            self.issues_found.append(f"{symbol}: {len(zero_var_features)} features constantes")
            issues_detected += 1
        else:
            print(f"✅ Sin features constantes")
        
        # 2. Features con distribución anormal
        skewness_threshold = 5.0
        for i in range(features_array.shape[1]):
            col_data = features_array[:, i]
            col_data = col_data[~np.isnan(col_data)]  # Remover NaN
            
            if len(col_data) > 3:
                # Calcular skewness manualmente
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                
                if std_val > 0:
                    skewness = np.mean(((col_data - mean_val) / std_val) ** 3)
                    
                    if abs(skewness) > skewness_threshold:
                        print(f"❌ Feature {i}: distribución muy sesgada (skew={skewness:.2f})")
                        self.issues_found.append(f"{symbol}: Feature {i} muy sesgada")
                        issues_detected += 1
        
        # 3. Correlación perfecta entre features
        try:
            correlation_matrix = np.corrcoef(features_array.T)
            
            for i in range(correlation_matrix.shape[0]):
                for j in range(i+1, correlation_matrix.shape[1]):
                    corr = correlation_matrix[i, j]
                    
                    if not np.isnan(corr) and abs(corr) > 0.99:
                        print(f"❌ Features {i} y {j}: correlación perfecta ({corr:.3f})")
                        self.issues_found.append(f"{symbol}: Features {i}-{j} perfectamente correlacionadas")
                        issues_detected += 1
        
        except Exception as e:
            print(f"⚠️ Error calculando correlaciones: {e}")
        
        # 4. Verificar escalas muy diferentes
        scales = []
        for i in range(features_array.shape[1]):
            col_data = features_array[:, i]
            col_data = col_data[~np.isnan(col_data)]
            
            if len(col_data) > 0:
                scale = np.std(col_data)
                scales.append(scale)
        
        if scales:
            max_scale = max(scales)
            min_scale = min([s for s in scales if s > 0])
            
            if max_scale / min_scale > 1000:
                print(f"❌ Escalas muy diferentes: ratio {max_scale/min_scale:.1f}")
                self.issues_found.append(f"{symbol}: Escalas muy diferentes entre features")
                issues_detected += 1
        
        if issues_detected == 0:
            print("✅ No se detectaron problemas específicos")
    
    def compare_normalization_methods(self, features_array: np.ndarray):
        """Comparar diferentes métodos de normalización"""
        print(f"\n🔧 COMPARACIÓN DE MÉTODOS DE NORMALIZACIÓN")
        print("=" * 50)
        
        if features_array is None:
            print("❌ No hay datos para analizar normalización")
            return
        
        # Tomar una muestra de features para análisis
        sample_features = features_array[:, :5]  # Primeras 5 features
        
        methods = {
            'Original': sample_features,
            'StandardScaler': self._standard_scale(sample_features),
            'MinMaxScaler': self._minmax_scale(sample_features),
            'RobustScaler': self._robust_scale(sample_features),
            'QuantileUniform': self._quantile_scale(sample_features)
        }
        
        for method_name, scaled_data in methods.items():
            if scaled_data is not None:
                mean_vals = np.nanmean(scaled_data, axis=0)
                std_vals = np.nanstd(scaled_data, axis=0)
                
                print(f"{method_name:15s}: μ_avg={np.nanmean(mean_vals):6.3f}, σ_avg={np.nanmean(std_vals):6.3f}")
            else:
                print(f"{method_name:15s}: ❌ Error en normalización")
    
    def _standard_scale(self, data):
        """StandardScaler manual"""
        try:
            mean_vals = np.nanmean(data, axis=0)
            std_vals = np.nanstd(data, axis=0)
            std_vals[std_vals == 0] = 1  # Evitar división por cero
            return (data - mean_vals) / std_vals
        except:
            return None
    
    def _minmax_scale(self, data):
        """MinMaxScaler manual"""
        try:
            min_vals = np.nanmin(data, axis=0)
            max_vals = np.nanmax(data, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Evitar división por cero
            return (data - min_vals) / range_vals
        except:
            return None
    
    def _robust_scale(self, data):
        """RobustScaler manual"""
        try:
            median_vals = np.nanmedian(data, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1  # Evitar división por cero
            return (data - median_vals) / iqr
        except:
            return None
    
    def _quantile_scale(self, data):
        """QuantileTransformer aproximado"""
        try:
            # Simplificado: mapear a percentiles
            result = np.zeros_like(data)
            for i in range(data.shape[1]):
                col_data = data[:, i]
                col_data = col_data[~np.isnan(col_data)]
                
                if len(col_data) > 0:
                    # Mapear a percentiles
                    ranks = np.searchsorted(np.sort(col_data), data[:, i])
                    result[:, i] = ranks / len(col_data)
                
            return result
        except:
            return None
    
    def generate_diagnosis_report(self):
        """Generar reporte final de diagnóstico"""
        print(f"\n" + "🔥" * 60)
        print(f"📋 REPORTE FINAL DE DIAGNÓSTICO")
        print("🔥" * 60)
        
        if not self.issues_found:
            print("✅ ¡NO SE ENCONTRARON PROBLEMAS CRÍTICOS!")
            print("💡 Las bajas confianzas pueden deberse a:")
            print("   - Diferentes versiones de librerías ML entre Windows/macOS")
            print("   - Diferencias en precisión de punto flotante")
            print("   - Modelos entrenados en un entorno específico")
        else:
            print(f"❌ SE ENCONTRARON {len(self.issues_found)} PROBLEMAS:")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"   {i}. {issue}")
            
            print(f"\n💡 RECOMENDACIONES:")
            print("   1. Verificar instalación de TA-Lib")
            print("   2. Revisar normalización de features")
            print("   3. Comparar features con versión de macOS")
            print("   4. Considerar re-entrenar modelos en Windows")

async def run_complete_diagnosis():
    """Ejecutar diagnóstico completo"""
    print("🔍 DIAGNÓSTICO COMPLETO DE FEATURES")
    print("=" * 60)
    
    diagnostics = FeaturesIssueDiagnostics()
    
    # 1. Verificar TA-Lib
    talib_available = diagnostics.check_talib_availability()
    
    # 2. Analizar features para cada par
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    features_data = {}
    
    for symbol in symbols:
        features_array = await diagnostics.analyze_real_data_features(symbol)
        if features_array is not None:
            features_data[symbol] = features_array
    
    # 3. Comparar normalización para BTC
    if "BTCUSDT" in features_data:
        diagnostics.compare_normalization_methods(features_data["BTCUSDT"])
    
    # 4. Generar reporte final
    diagnostics.generate_diagnosis_report()

if __name__ == "__main__":
    asyncio.run(run_complete_diagnosis()) 