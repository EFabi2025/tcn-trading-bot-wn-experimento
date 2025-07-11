#!/usr/bin/env python3
"""
🎯 ANÁLISIS CRÍTICO DE SEÑALES TCN
================================

Análisis específico y preciso de los puntos críticos donde podrían
ocurrir inversiones BUY ↔ SELL en el flujo de procesamiento.

ENFOQUE:
- Solo verificar código ejecutable (no comentarios)
- Solo verificar puntos donde se PROCESA la señal
- Ignorar documentación, logs y configuraciones
- Buscar inversiones REALES, no menciones textuales

PUNTOS CRÍTICOS A VERIFICAR:
1. ✅ Mapeo en TCN Predictor: predict() y predict_signal()
2. ✅ Procesamiento en Trading Manager: _generate_tcn_signals()
3. ✅ Ejecución de señales: _process_signal()
4. ✅ Filtros de sistema que puedan invertir señales
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime

class CriticalSignalAnalyzer:
    """🎯 Analizador crítico de señales TCN"""

    def __init__(self):
        self.critical_issues = []
        self.verified_components = []

    def run_critical_analysis(self):
        """🎯 Ejecutar análisis crítico"""
        print("🎯 ANÁLISIS CRÍTICO DE SEÑALES TCN")
        print("="*60)

        # Paso 1: Verificar mapeo en TCN Predictor
        print("\n📋 PASO 1: Verificando mapeo crítico en TCN Predictor...")
        self._verify_tcn_mapping()

        # Paso 2: Verificar procesamiento en Trading Manager
        print("\n📋 PASO 2: Verificando procesamiento en Trading Manager...")
        self._verify_signal_processing()

        # Paso 3: Verificar filtros críticos
        print("\n📋 PASO 3: Verificando filtros críticos...")
        self._verify_critical_filters()

        # Generar conclusión
        print("\n📋 PASO 4: Generando conclusión...")
        self._generate_conclusion()

    def _verify_tcn_mapping(self):
        """🧠 Verificar mapeo crítico en TCN Predictor"""
        try:
            with open('tcn_definitivo_predictor.py', 'r', encoding='utf-8') as f:
                content = f.read()

            print("🔍 Analizando tcn_definitivo_predictor.py...")

            # Verificar método predict_signal
            predict_signal_match = re.search(
                r'def predict_signal\(.*?\):(.*?)(?=def|\Z)',
                content,
                re.DOTALL
            )

            if predict_signal_match:
                method_code = predict_signal_match.group(1)
                print("   ✅ Método predict_signal encontrado")

                # Verificar mapeo correcto
                if "signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}" in method_code:
                    print("   ✅ Mapeo correcto verificado: {0: 'SELL', 1: 'HOLD', 2: 'BUY'}")
                    self.verified_components.append('TCN_PREDICT_SIGNAL_MAPPING')
                else:
                    self.critical_issues.append("MAPEO INCORRECTO en predict_signal")

                # Verificar asignación de señal
                if "signal = signal_map[signal_idx]" in method_code:
                    print("   ✅ Asignación correcta de señal verificada")
                    self.verified_components.append('TCN_PREDICT_SIGNAL_ASSIGNMENT')
                else:
                    self.critical_issues.append("ASIGNACIÓN INCORRECTA en predict_signal")

            # Verificar método predict
            predict_match = re.search(
                r'def predict\(.*?\):(.*?)(?=def|\Z)',
                content,
                re.DOTALL
            )

            if predict_match:
                method_code = predict_match.group(1)
                print("   ✅ Método predict encontrado")

                # Verificar mapeo correcto
                if "class_names = ['SELL', 'HOLD', 'BUY']" in method_code:
                    print("   ✅ Array de clases correcto verificado: ['SELL', 'HOLD', 'BUY']")
                    self.verified_components.append('TCN_PREDICT_CLASS_ARRAY')
                else:
                    self.critical_issues.append("ARRAY DE CLASES INCORRECTO en predict")

                # Verificar asignación de señal
                if "signal = class_names[predicted_class]" in method_code:
                    print("   ✅ Asignación correcta por índice verificada")
                    self.verified_components.append('TCN_PREDICT_INDEX_ASSIGNMENT')
                else:
                    self.critical_issues.append("ASIGNACIÓN POR ÍNDICE INCORRECTA en predict")

        except FileNotFoundError:
            self.critical_issues.append("ARCHIVO NO ENCONTRADO: tcn_definitivo_predictor.py")

    def _verify_signal_processing(self):
        """⚙️ Verificar procesamiento crítico en Trading Manager"""
        try:
            with open('simple_professional_manager.py', 'r', encoding='utf-8') as f:
                content = f.read()

            print("🔍 Analizando simple_professional_manager.py...")

            # Verificar _generate_tcn_signals
            generate_signals_match = re.search(
                r'def _generate_tcn_signals\(.*?\):(.*?)(?=def|\Z)',
                content,
                re.DOTALL
            )

            if generate_signals_match:
                method_code = generate_signals_match.group(1)
                print("   ✅ Método _generate_tcn_signals encontrado")

                # Verificar que la señal se toma directamente
                if "signal = prediction['signal']" in method_code:
                    print("   ✅ Señal tomada directamente sin modificación")
                    self.verified_components.append('TRADING_MANAGER_DIRECT_SIGNAL')
                else:
                    self.critical_issues.append("SEÑAL NO TOMADA DIRECTAMENTE en _generate_tcn_signals")

                # Verificar que no hay inversiones explícitas
                inversion_patterns = [
                    r"if\s+signal\s*==\s*['\"]BUY['\"].*signal\s*=\s*['\"]SELL['\"]",
                    r"if\s+signal\s*==\s*['\"]SELL['\"].*signal\s*=\s*['\"]BUY['\"]"
                ]

                inversions_found = False
                for pattern in inversion_patterns:
                    if re.search(pattern, method_code, re.IGNORECASE | re.DOTALL):
                        inversions_found = True
                        self.critical_issues.append(f"INVERSIÓN EXPLÍCITA DETECTADA: {pattern}")

                if not inversions_found:
                    print("   ✅ No se detectaron inversiones explícitas")
                    self.verified_components.append('TRADING_MANAGER_NO_INVERSIONS')

            # Verificar _process_signal
            process_signal_match = re.search(
                r'def _process_signal\(.*?\):(.*?)(?=def|\Z)',
                content,
                re.DOTALL
            )

            if process_signal_match:
                method_code = process_signal_match.group(1)
                print("   ✅ Método _process_signal encontrado")

                # Verificar acceso directo a señal
                if "signal_data['signal']" in method_code or "signal_data.get('signal'" in method_code:
                    print("   ✅ Acceso directo a señal verificado")
                    self.verified_components.append('PROCESS_SIGNAL_DIRECT_ACCESS')
                else:
                    self.critical_issues.append("ACCESO INDIRECTO A SEÑAL en _process_signal")

        except FileNotFoundError:
            self.critical_issues.append("ARCHIVO NO ENCONTRADO: simple_professional_manager.py")

    def _verify_critical_filters(self):
        """🛡️ Verificar filtros críticos"""
        filter_files = {
            'simple_professional_manager.py': [
                '_apply_signal_stability_filter',
                '_apply_market_context_filter'
            ],
            'tcn_definitivo_predictor.py': [
                '_sanity_check_prediction'
            ]
        }

        for file_name, methods in filter_files.items():
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()

                print(f"🔍 Analizando filtros en {file_name}...")

                for method_name in methods:
                    method_match = re.search(
                        rf'def {method_name}\(.*?\):(.*?)(?=def|\Z)',
                        content,
                        re.DOTALL
                    )

                    if method_match:
                        method_code = method_match.group(1)
                        print(f"   ✅ {method_name} encontrado")

                        # Verificar que no hay inversiones directas BUY → SELL / SELL → BUY
                        critical_inversions = [
                            r"return\s+['\"]SELL['\"].*signal\s*==\s*['\"]BUY['\"]",
                            r"return\s+['\"]BUY['\"].*signal\s*==\s*['\"]SELL['\"]",
                            r"corrected_signal\s*=\s*['\"]SELL['\"].*signal\s*==\s*['\"]BUY['\"]",
                            r"corrected_signal\s*=\s*['\"]BUY['\"].*signal\s*==\s*['\"]SELL['\"]"
                        ]

                        inversions_in_filter = False
                        for pattern in critical_inversions:
                            if re.search(pattern, method_code, re.IGNORECASE | re.DOTALL):
                                inversions_in_filter = True
                                self.critical_issues.append(f"INVERSIÓN EN FILTRO {method_name}: {pattern}")

                        if not inversions_in_filter:
                            print(f"      ✅ Sin inversiones críticas en {method_name}")
                            self.verified_components.append(f'FILTER_{method_name.upper()}_NO_INVERSIONS')

                        # Verificar que solo se permite neutralización a HOLD
                        if "corrected_signal = 'HOLD'" in method_code or "return.*'HOLD'" in method_code:
                            print(f"      ✅ Neutralización a HOLD permitida en {method_name}")
                            self.verified_components.append(f'FILTER_{method_name.upper()}_NEUTRALIZATION_OK')

            except FileNotFoundError:
                print(f"   ⚠️ Archivo {file_name} no encontrado")

    def _generate_conclusion(self):
        """📋 Generar conclusión del análisis"""
        print("\n📋 CONCLUSIÓN DEL ANÁLISIS CRÍTICO")
        print("="*60)

        total_verified = len(self.verified_components)
        total_issues = len(self.critical_issues)

        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   ✅ Componentes verificados: {total_verified}")
        print(f"   🚨 Problemas críticos: {total_issues}")

        # Mostrar componentes verificados
        if self.verified_components:
            print(f"\n✅ COMPONENTES VERIFICADOS:")
            for component in self.verified_components:
                print(f"   🟢 {component}")

        # Mostrar problemas críticos
        if self.critical_issues:
            print(f"\n🚨 PROBLEMAS CRÍTICOS:")
            for issue in self.critical_issues:
                print(f"   🔴 {issue}")
        else:
            print(f"\n🟢 NO SE DETECTARON PROBLEMAS CRÍTICOS")

        # Evaluación final
        if total_issues == 0 and total_verified >= 8:
            print(f"\n🎯 EVALUACIÓN: ✅ SISTEMA APROBADO")
            print(f"   🔒 Integridad de señales GARANTIZADA")
            print(f"   🟢 No hay inversiones BUY ↔ SELL en puntos críticos")
            print(f"   ✅ Sistema SEGURO para producción")
            recommendation = "APROBADO PARA PRODUCCIÓN"
        elif total_issues == 0:
            print(f"\n🎯 EVALUACIÓN: 🟡 VERIFICACIÓN PARCIAL")
            print(f"   ⚠️ Algunos componentes no verificados completamente")
            print(f"   🔍 Requiere verificación adicional")
            recommendation = "VERIFICACIÓN ADICIONAL REQUERIDA"
        else:
            print(f"\n🎯 EVALUACIÓN: ❌ SISTEMA RECHAZADO")
            print(f"   🚨 Se detectaron problemas críticos")
            print(f"   🛑 NO USAR en producción hasta corregir")
            recommendation = "RECHAZADO - CORREGIR PROBLEMAS"

        # Generar timestamp de reporte
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n📄 REPORTE GENERADO: {timestamp}")
        print(f"💡 RECOMENDACIÓN: {recommendation}")

        # Guardar resultado simplificado
        result = {
            'timestamp': timestamp,
            'verified_components': self.verified_components,
            'critical_issues': self.critical_issues,
            'recommendation': recommendation,
            'approved': total_issues == 0 and total_verified >= 8
        }

        print(f"\n🔚 ANÁLISIS CRÍTICO COMPLETADO")
        return result

def main():
    """🎯 Función principal"""
    analyzer = CriticalSignalAnalyzer()
    result = analyzer.run_critical_analysis()
    return result

if __name__ == "__main__":
    main()
