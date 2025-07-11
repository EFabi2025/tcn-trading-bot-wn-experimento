#!/usr/bin/env python3
"""
🔍 VERIFICACIÓN FINAL DE INTEGRIDAD DE SEÑALES TCN
================================================

Script final para verificar que NO existen inversiones BUY ↔ SELL
en ningún punto del flujo de procesamiento de señales TCN.

PUNTOS CRÍTICOS VERIFICADOS:
1. ✅ Mapeo del modelo: {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
2. ✅ TCN Predictor: predict() y predict_signal()
3. ✅ Trading Manager: _generate_tcn_signals()
4. ✅ Filtros: stability, context, sanity
5. ✅ Signal Processing: _process_signal()
6. ✅ Ejecución: Sin cambios BUY→SELL / SELL→BUY

CRITERIOS DE APROBACIÓN:
- Mapeo matemáticamente correcto
- No hay inversiones en ningún punto
- Filtros solo neutralizan, nunca invierten
- Señal final preserva intención del modelo
"""

import re
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

class SignalFlowVerifier:
    """🔍 Verificador completo del flujo de señales"""

    def __init__(self):
        self.verification_results = []
        self.critical_issues = []
        self.warnings = []

    def run_complete_verification(self):
        """🎯 Ejecutar verificación completa"""
        print("🔍 VERIFICACIÓN FINAL DE INTEGRIDAD DE SEÑALES TCN")
        print("="*80)

        # Paso 1: Verificar mapeo en TCN Predictor
        print("\n📋 PASO 1: Verificando mapeo de señales en TCN Predictor...")
        self._verify_tcn_predictor_mapping()

        # Paso 2: Verificar procesamiento en Trading Manager
        print("\n📋 PASO 2: Verificando procesamiento en Trading Manager...")
        self._verify_trading_manager_processing()

        # Paso 3: Verificar filtros del sistema
        print("\n📋 PASO 3: Verificando filtros del sistema...")
        self._verify_system_filters()

        # Paso 4: Verificar ejecución de señales
        print("\n📋 PASO 4: Verificando ejecución de señales...")
        self._verify_signal_execution()

        # Paso 5: Búsqueda de inversiones en código
        print("\n📋 PASO 5: Búsqueda de inversiones en código...")
        self._search_for_inversions_in_code()

        # Generar reporte final
        print("\n📋 PASO 6: Generando reporte final...")
        self._generate_final_report()

    def _verify_tcn_predictor_mapping(self):
        """🧠 Verificar mapeo correcto en TCN Predictor"""
        print("🔍 Verificando tcn_definitivo_predictor.py...")

        try:
            with open('tcn_definitivo_predictor.py', 'r', encoding='utf-8') as f:
                content = f.read()

            # Buscar definiciones de mapeo
            mapping_patterns = [
                r"signal_map\s*=\s*\{([^}]+)\}",
                r"class_names\s*=\s*\[([^\]]+)\]",
                r"\{0:\s*['\"]SELL['\"],\s*1:\s*['\"]HOLD['\"],\s*2:\s*['\"]BUY['\"]\}",
                r"\[['\"](SELL|BUY|HOLD)['\"],\s*['\"](SELL|BUY|HOLD)['\"],\s*['\"](SELL|BUY|HOLD)['\"]\]"
            ]

            mappings_found = []
            for pattern in mapping_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    mappings_found.extend(matches)

            # Verificar mapeo correcto específico
            correct_mapping_found = False
            if "{0: 'SELL', 1: 'HOLD', 2: 'BUY'}" in content:
                correct_mapping_found = True
                print("   ✅ Mapeo correcto encontrado: {0: 'SELL', 1: 'HOLD', 2: 'BUY'}")
            elif "['SELL', 'HOLD', 'BUY']" in content:
                correct_mapping_found = True
                print("   ✅ Array correcto encontrado: ['SELL', 'HOLD', 'BUY']")

            # Buscar mapeos incorrectos (inversiones)
            incorrect_patterns = [
                r"\{0:\s*['\"]BUY['\"],.*2:\s*['\"]SELL['\"]\}",  # Inversión directa
                r"\[['\"](BUY)['\"],.*['\"](SELL)['\"]\]"         # Array invertido
            ]

            inversions_found = False
            for pattern in incorrect_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    inversions_found = True
                    self.critical_issues.append(f"INVERSIÓN DETECTADA en tcn_definitivo_predictor.py: {pattern}")

            # Verificar consistencia entre métodos predict y predict_signal
            predict_methods = re.findall(r'def (predict\w*)\(.*?\):', content)
            print(f"   📊 Métodos de predicción encontrados: {predict_methods}")

            self.verification_results.append({
                'component': 'TCN_PREDICTOR',
                'mappings_found': len(mappings_found),
                'correct_mapping': correct_mapping_found,
                'inversions_found': inversions_found,
                'status': 'PASS' if correct_mapping_found and not inversions_found else 'FAIL'
            })

        except FileNotFoundError:
            self.critical_issues.append("ARCHIVO NO ENCONTRADO: tcn_definitivo_predictor.py")
            print("   ❌ Archivo tcn_definitivo_predictor.py no encontrado")

    def _verify_trading_manager_processing(self):
        """⚙️ Verificar procesamiento en Trading Manager"""
        print("🔍 Verificando simple_professional_manager.py...")

        try:
            with open('simple_professional_manager.py', 'r', encoding='utf-8') as f:
                content = f.read()

            # Buscar el método _generate_tcn_signals
            signal_generation_match = re.search(
                r'def _generate_tcn_signals.*?(?=def|\Z)',
                content,
                re.DOTALL
            )

            if signal_generation_match:
                signal_code = signal_generation_match.group(0)
                print("   ✅ Método _generate_tcn_signals encontrado")

                # Verificar que la señal se toma directamente sin modificación
                direct_signal_patterns = [
                    r"signal\s*=\s*prediction\[.signal.\]",
                    r"signal\s*=\s*prediction\.get\(.signal.\)",
                ]

                direct_assignment = False
                for pattern in direct_signal_patterns:
                    if re.search(pattern, signal_code):
                        direct_assignment = True
                        print("   ✅ Asignación directa de señal detectada")
                        break

                # Buscar posibles inversiones en el código
                inversion_patterns = [
                    r"if.*signal.*==.*['\"]BUY['\"].*signal.*=.*['\"]SELL['\"]",
                    r"if.*signal.*==.*['\"]SELL['\"].*signal.*=.*['\"]BUY['\"]",
                    r"signal.*=.*['\"]SELL['\"].*if.*['\"]BUY['\"]",
                    r"signal.*=.*['\"]BUY['\"].*if.*['\"]SELL['\"]",
                ]

                inversions_detected = False
                for pattern in inversion_patterns:
                    if re.search(pattern, signal_code, re.IGNORECASE):
                        inversions_detected = True
                        self.critical_issues.append(f"POSIBLE INVERSIÓN en _generate_tcn_signals: {pattern}")

                # Verificar filtros aplicados
                filter_calls = re.findall(r'_apply_\w+_filter', signal_code)
                print(f"   📊 Filtros aplicados: {filter_calls}")

                self.verification_results.append({
                    'component': 'TRADING_MANAGER',
                    'method_found': True,
                    'direct_assignment': direct_assignment,
                    'inversions_detected': inversions_detected,
                    'filters_found': len(filter_calls),
                    'status': 'PASS' if direct_assignment and not inversions_detected else 'FAIL'
                })

            else:
                self.critical_issues.append("MÉTODO NO ENCONTRADO: _generate_tcn_signals")
                print("   ❌ Método _generate_tcn_signals no encontrado")

        except FileNotFoundError:
            self.critical_issues.append("ARCHIVO NO ENCONTRADO: simple_professional_manager.py")
            print("   ❌ Archivo simple_professional_manager.py no encontrado")

    def _verify_system_filters(self):
        """🛡️ Verificar que los filtros no invierten señales"""
        print("🔍 Verificando filtros del sistema...")

        filter_files = [
            'simple_professional_manager.py',
            'tcn_definitivo_predictor.py'
        ]

        filter_methods = [
            '_apply_signal_stability_filter',
            '_apply_market_context_filter',
            '_sanity_check_prediction'
        ]

        for file_name in filter_files:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    content = f.read()

                print(f"   📄 Analizando {file_name}...")

                for method_name in filter_methods:
                    method_match = re.search(
                        rf'def {method_name}.*?(?=def|\Z)',
                        content,
                        re.DOTALL
                    )

                    if method_match:
                        method_code = method_match.group(0)
                        print(f"      ✅ {method_name} encontrado")

                        # Verificar que no hay inversiones BUY ↔ SELL
                        inversion_patterns = [
                            r"return.*['\"]SELL['\"].*if.*['\"]BUY['\"]",
                            r"return.*['\"]BUY['\"].*if.*['\"]SELL['\"]",
                            r"['\"]SELL['\"].*if.*signal.*==.*['\"]BUY['\"]",
                            r"['\"]BUY['\"].*if.*signal.*==.*['\"]SELL['\"]",
                        ]

                        for pattern in inversion_patterns:
                            if re.search(pattern, method_code, re.IGNORECASE):
                                self.critical_issues.append(f"INVERSIÓN EN FILTRO {method_name}: {pattern}")

                        # Verificar que solo se permite neutralización (→ HOLD)
                        neutralization_patterns = [
                            r"return.*['\"]HOLD['\"]",
                            r"corrected_signal.*=.*['\"]HOLD['\"]",
                        ]

                        neutralization_found = False
                        for pattern in neutralization_patterns:
                            if re.search(pattern, method_code, re.IGNORECASE):
                                neutralization_found = True
                                break

                        if neutralization_found:
                            print(f"         ✅ Neutralización a HOLD permitida")

            except FileNotFoundError:
                print(f"   ⚠️ Archivo {file_name} no encontrado")

    def _verify_signal_execution(self):
        """⚡ Verificar ejecución de señales"""
        print("🔍 Verificando ejecución de señales...")

        try:
            with open('simple_professional_manager.py', 'r', encoding='utf-8') as f:
                content = f.read()

            # Buscar método _process_signal
            process_signal_match = re.search(
                r'def _process_signal.*?(?=def|\Z)',
                content,
                re.DOTALL
            )

            if process_signal_match:
                process_code = process_signal_match.group(0)
                print("   ✅ Método _process_signal encontrado")

                # Verificar que las señales se procesan sin modificación
                signal_access_patterns = [
                    r"signal_data\[.signal.\]",
                    r"signal_data\.get\(.signal.\)",
                ]

                signal_access_found = False
                for pattern in signal_access_patterns:
                    if re.search(pattern, process_code):
                        signal_access_found = True
                        print("   ✅ Acceso directo a señal detectado")
                        break

                # Verificar que no hay modificaciones de BUY a SELL o viceversa
                modification_patterns = [
                    r"if.*BUY.*SELL",
                    r"if.*SELL.*BUY",
                    r"BUY.*=.*SELL",
                    r"SELL.*=.*BUY",
                ]

                modifications_found = False
                for pattern in modification_patterns:
                    if re.search(pattern, process_code, re.IGNORECASE):
                        modifications_found = True
                        self.critical_issues.append(f"MODIFICACIÓN DE SEÑAL en _process_signal: {pattern}")

                self.verification_results.append({
                    'component': 'SIGNAL_EXECUTION',
                    'method_found': True,
                    'signal_access': signal_access_found,
                    'modifications_found': modifications_found,
                    'status': 'PASS' if signal_access_found and not modifications_found else 'FAIL'
                })

            else:
                self.critical_issues.append("MÉTODO NO ENCONTRADO: _process_signal")
                print("   ❌ Método _process_signal no encontrado")

        except FileNotFoundError:
            self.critical_issues.append("ARCHIVO NO ENCONTRADO para verificación de ejecución")

    def _search_for_inversions_in_code(self):
        """🔍 Búsqueda exhaustiva de inversiones en todo el código"""
        print("🔍 Búsqueda exhaustiva de inversiones...")

        python_files = [f for f in os.listdir('.') if f.endswith('.py')]

        # Patrones críticos de inversión
        critical_inversion_patterns = [
            (r"BUY.*SELL|SELL.*BUY", "Intercambio potencial BUY-SELL"),
            (r"\{0.*BUY.*2.*SELL\}", "Mapeo invertido en diccionario"),
            (r"\[.*BUY.*SELL.*\]", "Array con orden incorrecto"),
            (r"if.*signal.*==.*BUY.*signal.*=.*SELL", "Conversión directa BUY→SELL"),
            (r"if.*signal.*==.*SELL.*signal.*=.*BUY", "Conversión directa SELL→BUY"),
        ]

        for file_name in python_files:
            try:
                with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern, description in critical_inversion_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Filtrar coincidencias en comentarios o strings benignos
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if re.search(pattern, line, re.IGNORECASE):
                                # Verificar si es un comentario o string de documentación
                                stripped = line.strip()
                                if (not stripped.startswith('#') and
                                    not stripped.startswith('"""') and
                                    not stripped.startswith("'''") and
                                    'print(' not in line and
                                    'log' not in line.lower()):

                                    self.critical_issues.append(
                                        f"POSIBLE INVERSIÓN en {file_name}:{i+1} - {description}: {line.strip()}"
                                    )

            except Exception as e:
                print(f"   ⚠️ Error leyendo {file_name}: {e}")

        print(f"   📊 {len(python_files)} archivos Python analizados")

    def _generate_final_report(self):
        """📋 Generar reporte final"""
        print("\n📋 REPORTE FINAL DE VERIFICACIÓN DE INTEGRIDAD")
        print("="*80)

        # Contar componentes que pasaron
        passed_components = sum(1 for result in self.verification_results if result['status'] == 'PASS')
        total_components = len(self.verification_results)

        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   ✅ Componentes verificados: {total_components}")
        print(f"   🟢 Componentes aprobados: {passed_components}")
        print(f"   🔴 Componentes fallidos: {total_components - passed_components}")
        print(f"   🚨 Problemas críticos: {len(self.critical_issues)}")
        print(f"   ⚠️ Advertencias: {len(self.warnings)}")

        # Calcular score de integridad
        if total_components > 0:
            integrity_score = (passed_components / total_components) * 100
            if len(self.critical_issues) > 0:
                integrity_score *= 0.5  # Penalización por problemas críticos
        else:
            integrity_score = 0

        print(f"\n🎯 SCORE DE INTEGRIDAD: {integrity_score:.1f}%")

        # Evaluar nivel de integridad
        if integrity_score >= 95 and len(self.critical_issues) == 0:
            print("🟢 INTEGRIDAD EXCELENTE: Sistema completamente verificado")
            recommendation = "Sistema seguro para producción"
        elif integrity_score >= 80 and len(self.critical_issues) == 0:
            print("🟡 INTEGRIDAD BUENA: Sistema funcional con advertencias menores")
            recommendation = "Sistema operativo con monitoreo"
        elif len(self.critical_issues) > 0:
            print("🔴 INTEGRIDAD COMPROMETIDA: Problemas críticos detectados")
            recommendation = "DETENER OPERACIONES - Revisar problemas críticos"
        else:
            print("🟠 INTEGRIDAD DUDOSA: Verificación incompleta")
            recommendation = "Completar verificación antes de operar"

        # Mostrar problemas críticos
        if self.critical_issues:
            print(f"\n🚨 PROBLEMAS CRÍTICOS DETECTADOS:")
            for issue in self.critical_issues:
                print(f"   🔴 {issue}")

        # Mostrar advertencias
        if self.warnings:
            print(f"\n⚠️ ADVERTENCIAS:")
            for warning in self.warnings:
                print(f"   🟡 {warning}")

        # Generar resumen por componente
        print(f"\n📊 RESUMEN POR COMPONENTE:")
        for result in self.verification_results:
            status_emoji = "✅" if result['status'] == 'PASS' else "❌"
            print(f"   {status_emoji} {result['component']}: {result['status']}")

        print(f"\n💡 RECOMENDACIÓN: {recommendation}")

        # Guardar reporte detallado
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'integrity_score': integrity_score,
            'verification_results': self.verification_results,
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'recommendation': recommendation,
            'summary': {
                'total_components': total_components,
                'passed_components': passed_components,
                'critical_issues_count': len(self.critical_issues),
                'warnings_count': len(self.warnings)
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'final_signal_verification_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"\n📄 Reporte completo guardado: {filename}")
        print("\n🔚 VERIFICACIÓN FINAL COMPLETADA")

def main():
    """🎯 Función principal"""
    verifier = SignalFlowVerifier()
    verifier.run_complete_verification()

if __name__ == "__main__":
    main()
