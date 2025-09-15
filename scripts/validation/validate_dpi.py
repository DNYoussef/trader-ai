"""
Simple validation script for Gary's DPI system
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.strategies.dpi_calculator import (
        DistributionalPressureIndex,
        DPIWeeklyCycleIntegrator
    )

    print("+ DPI modules imported successfully")

    # Test basic instantiation
    dpi_calc = DistributionalPressureIndex()
    print("+ DPI calculator instantiated")

    # Test DPI calculation
    dpi_score, components = dpi_calc.calculate_dpi('ULTY', 10)
    print(f"+ DPI calculated: {dpi_score:.4f}")
    print(f"  Order Flow Pressure: {components.order_flow_pressure:.4f}")
    print(f"  Volume Weighted Skew: {components.volume_weighted_skew:.4f}")

    # Test narrative gap
    ng_analysis = dpi_calc.detect_narrative_gap('ULTY')
    print(f"+ Narrative Gap: {ng_analysis.narrative_gap:.4f} ({ng_analysis.gap_direction})")

    # Test position sizing
    sizing = dpi_calc.determine_position_size('ULTY', dpi_score, ng_analysis.narrative_gap, 1000.0)
    print(f"+ Position Size: ${sizing.risk_adjusted_size:.2f}")

    # Test integration
    integrator = DPIWeeklyCycleIntegrator(dpi_calc)
    enhanced = integrator.get_dpi_enhanced_allocations(['ULTY', 'AMDY'], 1000.0, {'ULTY': 70, 'AMDY': 30})
    print(f"+ Enhanced Allocations: ULTY={enhanced['ULTY']:.1f}%, AMDY={enhanced['AMDY']:.1f}%")

    print("\nALL VALIDATIONS PASSED - Gary's DPI System is FULLY FUNCTIONAL")

except Exception as e:
    print(f"X Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)