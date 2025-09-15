"""Test Phase 2 Integration"""

from src.integration.phase2_factory import Phase2SystemFactory

def test_integration():
    """Test Phase 2 integration factory"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


    # Test factory initialization
    factory = Phase2SystemFactory()
    print('Phase 2 Factory: Initialized')

    # Test Phase 1 initialization
    try:
        phase1 = factory.initialize_phase1_systems()
        print(f'Phase 1 Systems: {len(phase1)} systems initialized')
        print('  Systems:', ', '.join(phase1.keys()))
    except Exception as e:
        print(f'Phase 1 Error: {e}')
        return False

    # Test Phase 2 initialization
    try:
        phase2 = factory.initialize_phase2_systems()
        print(f'Phase 2 Systems: {len(phase2)} systems initialized')
        print('  Systems:', ', '.join(phase2.keys()))
    except Exception as e:
        print(f'Phase 2 Error: {e}')
        return False

    # Test validation
    try:
        validation = factory.validate_integration()
        ready = validation.get('all_systems_ready', False)
        print(f'\nIntegration Status: {"READY" if ready else "NOT READY"}')

        # Show validation details
        print('\nValidation Details:')
        for key, value in validation.items():
            if key != 'all_systems_ready':
                status = 'OK' if value else 'FAIL'
                print(f'  {key}: {status}')

        return ready
    except Exception as e:
        print(f'Validation Error: {e}')
        return False

if __name__ == "__main__":
    success = test_integration()

    if success:
        print("\n[SUCCESS] PHASE 2 INTEGRATION: COMPLETE")
        print("All systems are properly wired and integrated!")
    else:
        print("\n[FAILED] PHASE 2 INTEGRATION: INCOMPLETE")
        print("Integration issues need to be resolved")