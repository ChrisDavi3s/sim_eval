# simulation_benchmarks/calculators/__init__.py

import warnings
import sys

def warn_with_print(message, category, filename, lineno, file=None, line=None):
    print(f'\033[93mWarning: {message}\033[0m', file=sys.stderr)

warnings.showwarning = warn_with_print

# Always import these
from .base_calculator import PropertyCalculator
from .vasp_xml_calculator import VASPXMLPropertyCalculator

# Try to import NequIP-related classes
try:
    from .nequip_calculator import NequIPPropertyCalculator
except ImportError as e:
    message = f"NequIP is not installed. NequIPPropertyCalculator will not be available. To use it, please install NequIP using 'pip install nequip'. Error: {str(e)}"
    warnings.warn(message, ImportWarning)
    print(f'\033[93mWarning: {message}\033[0m', file=sys.stderr)
    NequIPPropertyCalculator = None

# Try to import CHGNet-related classes
try:
    from .chgnet_calculator import CHGNetPropertyCalculator
except ImportError as e:
    message = f"CHGNet is not installed. CHGNetPropertyCalculator will not be available. To use it, please install CHGNet using 'pip install chgnet'. Error: {str(e)}"
    warnings.warn(message, ImportWarning)
    print(f'\033[93mWarning: {message}\033[0m', file=sys.stderr)
    CHGNetPropertyCalculator = None

__all__ = ['PropertyCalculator', 'VASPXMLPropertyCalculator', 'NequIPPropertyCalculator', 'CHGNetPropertyCalculator']

