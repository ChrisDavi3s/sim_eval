import warnings
import sys

# Always import these
from .base_calculator import PropertyCalculator
from .vasp_xml_dir_calculator import VASPXMLDiretoryPropertyCalculator
from .vasp_outcar_calculator import VASPOUTCARPropertyCalculator
from .vasp_outcar_dir_calculator import VASPOUTCARDirectoryPropertyCalculator

def warn_with_print(message, category, filename, lineno, file=None, line=None):
    print(f'\033[93mWarning: {message}\033[0m', file=sys.stderr)

warnings.showwarning = warn_with_print

# Import optional calculators
try:
    from .nequip_calculator import NequIPPropertyCalculator
except ImportError:
    warnings.warn("NequIP is not installed. NequIPPropertyCalculator will not be available. To use it, install NequIP using 'pip install nequip'", ImportWarning)
    NequIPPropertyCalculator = None

try:
    from .chgnet_calculator import CHGNetPropertyCalculator
except ImportError:
    warnings.warn("CHGNet is not installed. CHGNetPropertyCalculator will not be available. To use it, install CHGNet using 'pip install chgnet'", ImportWarning)
    CHGNetPropertyCalculator = None

try:
    from .mace_calculator import MACEPropertyCalculator
except ImportError:
    warnings.warn("MACE is not installed. MACEPropertyCalculator will not be available. To use it, install MACE using 'pip install mace-torch'", ImportWarning)
    MACEPropertyCalculator = None

try:
    from .schnet_calculator import SchnetPropertyCalculator
except ImportError:
    warnings.warn("Schnet is not installed. SchnetPropertyCalculator will not be available. To use it, install Schnet using 'pip install schnetpack'", ImportWarning)
    SchnetPropertyCalculator = None

# Define __all__
__all__ = ['PropertyCalculator', 'VASPXMLDiretoryPropertyCalculator', 'VASPOUTCARPropertyCalculator', 'VASPOUTCARDirectoryPropertyCalculator']

# Add optional calculators to __all__ if they are available
if NequIPPropertyCalculator is not None:
    __all__.append('NequIPPropertyCalculator')
if CHGNetPropertyCalculator is not None:
    __all__.append('CHGNetPropertyCalculator')
if MACEPropertyCalculator is not None:
    __all__.append('MACEPropertyCalculator')
if SchnetPropertyCalculator is not None:
    __all__.append('SchnetPropertyCalculator')