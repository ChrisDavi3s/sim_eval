import warnings
import sys
from importlib import import_module

# Always import these
from .base_calculator import PropertyCalculator
from .vasp_xml_calculator import VASPXMLPropertyCalculator


def warn_with_print(message, category, filename, lineno, file=None, line=None):
    print(f'\033[93mWarning: {message}\033[0m', file=sys.stderr)


warnings.showwarning = warn_with_print


# Define a function to handle optional imports
def import_optional(module_name, class_name, install_command):
    try:
        module = import_module(f'.{module_name}_calculator', package=__package__)
        return getattr(module, f'{class_name}PropertyCalculator')
    except ImportError:
        message = f"{class_name} is not installed. {class_name}PropertyCalculator will not be available. To use it, install {class_name} using '{install_command}'"
        warnings.warn(message, ImportWarning)
        return None


# Import optional calculators
NequIPPropertyCalculator = import_optional('nequip', 'NequIP', 'pip install nequip')
CHGNetPropertyCalculator = import_optional('chgnet', 'CHGNet', 'pip install chgnet')
MACEPropertyCalculator = import_optional('mace', 'MACE', 'pip install mace-torch')


__all__ = ['PropertyCalculator', 
           'VASPXMLPropertyCalculator', 
           'NequIPPropertyCalculator', 
           'CHGNetPropertyCalculator',
           'MACEPropertyCalculator']