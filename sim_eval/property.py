from enum import Enum

class Property(Enum):
    """Enum for different types of properties with their corresponding units."""
    ENERGY = ('energy', 'eV')
    FORCES = ('forces', 'eV/Å')
    STRESS = ('stress', 'eV/Å³')

    def __init__(self, key: str, unit: str):
        self.key = key
        self.unit = unit

    def get_units(self):
        return self.unit

    def get_name(self):
        return self.key
