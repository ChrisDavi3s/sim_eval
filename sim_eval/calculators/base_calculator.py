# simulation_benchmarks/calculators/base_calculator.py

class PropertyCalculator:
    '''
    Base class for property calculators. Acts as a wrapper for different calculators (VASP, NequIP, etc.)
    '''
    def __init__(self, name, has_energy=True, has_forces=True, has_stress=True):
        self.name = name
        self.has_energy = has_energy
        self.has_forces = has_forces
        self.has_stress = has_stress

    def compute_properties(self, frames):
        raise NotImplementedError("Subclasses must implement the compute_properties method.")