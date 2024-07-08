# simulation_benchmarks/calculators/base_calculator.py

class PropertyCalculator:
    '''
    Base class for property calculators. Acts as a wrapper for different calculators (VASP, NequIP, etc.)

    This class provides a common interface for various property calculators,
    allowing for easy integration of different calculation methods into the
    simulation benchmarks framework.

    Attributes:
        name (str): The name of the calculator. This is used in plots / metrics etc.
        has_energy (bool): Whether the calculator can compute energy.
        has_forces (bool): Whether the calculator can compute forces.
        has_stress (bool): Whether the calculator can compute stress.
    '''


    def __init__(self, name, has_energy=True, has_forces=True, has_stress=True):
        """
        Initialize a PropertyCalculator.

        Args:
            name (str): The name of the calculator.
            has_energy (bool, optional): Whether the calculator can compute energy. Defaults to True.
            has_forces (bool, optional): Whether the calculator can compute forces. Defaults to True.
            has_stress (bool, optional): Whether the calculator can compute stress. Defaults to True.
        """
        self.name = name
        self.has_energy = has_energy
        self.has_forces = has_forces
        self.has_stress = has_stress


    def compute_properties(self, frames):
        """"
        Compute properties for the given frames.

        This method should be implemented by subclasses to perform the actual
        property calculations using the specific calculator being wrapped.

        Args:
            frames: The frames object containing the atomic structures for which
                    properties should be calculated.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """

        raise NotImplementedError("Subclasses must implement the compute_properties method.")