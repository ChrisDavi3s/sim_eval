from tqdm import tqdm
try:
    from nequip.ase import NequIPCalculator
except ImportError as e:
    NequIPCalculator = None
    print(e)
from .base_calculator import PropertyCalculator

class NequIPPropertyCalculator(PropertyCalculator):
    '''
    Implementation of PropertyCalculator for NequIP models (.pth files)
    '''
    def __init__(self, name, has_energy=True, has_forces=True, has_stress=True, **kwargs):
        """
        Initialize a NequIPPropertyCalculator.

        Args:
            name (str): The name of the calculator.
            has_energy (bool, optional): Whether the calculator can compute energy. Defaults to True.
            has_forces (bool, optional): Whether the calculator can compute forces. Defaults to True.
            has_stress (bool, optional): Whether the calculator can compute stress. Defaults to True.
            **kwargs: Additional keyword arguments for specific calculators.
                model_path (str, optional): Path to the NequIP model file. Defaults to None (No foundation model so this will fail).
                device (str, optional): Device to use for computation. Defaults to NequIP default (ie cuda if available).
        """
        if NequIPCalculator is None:
            raise ImportError("NequIP is not installed. Please install it using 'pip install SimulationBenchmarks[nequip]'.")
        super().__init__(name, has_energy, has_forces, has_stress)
        self.model_path = kwargs.get('model_path', None)
        self.device = kwargs.get('device', "cpu")

    def compute_properties(self, frames):
        if NequIPCalculator is None:
            raise ImportError("NequIP is not installed. Please install it using 'pip install nequip'.")
        calc = NequIPCalculator.from_deployed_model(self.model_path, device=self.device)
        
        for i, atom in tqdm(enumerate(frames.frames), total=len(frames.frames), desc=f"Computing {self.name} properties"):
            atom.calc = calc
            if self.has_energy:
                if f'{self.name}_total_energy' in atom.info:
                    raise ValueError(f"{self.name}_total_energy already exists in atom {i}")
                atom.info[f'{self.name}_total_energy'] = atom.get_potential_energy()
            if self.has_forces:
                if f'{self.name}_forces' in atom.arrays:
                    raise ValueError(f"{self.name}_forces already exists in atom {i}")
                atom.arrays[f'{self.name}_forces'] = atom.get_forces()
            if self.has_stress:
                if f'{self.name}_stress' in atom.info:
                    raise ValueError(f"{self.name}_stress already exists in atom {i}")
                atom.info[f'{self.name}_stress'] = atom.get_stress()