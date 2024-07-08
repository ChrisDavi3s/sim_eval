from tqdm import tqdm
try:
    from nequip.ase import NequIPCalculator
except ImportError:
    NequIPCalculator = None
from .base_calculator import PropertyCalculator

class NequIPPropertyCalculator(PropertyCalculator):
    '''
    Implementation of PropertyCalculator for NequIP models (.pth files)
    '''
    def __init__(self, name, model_path, has_energy=True, has_forces=True, has_stress=True):
        if NequIPCalculator is None:
            raise ImportError("NequIP is not installed. Please install it using 'pip install SimulationBenchmarks[nequip]'.")
        super().__init__(name, has_energy, has_forces, has_stress)
        self.model_path = model_path

    def compute_properties(self, trajectory):
        if NequIPCalculator is None:
            raise ImportError("NequIP is not installed. Please install it using 'pip install SimulationBenchmarks[nequip]'.")
        calc = NequIPCalculator.from_deployed_model(self.model_path)
        
        for i, atom in tqdm(enumerate(trajectory.frames), total=len(trajectory.frames), desc=f"Computing {self.name} properties"):
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