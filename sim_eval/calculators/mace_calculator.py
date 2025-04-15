from tqdm import tqdm
try:
    from mace.calculators import MACECalculator
except ImportError:
    MACECalculator = None
from .base_calculator import PropertyCalculator

class MACEPropertyCalculator(PropertyCalculator):
    """
    Implementation of PropertyCalculator for MACE models    
    """
    def __init__(self, name, has_energy=True, has_forces=True, has_stress=True, **kwargs):
        """
        Initialize a MACEPropertyCalculator.

        Args:
            name (str): The name of the calculator.
            has_energy (bool, optional): Whether the calculator can compute energy. Defaults to True.
            has_forces (bool, optional): Whether the calculator can compute forces. Defaults to True.
            has_stress (bool, optional): Whether the calculator can compute stress. Defaults to True.
            **kwargs: Additional keyword arguments for specific calculators.
                model_path (str, optional): Path to the MACE model file. Defaults to None (Ie the default model).
                device (str, optional): Device to use for computation. Defaults to MACE default.
                dtype (str, optional): Data type to use for computation. Defaults to MACE default.
        """

        if MACECalculator is None:
            raise ImportError("MACE is not installed. Please install it using 'pip install mace-torch'.")
        super().__init__(name, has_energy, has_forces, has_stress)
        self.model_path = kwargs.get('model_path', None)
        self.device = kwargs.get('device', None)
        self.dtype = kwargs.get('dtype', "")
        if not self.model_path:
            raise ValueError("A model path must be provided for MACEPropertyCalculator.")

    def compute_properties(self, frames):
        if MACECalculator is None:
            raise ImportError("MACE is not installed. Please install it using 'pip install mace-torch'.")
        
        calc = MACECalculator(model_path=self.model_path, device=self.device, default_dtype=self.dtype)

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