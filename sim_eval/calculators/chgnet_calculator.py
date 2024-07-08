from tqdm import tqdm
from chgnet.model import CHGNetCalculator
from ..frames import Frames
from .base_calculator import PropertyCalculator

class CHGNetPropertyCalculator(PropertyCalculator):
    """
    Implementation of PropertyCalculator for CHGNet models
    """
    def __init__(self, name, model_path=None, has_energy=True, has_forces=True, has_stress=True):
        if CHGNetCalculator is None:
            raise ImportError("CHGNet is not installed. Please install it using 'pip install chgnet'.")
        super().__init__(name, has_energy, has_forces, has_stress)
        self.model_path = model_path

    def compute_properties(self, frames: Frames):
        if CHGNetCalculator is None:
            raise ImportError("CHGNet is not installed. Please install it using 'pip install chgnet'.")
        
        # If model_path is provided, load a specific model. Otherwise, use the default model.
        if self.model_path:
            calc = CHGNetCalculator.from_file(model_path=self.model_path)
        else:
            calc = CHGNetCalculator()

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