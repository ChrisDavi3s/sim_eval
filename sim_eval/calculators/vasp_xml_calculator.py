# simulation_benchmarks/calculators/vasp_calculator.py
import os
import re
from tqdm import tqdm
from ase.io import read
from .base_calculator import PropertyCalculator

class VASPXMLPropertyCalculator(PropertyCalculator):
    '''
    Implementation of PropertyCalculator for VASP calculations (VASP XML files)
    '''
    def __init__(self, name, directory, base_name, has_energy=True, has_forces=True, has_stress=True):
        super().__init__(name, has_energy, has_forces, has_stress)
        self.directory = directory
        self.base_name = base_name

    def compute_properties(self, frames):            
        pattern = re.compile(f"{self.base_name}_(\\d+)\\.xml")
        vasp_files = [f for f in os.listdir(self.directory) if pattern.match(f)]
        
        def get_frame_number(filename):
            match = pattern.match(filename)
            return int(match.group(1)) if match else -1
        
        sorted_vasp_files = sorted(vasp_files, key=get_frame_number)
        
        for filename in tqdm(sorted_vasp_files, desc=f"Computing {self.name} properties"):
            vasp_atom = read(os.path.join(self.directory, filename))
            frame_number = get_frame_number(filename)
            
            if self.has_energy:
                if f'{self.name}_total_energy' in frames.frames[frame_number].info:
                    raise ValueError(f"{self.name}_total_energy already exists in frame {frame_number}")
                frames.frames[frame_number].info[f'{self.name}_total_energy'] = vasp_atom.get_potential_energy()
            if self.has_forces:
                if f'{self.name}_forces' in frames.frames[frame_number].arrays:
                    raise ValueError(f"{self.name}_forces already exists in frame {frame_number}")
                frames.frames[frame_number].arrays[f'{self.name}_forces'] = vasp_atom.get_forces()
            if self.has_stress:
                if f'{self.name}_stress' in frames.frames[frame_number].info:
                    raise ValueError(f"{self.name}_stress already exists in frame {frame_number}")
                frames.frames[frame_number].info[f'{self.name}_stress'] = vasp_atom.get_stress()