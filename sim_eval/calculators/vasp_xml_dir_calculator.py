import os
import re
from typing import Union, List
from tqdm import tqdm
from ase.io import read
from ase.atoms import Atoms
from .base_calculator import PropertyCalculator

class VASPXMLDiretoryPropertyCalculator(PropertyCalculator):
    """
    Implementation of PropertyCalculator for VASP calculations using XML files in a single directory.

    This calculator processes VASP XML files (vasprun_frame_X.xml) located in a specified directory.
    Each XML file corresponds to a frame in the simulation trajectory.

    Args:
        name (str): Name of the calculator (used as a prefix for property keys).
        directory (str): Path to the directory containing vasprun_frame_X.xml files.
        base_name (str): Base name used for file matching. Files should be named as "{base_name}_X.xml".
        index (Union[int, slice, str]): Specifies which XML files to process. Default ':' processes all.
            - int: A specific file index.
            - slice: A slice object for selecting a range of files.
            - str: ':' for all files, or a slice string like '::2' for every second file.
        has_energy (bool): Whether to extract and store energy information. Default is True.
        has_forces (bool): Whether to extract and store force information. Default is True.
        has_stress (bool): Whether to extract and store stress information. Default is True.

    Attributes:
        directory (str): Path to the directory containing XML files.
        base_name (str): Base name for file matching.
        index (Union[int, slice, str]): File selection index.

    Note:
        - XML files should be named as "{base_name}_X.xml" where X is a number.
        - Files are processed in numerical order based on their names.
        - Files that don't match the expected naming pattern will be skipped with a warning.
    """

    def __init__(self, name:
                  str, directory: str,
                  base_name: str, 
                  index: Union[int, slice, str] = ':',
                 has_energy: bool = True, has_forces: bool = True, has_stress: bool = True):
        super().__init__(name, has_energy, has_forces, has_stress)
        self.directory: str = directory
        self.base_name: str = base_name
        self.index: Union[int, slice, str] = index

    def compute_properties(self, frames: 'Frames') -> None:  # noqa F821
        pattern = re.compile(f"{self.base_name}_(\\d+)\\.xml")
        vasp_files = [f for f in os.listdir(self.directory) if pattern.match(f)]        
        
        def get_frame_number(filename):
            match = pattern.match(filename)
            return int(match.group(1)) if match else -1
        
        sorted_vasp_files = sorted(vasp_files, key=get_frame_number)

        # Apply the index to select directories
        if isinstance(self.index, int):
            sorted_vasp_files = [sorted_vasp_files[self.index]]
        elif isinstance(self.index, slice):
            sorted_vasp_files = sorted_vasp_files[self.index]
        # If it's a string ':' we keep all directories

        if len(sorted_vasp_files) == 1:
            print("Warning: Only one frame found in OUTCAR. Repeating for all input frames.")
            sorted_vasp_files = [sorted_vasp_files[0]] * len(frames)
        elif len(sorted_vasp_files) < len(frames):
            raise ValueError(f"Not enough frames in OUTCAR ({len(sorted_vasp_files)}) to match input frames ({len(frames)})")
        elif len(sorted_vasp_files) > len(frames):
            print(f"Warning: More frames in OUTCAR ({len(sorted_vasp_files)}) than input frames ({len(frames)}). Using only the first {len(frames)} OUTCAR frames.")
            sorted_vasp_files = sorted_vasp_files[:len(frames)]


        for i, (filename, frame) in enumerate(tqdm(zip(sorted_vasp_files, frames.frames), total=len(frames), desc=f"Computing {self.name} properties")):
            
            file = os.path.join(self.directory, filename)

            try:
                vasp_atoms:  Union[Atoms, List[Atoms]] = read(file)
            except Exception as e:
                print(f"Error reading XML file {filename}: {str(e)}. Skipping.")
                continue
            
            # if we get a list of Atoms, we proceed with the first structure
            if isinstance(vasp_atoms, List):
                print(f"Error reading {filename} XML file : Got a list of Atoms. Proceeding with the first structure from the filename")
                filename = filename[0]
                continue


            if self.has_energy:
                energy_key = f'{self.name}_total_energy'
                if energy_key in frame.info:
                    raise ValueError(f"{energy_key} already exists in frame {i}")
                frame.info[energy_key] = vasp_atoms.get_potential_energy()

            if self.has_forces:
                forces_key = f'{self.name}_forces'
                if forces_key in frame.arrays:
                    raise ValueError(f"{forces_key} already exists in frame {i}")
                frame.arrays[forces_key] = vasp_atoms.get_forces()

            if self.has_stress:
                stress_key = f'{self.name}_stress'
                if stress_key in frame.info:
                    raise ValueError(f"{stress_key} already exists in frame {i}")
                frame.info[stress_key] = vasp_atoms.get_stress()
