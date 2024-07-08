from __future__ import annotations
import os
import re
from typing import Union, List
from tqdm import tqdm
from ase.io import read
from ase.atoms import Atoms
from .base_calculator import PropertyCalculator

# This name is HORRIBLE and should be changed ... please ... please change it
class VASPOUTCARDirectoryPropertyCalculator(PropertyCalculator):
    """
    Implementation of PropertyCalculator for VASP calculations (multiple OUTCAR files in a directory).

    This calculator processes multiple OUTCAR files in a specified directory, where each file
    corresponds to a frame in the simulation trajectory. The calculator extracts energy, forces,
    and stress information from the OUTCAR files and assigns them to the corresponding frames.

    Args:
        name (str): Name of the calculator (used as a prefix for property keys).
        directory (str): Path to the directory containing OUTCAR files.
        base_name (str): Base name used in OUTCAR filenames. Default is "OUTCAR".
        index (Union[int, str, slice]): Which frame(s) to read from each OUTCAR file. Default '-1' reads the last frame.
        has_energy (bool): Whether to extract and store energy information. Default is True.
        has_forces (bool): Whether to extract and store force information. Default is True.
        has_stress (bool): Whether to extract and store stress information. Default is True.

    Note:
        OUTCAR files should be named as: {base_name}_X where X is the frame number.
        For example: with default base_name, files should be named like OUTCAR_0, OUTCAR_1, OUTCAR_2, etc.
    """

    def __init__(self, name: str, directory: str, base_name: str = "OUTCAR", index: Union[int, str, slice] = '-1',
                 has_energy: bool = True, has_forces: bool = True, has_stress: bool = True):
        super().__init__(name, has_energy, has_forces, has_stress)
        self.directory: str = directory
        self.base_name: str = base_name
        self.index: Union[int, str, slice] = index

    def compute_properties(self, frames: 'Frames') -> None: # noqa F821
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        pattern = re.compile(f"{re.escape(self.base_name)}_(\d+)")
        outcar_files: List[str] = [f for f in os.listdir(self.directory) if pattern.match(f)]

        if not outcar_files:
            print(f"Warning: No {self.base_name} files found in {self.directory}")
            return

        def get_frame_number(filename: str) -> int:
            match = pattern.match(filename)
            return int(match.group(1)) if match else -1
        
        sorted_outcar_files = sorted(outcar_files, key=get_frame_number)

        # Apply the index to select directories
        if isinstance(self.index, int):
            sorted_outcar_files = [sorted_outcar_files[self.index]]
        elif isinstance(self.index, slice):
            sorted_outcar_files = sorted_outcar_files[self.index]
        # If it's a string ':' we keep all directories
        
        # Handle frame count mismatches
        if len(sorted_outcar_files) == 1:
            print("Warning: Only one frame found in OUTCAR. Repeating for all input frames.")
            sorted_outcar_files = [sorted_outcar_files[0]] * len(frames)
        elif len(sorted_outcar_files) < len(frames):
            raise ValueError(f"Not enough frames in OUTCAR ({len(sorted_outcar_files)}) to match input frames ({len(frames)})")
        elif len(sorted_outcar_files) > len(frames):
            print(f"Warning: More frames in OUTCAR ({len(sorted_outcar_files)}) than input frames ({len(frames)}). Using only the first {len(frames)} OUTCAR frames.")
            sorted_outcar_files = sorted_outcar_files[:len(frames)]

        for i, (filename, frame) in enumerate(tqdm(zip(sorted_outcar_files, frames.frames), total=len(frames), desc=f"Computing {self.name} properties")):
            
            outcar_path: str = os.path.join(self.directory, filename)

            try:
                outcar_frame: Union[Atoms, List[Atoms]] = read(outcar_path, format='vasp-out')
            except Exception as e:
                print(f"Error reading {outcar_path} OUTCAR file : {str(e)}. Skipping.")
                continue

            # if we get a list of Atoms, we proceed with the first structure
            if isinstance(outcar_frame, List):
                print(f"Error reading {outcar_path} OUTCAR file : Got a list of Atoms. Proceeding with the first structure from the OUTCAR")
                outcar_frame = outcar_frame[0]
                continue

            if self.has_energy:
                energy_key = f'{self.name}_total_energy'
                if energy_key in frame.info:
                    raise ValueError(f"{energy_key} already exists in frame {i}")
                frame.info[energy_key] = outcar_frame.get_potential_energy()

            if self.has_forces:
                forces_key = f'{self.name}_forces'
                if forces_key in frame.arrays:
                    raise ValueError(f"{forces_key} already exists in frame {i}")
                frame.arrays[forces_key] = outcar_frame.get_forces()

            if self.has_stress:
                stress_key = f'{self.name}_stress'
                if stress_key in frame.info:
                    raise ValueError(f"{stress_key} already exists in frame {i}")
                frame.info[stress_key] = outcar_frame.get_stress()

