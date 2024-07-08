import os
import re
from typing import Union, List
from tqdm import tqdm
from ase.io import read
from ase.atoms import Atoms
from .base_calculator import PropertyCalculator

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

    def compute_properties(self, frames: 'Frames') -> None:
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

        sorted_outcar_files: List[str] = sorted(outcar_files, key=get_frame_number)
        processed_files = 0

        for filename in tqdm(sorted_outcar_files, desc=f"Computing {self.name} properties"):
            outcar_path: str = os.path.join(self.directory, filename)
            frame_number = get_frame_number(filename)
            
            if frame_number >= len(frames):
                print(f"Warning: Frame number {frame_number} from {filename} exceeds the number of input frames. Skipping.")
                continue

            try:
                outcar_frame: Atoms = read(outcar_path, index=self.index)
            except Exception as e:
                print(f"Error reading {self.base_name} file {filename}: {str(e)}. Skipping.")
                continue

            if self.has_energy:
                energy_key = f'{self.name}_total_energy'
                if energy_key in frames.frames[frame_number].info:
                    raise ValueError(f"{energy_key} already exists in frame {frame_number}")
                frames.frames[frame_number].info[energy_key] = outcar_frame.get_potential_energy()

            if self.has_forces:
                forces_key = f'{self.name}_forces'
                if forces_key in frames.frames[frame_number].arrays:
                    raise ValueError(f"{forces_key} already exists in frame {frame_number}")
                frames.frames[frame_number].arrays[forces_key] = outcar_frame.get_forces()

            if self.has_stress:
                stress_key = f'{self.name}_stress'
                if stress_key in frames.frames[frame_number].info:
                    raise ValueError(f"{stress_key} already exists in frame {frame_number}")
                frames.frames[frame_number].info[stress_key] = outcar_frame.get_stress()

            processed_files += 1
