import os
from typing import Union, List
from ase.io import read
from ase.atoms import Atoms
from tqdm import tqdm
from .base_calculator import PropertyCalculator

class VASPOUTCARPropertyCalculator(PropertyCalculator):
    """
    Implementation of PropertyCalculator for VASP calculations (single OUTCAR file).

    This calculator processes a single OUTCAR file from a VASP calculation, which may contain
    multiple frames. It extracts energy, forces, and stress information from the OUTCAR file
    and assigns them to the corresponding frames in the input Frames object.

    Args:
        name (str): Name of the calculator (used as a prefix for property keys).
        outcar_path (str): Path to the OUTCAR file.
        index (Union[int, str, slice]): Which frame(s) to read from the OUTCAR file.
            Default ':' reads all frames. Can be an integer for a specific frame,
            or a slice object for a range of frames.
        has_energy (bool): Whether to extract and store energy information. Default is True.
        has_forces (bool): Whether to extract and store force information. Default is True.
        has_stress (bool): Whether to extract and store stress information. Default is True.

    Attributes:
        outcar_path (str): Path to the OUTCAR file.
        index (Union[int, str, slice]): Frame selection index for the OUTCAR file.

    Raises:
        FileNotFoundError: If the specified OUTCAR file is not found.
        ValueError: If no frames are found in the OUTCAR file or if there's an error reading the file.

    Note:
        This calculator can handle OUTCAR files with single or multiple frames. It will attempt
        to match the number of frames in the OUTCAR file with the number of input frames,
        raising warnings or errors if there's a mismatch.
    """

    def __init__(self, name: str, outcar_path: str, index: Union[int, str, slice] = ':',
                 has_energy: bool = True, has_forces: bool = True, has_stress: bool = True):
        super().__init__(name, has_energy, has_forces, has_stress)
        self.outcar_path: str = outcar_path
        self.index: Union[int, str, slice] = index

    def compute_properties(self, frames: 'Frames') -> None:
        """
        Compute properties for the given frames using the VASP OUTCAR file.

        This method reads the specified OUTCAR file, extracts the relevant properties
        (energy, forces, stress), and assigns them to the corresponding frames in the
        input Frames object.

        Args:
            frames (Frames): The Frames object to which properties will be added.

        Raises:
            FileNotFoundError: If the OUTCAR file is not found at the specified path.
            ValueError: If no frames are found in the OUTCAR file, if there's an error
                        reading the file, or if a property already exists in a frame.

        Note:
            - If the number of frames in the OUTCAR file doesn't match the number of
              input frames, the method will adjust by either repeating the single OUTCAR
              frame, truncating excess OUTCAR frames, or raising an error if there are
              not enough OUTCAR frames.
            - The method uses tqdm to display a progress bar during property computation.
        """
        if not os.path.exists(self.outcar_path):
            raise FileNotFoundError(f"OUTCAR file not found at {self.outcar_path}")

        # Read the OUTCAR file
        try:
            outcar_frames: Union[Atoms, List[Atoms]] = read(self.outcar_path, index=self.index, format='vasp-out')
        except Exception as e:
            raise ValueError(f"Error reading OUTCAR file: {str(e)}")

        if not outcar_frames:
            raise ValueError(f"No frames found in OUTCAR file at {self.outcar_path}")

        # Ensure outcar_frames is a list
        if not isinstance(outcar_frames, list):
            outcar_frames = [outcar_frames]

        # Handle frame count mismatches
        if len(outcar_frames) == 1:
            outcar_frames = [outcar_frames[0]] * len(frames)
        elif len(outcar_frames) < len(frames):
            raise ValueError(f"Not enough frames in OUTCAR ({len(outcar_frames)}) to match input frames ({len(frames)})")
        elif len(outcar_frames) > len(frames):
            print(f"Warning: More frames in OUTCAR ({len(outcar_frames)}) than input frames ({len(frames)}). Using only the first {len(frames)} OUTCAR frames.")
            outcar_frames = outcar_frames[:len(frames)]

        # Compute properties for each frame
        for i, (outcar_frame, frame) in enumerate(tqdm(zip(outcar_frames, frames), total=len(frames), desc=f"Computing {self.name} properties")):
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