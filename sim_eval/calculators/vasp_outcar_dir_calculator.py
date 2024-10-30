import os
import re
from typing import Union, List
from tqdm import tqdm
from ase.io import read
from ase.atoms import Atoms
from .base_calculator import PropertyCalculator

class VASPOUTCARDirectoryPropertyCalculator(PropertyCalculator):
  """
  Implementation of PropertyCalculator for VASP calculations using OUTCAR files in a single directory.

  This calculator processes VASP OUTCAR files (OUTCAR_X) located in a specified directory.
  Each OUTCAR file corresponds to a frame in the simulation trajectory.

  Args:
      name (str): Name of the calculator (used as a prefix for property keys).
      directory (str): Path to the directory containing OUTCAR_X files.
      index (Union[int, slice, str]): Specifies which OUTCAR files to process. Default ':' processes all.
          - int: A specific file index.
          - slice: A slice object for selecting a range of files.
          - str: ':' for all files, or a slice string like '::2' for every second file.
      has_energy (bool): Whether to extract and store energy information. Default is True.
      has_forces (bool): Whether to extract and store force information. Default is True.
      has_stress (bool): Whether to extract and store stress information. Default is True.

  Attributes:
      directory (str): Path to the directory containing OUTCAR files.
      index (Union[int, slice, str]): File selection index.

  Note:
      - OUTCAR files should be named as "OUTCAR_X" where X is a number.
      - Files are processed in numerical order based on their names.
      - Files that don't match the expected naming pattern will be skipped with a warning.
  """

  def __init__(self, name: str, directory: str, 
               index: Union[int, slice, str] = ':',
               has_energy: bool = True, has_forces: bool = True, has_stress: bool = True):
      super().__init__(name, has_energy, has_forces, has_stress)
      self.directory: str = directory
      self.index: Union[int, slice, str] = index

  def compute_properties(self, frames: 'Frames') -> None:  # noqa F821
      pattern = re.compile(r"OUTCAR_(\d+)")
      outcar_files = [f for f in os.listdir(self.directory) if pattern.match(f)]        
      
      def get_frame_number(filename):
          match = pattern.match(filename)
          return int(match.group(1)) if match else -1
      
      sorted_outcar_files = sorted(outcar_files, key=get_frame_number)

      # Apply the index to select files
      if isinstance(self.index, int):
          sorted_outcar_files = [sorted_outcar_files[self.index]]
      elif isinstance(self.index, slice):
          sorted_outcar_files = sorted_outcar_files[self.index]
      # If it's a string ':' we keep all files

      if len(sorted_outcar_files) == 1:
          print("Warning: Only one OUTCAR file found. Repeating for all input frames.")
          sorted_outcar_files = [sorted_outcar_files[0]] * len(frames)
      elif len(sorted_outcar_files) < len(frames):
          raise ValueError(f"Not enough OUTCAR files ({len(sorted_outcar_files)}) to match input frames ({len(frames)})")
      elif len(sorted_outcar_files) > len(frames):
          print(f"Warning: More OUTCAR files ({len(sorted_outcar_files)}) than input frames ({len(frames)}). Using only the first {len(frames)} OUTCAR files.")
          sorted_outcar_files = sorted_outcar_files[:len(frames)]

      for i, (filename, frame) in enumerate(tqdm(zip(sorted_outcar_files, frames.frames), total=len(frames), desc=f"Computing {self.name} properties")):
          
          file_path = os.path.join(self.directory, filename)

          try:
              vasp_atoms: Atoms = read(file_path, format='vasp-out')
          except Exception as e:
              print(f"Error reading OUTCAR file {filename}: {str(e)}. Skipping.")
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