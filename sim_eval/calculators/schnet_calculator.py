from __future__ import annotations
from tqdm import tqdm

try:
    import torch.serialization
    torch.serialization.add_safe_globals([slice])
    import torch # Schnetpack requires torch
    import schnetpack as spk
    from schnetpack.interfaces import SpkCalculator
    from schnetpack.transform import ASENeighborList
    SCHNETPACK_INSTALLED = True
except ImportError:
    # Set flags if Schnetpack is not installed
    SCHNETPACK_INSTALLED = False
    SpkCalculator = None
    ASENeighborList = None
    spk = None # Define spk as None for type checking if needed

from .base_calculator import PropertyCalculator

class SchnetPropertyCalculator(PropertyCalculator):
    """
    Implementation of PropertyCalculator for SchNetPack models.

    Wraps the schnetpack.interfaces.SpkCalculator.
    """
    def __init__(
        self,
        name: str = "schnet",
        has_energy: bool = True,
        has_forces: bool = True,
        has_stress: bool = False, # Default to False as not all models compute stress
        **kwargs
    ):
        """
        Initialize a SchnetPropertyCalculator.

        Args:
            name (str, optional): The name of the calculator. Defaults to "schnet".
            has_energy (bool, optional): Whether the calculator should compute energy. Defaults to True.
            has_forces (bool, optional): Whether the calculator should compute forces. Defaults to True.
            has_stress (bool, optional): Whether the calculator should compute stress.
                                         Set to True only if the SchNetPack model predicts stress
                                         and 'model_stress_key' is provided. Defaults to False.
            **kwargs: Keyword arguments passed directly or indirectly to schnetpack.interfaces.SpkCalculator.
                Required Kwargs:
                    model_path (str): Path to the trained SchNetPack model directory or file (.pt).
                    cutoff_radius (float): Cutoff radius used during model training (in Angstrom).
                Optional Kwargs (with defaults):
                    device (str): Device to run calculations on ('cpu', 'cuda', etc.). Defaults to 'cpu'.
                    dtype (torch.dtype): Data type for model calculations. Defaults to torch.float32.
                    model_energy_unit (str): Energy unit used by the model internally. Defaults to 'eV'.
                    model_position_unit (str): Position unit used by the model internally. Defaults to 'Ang'.
                    model_energy_key (str): Key for energy output in the model dictionary. Defaults to schnetpack.properties.energy (usually 'energy').
                    model_force_key (str): Key for force output in the model dictionary. Defaults to schnetpack.properties.forces (usually 'forces').
                    model_stress_key (str | None): Key for stress output in the model dictionary.
                                                  Required if has_stress=True. Defaults to None.

        Raises:
            ImportError: If SchNetPack is not installed.
            ValueError: If required keyword arguments (model_path, cutoff_radius) are missing,
                        or if has_stress=True but model_stress_key is not provided.
        """
        if not SCHNETPACK_INSTALLED:
            raise ImportError("SchNetPack is not installed. Please install it using 'pip install schnetpack'.")

        super().__init__(name, has_energy, has_forces, has_stress)

        # --- Extract SchNetPack specific parameters ---
        self.model_path = kwargs.get('model_path', None)
        self.cutoff_radius = kwargs.get('cutoff_radius', None)
        self.device = kwargs.get('device', 'cpu')
        self.dtype = kwargs.get('dtype', torch.float32)

        # Get defaults directly from schnetpack if possible, otherwise use common strings
        default_energy_key = getattr(spk.properties, 'energy', 'energy')
        default_force_key = getattr(spk.properties, 'forces', 'forces')
        default_stress_key = getattr(spk.properties, 'stress', None) # Some versions might have this

        self.model_energy_unit = kwargs.get('model_energy_unit', 'eV')
        self.model_position_unit = kwargs.get('model_position_unit', 'Ang')
        self.model_energy_key = kwargs.get('model_energy_key', default_energy_key)
        self.model_force_key = kwargs.get('model_force_key', default_force_key)
        self.model_stress_key = kwargs.get('model_stress_key', default_stress_key) # Use provided or schnetpack default

        # --- Validate required parameters ---
        if not self.model_path:
            raise ValueError("A 'model_path' must be provided for SchnetPropertyCalculator.")
        if self.cutoff_radius is None: # Check for None explicitly as 0.0 could be valid (though unlikely)
            raise ValueError("A 'cutoff_radius' must be provided for SchnetPropertyCalculator.")
        if self.has_stress and not self.model_stress_key:
            raise ValueError("If has_stress=True, a 'model_stress_key' must be provided.")
        if not self.has_stress:
             # Ensure stress key is None if stress is not calculated
             self.model_stress_key = None

    def compute_properties(self, frames):
        """
        Compute properties (energy, forces, stress) for the given frames using SchNetPack.

        Adds the computed properties to the ASE Atoms objects within the frames container.
        Properties are stored under keys like '{self.name}_total_energy',
        '{self.name}_forces', '{self.name}_stress'.

        Args:
            frames: An object containing a list-like attribute `frames` where each element
                    is an `ase.atoms.Atoms` object.

        Raises:
            ImportError: If SchNetPack is not installed (should be caught in __init__, but check again).
            ValueError: If a property key already exists in an Atoms object before calculation.
            FileNotFoundError: If the model_path is invalid.
            Exception: For errors during SpkCalculator initialization or calculation.
        """
        if not SCHNETPACK_INSTALLED:
             # This check is technically redundant if __init__ succeeded, but safe
            raise ImportError("SchNetPack is not installed. Please install it using 'pip install schnetpack'.")

        try:
            # Create neighbor list transform - required by SpkCalculator
            neighbor_list = ASENeighborList(cutoff=self.cutoff_radius)

            # Create the SchNetPack Calculator instance
            # Pass stress_key only if needed to avoid potential errors with models not predicting it
            calc = SpkCalculator(
                model_file=self.model_path,
                neighbor_list=neighbor_list,
                energy_key=self.model_energy_key if self.has_energy else None,
                force_key=self.model_force_key if self.has_forces else None,
                stress_key=self.model_stress_key if self.has_stress else None,
                energy_unit=self.model_energy_unit,
                position_unit=self.model_position_unit,
                device=self.device,
                dtype=self.dtype,
            )

        except Exception as e:
            print(f"ERROR: Failed to initialize SpkCalculator for '{self.name}': {e}")
            raise

        # --- Loop through frames and compute properties ---
        property_label = f"{self.name} properties"
        for i, atom in tqdm(enumerate(frames.frames), total=len(frames.frames), desc=f"Computing {property_label}"):
            # Attach the calculator to the current Atoms object
            # This calculator instance is reused for efficiency
            atom.calc = calc

            try:
                # Compute Energy
                if self.has_energy:
                    prop_key = f'{self.name}_total_energy'
                    if prop_key in atom.info:
                        raise ValueError(f"Property '{prop_key}' already exists in frame {i}. Aborting to prevent overwrite.")
                    # This triggers the calculation if energy hasn't been computed yet for this atom with this calculator
                    energy = atom.get_potential_energy()
                    atom.info[prop_key] = energy

                # Compute Forces
                if self.has_forces:
                    prop_key = f'{self.name}_forces'
                    if prop_key in atom.arrays:
                         raise ValueError(f"Property '{prop_key}' already exists in frame {i}. Aborting to prevent overwrite.")
                    # This triggers the calculation if forces haven't been computed yet
                    forces = atom.get_forces()
                    atom.arrays[prop_key] = forces

                # Compute Stress
                if self.has_stress:
                    prop_key = f'{self.name}_stress'
                    if prop_key in atom.info:
                         raise ValueError(f"Property '{prop_key}' already exists in frame {i}. Aborting to prevent overwrite.")
                     # This triggers the calculation if stress hasn't been computed yet
                    stress = atom.get_stress(voigt=False) # Get full 3x3 tensor, SpkCalculator handles units
                    atom.info[prop_key] = stress
            finally:
                # Detach calculator to prevent potential issues if the atom object is used elsewhere
                # or pickled without the calculator being properly serializable.
                # Optional, but can be good practice.
                atom.calc = None

