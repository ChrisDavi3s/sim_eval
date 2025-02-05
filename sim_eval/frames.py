from collections import defaultdict
from typing import List, Optional, Union, Dict
import numpy as np
from ase.io import read
from scipy.stats import pearsonr
from .property import Property, PropertyMetric, MetricType
from .calculators import PropertyCalculator
from ase.atoms import Atoms
from ase.build.tools import sort
from .tools import normalize_frame_selection, calculate_von_mises_stress


class Frames:
    """
    This class is used to store and manage atomic structures for property calculations.
    It supports reading frames from files (using ASE) or directly from ASE Atoms objects.
    """
    def __init__(self,
                 file_path: Optional[str] = None,
                 format: Optional[str] = None,
                 index: Union[int, str] = ':',
                 sort_atoms_by_atomic_number: bool = False,
                 atoms: Union[Atoms, List[Atoms], None] = None):
        """
        Initialize Frames object from a file or ASE Atoms objects.

        Args:
            file_path (Optional[str]): Path to the file containing atomic structures.
                Required if atoms is not provided.
            format (Optional[str]): File format. If None, ASE will guess the format.
                Ignored if atoms is provided.
            index (Union[int, str]): Which frame(s) to read from the file.
                Default ':' reads all frames. Ignored if atoms is provided.
            sort_atoms_by_atomic_number (bool): Whether to sort atoms by atomic number.
            atoms (Union[Atoms, List[Atoms], None]): ASE Atoms object(s) to use directly.
                If provided, file_path is ignored.

        Raises:
            ValueError: If neither file_path nor atoms are provided, or if both are provided.
            ValueError: If no frames are loaded.
            TypeError: If atoms is not an ASE Atoms object or a list of them.
        """
        if (file_path is not None) and (atoms is not None):
            raise ValueError("Cannot specify both file_path and atoms; choose one.")
        if file_path is None and atoms is None:
            raise ValueError("Either file_path or atoms must be provided.")

        if atoms is not None:
            self.file_path = None
            if isinstance(atoms, Atoms):
                self.frames = [atoms]
            elif isinstance(atoms, list):
                if not all(isinstance(a, Atoms) for a in atoms):
                    raise TypeError("All elements in atoms list must be ASE Atoms objects.")
                self.frames = atoms.copy()
            else:
                raise TypeError("atoms must be an ASE Atoms object or a list of them.")
        else:
            self.file_path = file_path
            self.frames = read(self.file_path, index=index, format=format)
            if not isinstance(self.frames, list):
                self.frames = [self.frames]

        if not self.frames:
            raise ValueError("No frames were loaded.")

        if sort_atoms_by_atomic_number:
            for i, frame in enumerate(self.frames):
                self.frames[i] = sort(frame, tags=frame.get_atomic_numbers())

        

    def __len__(self) -> int:
        return len(self.frames)

    def get_number_of_atoms(self,
                            frame_number: Union[int,
                                                List[int],
                                                slice] = slice(None)
                            ) -> Union[int, List[int]]:
        """
        Get the number of atoms in specified frame(s).

        Args:
            frame_number (Union[int, List[int], slice], optional):
                        The frame number(s) to get atom counts for.
                        Defaults to all frames.

        Returns:
            Union[int, List[int]]: Number of atoms in the specified frame(s).

        Raises:
            ValueError: If the frame number input is invalid or out of bounds.
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        atom_counts = [len(self.frames[i]) for i in frames]
        return atom_counts[0] if len(atom_counts) == 1 else atom_counts

    def get_atom_types_and_indices(self, frame_number: Optional[Union[int, slice]] = None) -> Dict[str, List[int]]:
        """
        Get the atom types and their corresponding indices.

        Args:
            frame_number (Optional[Union[int, slice]]): The frame number(s) to consider. 
                If None, consider all frames. Defaults to None.

        Returns:
            Dict[str, List[int]]: Dictionary with atom types as keys and
            their indices as values.
        """
        # Precompute cumulative atoms for O(1) lookups
        cumulative_atoms = np.cumsum([0] + [len(f) for f in self.frames])

        atom_types: Dict[str, List[int]] = defaultdict(list)

        # Get original frame indices to process
        if frame_number is None:
            frame_indices = range(len(self.frames))
        elif isinstance(frame_number, int):
            if frame_number < 0 or frame_number >= len(self.frames):
                raise ValueError(f"Invalid frame number {frame_number}")
            frame_indices = [frame_number]
        elif isinstance(frame_number, slice):
            start, stop, step = frame_number.indices(len(self.frames))
            frame_indices = range(start, stop, step)
        else:
            raise ValueError("frame_number must be int, slice, or None")

        # Process selected frames
        for orig_frame_idx in frame_indices:
            frame = self.frames[orig_frame_idx]
            start_idx = cumulative_atoms[orig_frame_idx]
            
            for atom_local_idx, atom in enumerate(frame):
                global_idx = start_idx + atom_local_idx
                atom_types[atom.symbol].append(global_idx)

        return dict(atom_types)

    def add_method_data(self, method: 'PropertyCalculator') -> None:
        """
        Compute properties using the given method and add them to the frames.

        Args:
            method (PropertyCalculator): The calculator to use for
        """
        method.compute_properties(self)

    def get_property(self, property: Property,
                    calculator: 'PropertyCalculator',
                    frame_number: Union[int, List[int], slice] = slice(None)
                    ) -> np.ndarray:
        """
        Get a specific property for given frame(s) and calculator as a numpy array.

        Args:
            property (Property): The type of property to retrieve.
            calculator (PropertyCalculator): The ASE calculator used to compute the property.
            frame_number (Union[int, slice], optional): The frame number(s) to get the property for.
                                                        Defaults to all frames.

        Returns:
            np.ndarray: 
            - For ENERGY: 1D array of shape (n_frames,)
            - For FORCES: If all frames have the same number of atoms, 3D array of shape (n_frames, n_atoms, 3).
                        If atom counts vary, an object array of 2D arrays (n_frames,).
            - For STRESS: 2D array of shape (n_frames, 6)

        Raises:
            ValueError: If an invalid property type is provided or frame number is out of bounds.
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)

        property_map = {
            Property.ENERGY:
                lambda frame: frame.info[f'{calculator.name}_total_energy'],
            Property.FORCES:
                lambda frame: frame.arrays[f'{calculator.name}_forces'],
            Property.STRESS:
                lambda frame: frame.info[f'{calculator.name}_stress']
        }
        if property not in property_map:
            raise ValueError("Invalid property type")

        get_frame_property = property_map[property]

        if property == Property.FORCES:
            forces_list = [get_frame_property(self.frames[i]) for i in frames]
            try:
                # Attempt to create a 3D array (n_frames, n_atoms, 3)
                return np.array(forces_list)
            except ValueError:
                # Handle varying atom counts with object array
                arr = np.empty(len(forces_list), dtype=object)
                arr[:] = forces_list
                return arr
        else:
            return np.array([get_frame_property(self.frames[i]) for i in frames])
    
    def get_property_magnitude(
        self,
        property_metric: PropertyMetric,
        calculator: 'PropertyCalculator',
        frame_number: Union[int, List[int], slice] = slice(None)
    ) -> np.ndarray:
        """
        Calculate and return the magnitude of a specified property for selected frames.

        Args:
            property_metric (PropertyMetric): Contains both property type and metric type
            calculator (PropertyCalculator): Calculator used for property computation
            frame_number: Frame selection. Defaults to all frames.

        Returns:
            np.ndarray: Property magnitudes with shapes:
            - ENERGY:
                - PER_STRUCTURE: (n_frames,)
                - PER_ATOM: (n_frames,)
            - FORCES:
                - PER_STRUCTURE: (n_frames,)
                - PER_ATOM: (n_frames,) object array of (n_atoms,) arrays
            - STRESS:
                - PER_STRUCTURE: (n_frames,)
                - PER_ATOM: (n_frames,)

        Raises:
            ValueError: For unsupported property types
        """
        # Get normalized frame indices and raw property data
        frames = normalize_frame_selection(len(self.frames), frame_number)
        data = self.get_property(property_metric.property_type, calculator, frames)

        # Convert to numpy array for vector operations (except object arrays)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # ENERGY HANDLING
        if property_metric.property_type == Property.ENERGY:
            if property_metric.metric_type == MetricType.PER_ATOM:
                num_atoms = self.get_number_of_atoms(frames)
                # Changed: Remove reshape to avoid broadcasting to matrix
                data = data / np.array(num_atoms)  # Now element-wise division

        # FORCES HANDLING 
        elif property_metric.property_type == Property.FORCES:
            if property_metric.metric_type == MetricType.PER_STRUCTURE:
                # Sum forces per structure and compute magnitude
                sum_forces = [np.sum(frame, axis=0) for frame in data]
                data = np.array([np.linalg.norm(sf) for sf in sum_forces])
            else:  # PER_ATOM
                # Calculate magnitudes per atom, handle variable atom counts
                force_mags = [np.linalg.norm(frame, axis=-1) for frame in data]
                try:
                    data = np.array(force_mags)  # Uniform atom counts
                except ValueError:
                    data = np.empty(len(force_mags), dtype=object)
                    data[:] = force_mags  # Jagged array storage

        # STRESS HANDLING
        elif property_metric.property_type == Property.STRESS:
            data = calculate_von_mises_stress(data)
            if property_metric.metric_type == MetricType.PER_ATOM:
                num_atoms = self.get_number_of_atoms(frames)
                # Changed: Direct element-wise division without reshape
                data = data / np.array(num_atoms)  # Preserves 1D shape

        else:
            raise ValueError(f"Unsupported property type: {property_metric.property_type}")

        # Remove singleton dimensions while preserving object arrays
        return data.squeeze()

    def get_flattened_property_magnitude(self,
                                        property_metric: PropertyMetric,
                                        calculator: 'PropertyCalculator',
                                        frame_number: Union[int, List[int], slice] = slice(None)
                                        ) -> np.ndarray:
        """
        Get the flattened magnitude of a property for given frame(s) and calculator.

        Args:
            property_metric (PropertyMetric): The property and metric type to retrieve.
            calculator (PropertyCalculator): The calculator used to compute the property.
            frame_number (Union[int, List[int], slice], optional): The frame number(s) to get the property for.
                                                                Defaults to all frames.

        Returns:
            np.ndarray: Flattened 1D array of property magnitudes. For PER_ATOM forces with varying atoms,
                        this concatenates all per-atom values into a single 1D array.
        """
        data = self.get_property_magnitude(property_metric, calculator, frame_number)
        
        if isinstance(data, np.ndarray) and data.dtype == object:
            # Handle object arrays (e.g., PER_ATOM forces with varying atom counts)
            return np.concatenate(data.tolist())
        else:
            return data.ravel()
            
    def get_mae(self,
                property_metric: PropertyMetric,
                reference_calculator: 'PropertyCalculator',
                target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                frame_number: Union[int, List[int], slice] = slice(None),
                ) -> np.ndarray:
        """
        Calculate Mean Absolute Error (MAE) between reference and target calculators.

        Args:
            property_metric (PropertyMetric): The property and metric type for MAE calculation.
            reference_calculator (PropertyCalculator): Reference calculator providing true values.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): Target calculator(s) to evaluate.
            frame_number (Union[int, List[int], slice], optional): Frames to include. Defaults to all.

        Returns:
            np.ndarray: MAE values for each target calculator. Shape: (n_calculators,).
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        reference_data = self.get_flattened_property_magnitude(property_metric, reference_calculator, frames)
        target_calculators = target_calculator if isinstance(target_calculator, list) else [target_calculator]

        maes = []
        for calc in target_calculators:
            target_data = self.get_flattened_property_magnitude(property_metric, calc, frames)
            maes.append(np.mean(np.abs(reference_data - target_data)))

        return np.array(maes)

    def get_rmse(self, 
                property_metric: PropertyMetric,
                reference_calculator: 'PropertyCalculator',
                target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                frame_number: Union[int, List[int], slice] = slice(None)
                ) -> np.ndarray:
        """
        Calculate Root Mean Square Error (RMSE) between reference and target calculators.

        Args:
            property_metric (PropertyMetric): The property and metric type for RMSE calculation.
            reference_calculator (PropertyCalculator): Reference calculator providing true values.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): Target calculator(s) to evaluate.
            frame_number (Union[int, List[int], slice], optional): Frames to include. Defaults to all.

        Returns:
            np.ndarray: RMSE values for each target calculator. Shape: (n_calculators,).
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        reference_data = self.get_flattened_property_magnitude(property_metric, reference_calculator, frames)
        target_calculators = target_calculator if isinstance(target_calculator, list) else [target_calculator]

        rmses = []
        for calc in target_calculators:
            target_data = self.get_flattened_property_magnitude(property_metric, calc, frames)
            mse = np.mean((reference_data - target_data) ** 2)
            rmses.append(np.sqrt(mse))

        return np.array(rmses)

    def get_correlation(self,
                        property_metric: PropertyMetric,
                        reference_calculator: 'PropertyCalculator',
                        target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                        frame_number: Union[int, List[int], slice] = slice(None)
                        ) -> np.ndarray:
        """
        Calculate Pearson correlation between reference and target calculators.

        Args:
            property_metric (PropertyMetric): The property and metric type for correlation.
            reference_calculator (PropertyCalculator): Reference calculator providing true values.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): Target calculator(s) to evaluate.
            frame_number (Union[int, List[int], slice], optional): Frames to include. Defaults to all.

        Returns:
            np.ndarray: Pearson correlation coefficients. Shape: (n_calculators,).
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        reference_data = self.get_flattened_property_magnitude(property_metric, reference_calculator, frames)
        target_calculators = target_calculator if isinstance(target_calculator, list) else [target_calculator]

        correlations = []
        for calc in target_calculators:
            target_data = self.get_flattened_property_magnitude(property_metric, calc, frames)
            if reference_data.size < 2 or target_data.size < 2:
                correlations.append(np.nan)
            else:
                correlations.append(pearsonr(reference_data, target_data)[0])

        return np.array(correlations)