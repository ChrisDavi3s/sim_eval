from collections import defaultdict
from typing import List, Optional, Union, Dict
import numpy as np
from ase.io import read
from scipy.stats import pearsonr
from .property import Property, PropertyMetric, MetricType
from .calculators import PropertyCalculator
from ase.atoms import Atoms
from .tools import normalize_frame_selection, calculate_von_mises_stress


class Frames:
    """
    This class is used to store and manage atomic structures for property
    calculations. It supports reading frames from any format that ASE can read.
    """
    def __init__(self,
                 file_path: str,
                 format: Optional[str] = None,
                 index: Union[int, str] = ':'):
        """
        Initialize Frames object.

        Args:
            file_path (str): Path to the file containing atomic structures.
            format (Optional[str]): File format. If None, ASE will try to guess
            the format.
            index (Union[int, str]): Which frame(s) to read.
                                     Default ':' reads all frames.

        Raises:
            ValueError: If no frames are loaded.
        """
        self.file_path = file_path
        self.frames: List[Atoms] = read(self.file_path,
                                        index=index,
                                        format=format)
        if not self.frames:
            raise ValueError("No frames were loaded from the file.")
        if not isinstance(self.frames, list):
            self.frames = [self.frames]

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
        atom_types: Dict[str, List[int]] = defaultdict(list)

        if frame_number is None:
            frames_to_check = self.frames
        elif isinstance(frame_number, int):
            if frame_number < 0 or frame_number >= len(self.frames):
                raise ValueError(f"Invalid frame number {frame_number}")
            frames_to_check = [self.frames[frame_number]]
        elif isinstance(frame_number, slice):
            frames_to_check = self.frames[frame_number]
        else:
            raise ValueError("frame_number must be int, slice, or None")

        for frame_idx, frame in enumerate(frames_to_check):
            for atom_idx, atom in enumerate(frame):
                if isinstance(frame_number, slice):
                    global_idx = frame_idx * len(frame) + atom_idx
                else:
                    global_idx = atom_idx
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
                     ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get a specific property for given frame(s) and calculator.

        Args:
            property (Property): The type of property to retrieve.
            calculator (PropertyCalculator): The ASE calculator used to compute
                                             the property.
            frame_number (Union[int, slice], optional):
                                             The frame number(s) to
                                             get the property for.
                                             Defaults to all frames.

        Returns:
            Union[np.ndarray, List[np.ndarray]]:
            - For ENERGY: 1D array of shape (n_frames,)
            - For FORCES: 3D array of shape (n_frames, n_atoms, 3) if slice,
                          or 2D array of shape (n_atoms, 3) if int
            - For STRESS: 2D array of shape (n_frames, 6) if slice,
                          or 1D array of shape (6,) if int

        Raises:
            ValueError: If an invalid property type is provided or frame
                        number is out of bounds.
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
            # Return a list of arrays for forces, as they may have different shapes
            return [get_frame_property(self.frames[i]) for i in frames]
        else:
            # For energy and stress, we can still use np.array
            return np.array([get_frame_property(self.frames[i]) for i in frames])    

    def get_property_magnitude(self,
                               property: PropertyMetric,
                               calculator: 'PropertyCalculator',
                               frame_number: Union[int,
                                                   List[int],
                                                   slice] = slice(None)
                               ) -> np.ndarray:
        """
        Get the magnitude of a property for given frame(s) and calculator.

        Args:
            property (PropertyType): The type of property to retrieve.
            calculator (PropertyCalculator): The calculator used to compute
                                             the property.
            frame_number (Union[int, slice], optional):
                                    The frame number(s) to get the property for
                                    Defaults to all frames.

        Returns:
            np.ndarray:
            - For ENERGY: 1D array of shape (n_frames,)
            - For FORCES: 2D array of shape (n_frames, n_atoms) or (n_frames,)
            - For STRESS: 1D array of shape (n_frames,)
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        data = self.get_property(property.property_type, calculator, frames)

        if property.property_type == Property.ENERGY:
            if property.metric_type == MetricType.PER_ATOM:
                num_atoms = np.array(self.get_number_of_atoms(frames))
                data /= num_atoms

        elif property.property_type == Property.FORCES:
            # Calculate the magnitude of the force for each atom
            data = [np.linalg.norm(frame_data, axis=-1) for frame_data in data]
            if property.metric_type == MetricType.PER_STRUCTURE:
                # Sum over atoms for each frame
                data = np.array([np.sum(frame_data) for frame_data in data])
            # For PER_ATOM, keep it as a list of arrays

        elif property.property_type == Property.STRESS:
            data = calculate_von_mises_stress(data)

            if property.metric_type == MetricType.PER_ATOM:
                num_atoms = np.array(self.get_number_of_atoms(frames))
                data /= num_atoms

        else:
            raise ValueError("Invalid property type")

        return data
    
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
            np.ndarray: Flattened 1D array of property magnitudes.
        """
        data = self.get_property_magnitude(property_metric, calculator, frame_number)
        
        if isinstance(data, list):  # For forces with variable atom counts
            return np.concatenate(data)
        else:
            return data.flatten()
    
    def get_mae(self,
                property_metric: PropertyMetric,
                reference_calculator: 'PropertyCalculator',
                target_calculator: Union['PropertyCalculator',
                                         List['PropertyCalculator']],
                frame_number: Union[int, List[int], slice] = slice(None),
                ) -> np.ndarray:
        """
        Calculate Mean Absolute Error (MAE) across specified frames.

        Args:
            property_metric (PropertyMetric):
                            The property and metric type to calculate MAE for.
            reference_calculator (PropertyCalculator):
                            The reference calculator.
            target_calculator (Union[PropertyCalculator,
                               List[PropertyCalculator]]):
                            The target calculator(s).
            frame_number (Union[int, List[int], slice], optional):
                            The frame number(s) to calculate MAE for.
                            Defaults to all frames.

        Returns:
            np.ndarray: MAE values.
            The shape depends on the property_metric and number of calculators:
            - For ENERGY and STRESS: (n_calculators,)
            - For FORCES:
                - PER_STRUCTURE: (n_calculators,)
                - PER_ATOM: (n_calculators, n_atoms)
            Where n_calculators is 1 if a single calculator is provided.

        Raises:
            ValueError: If an invalid property_metric is provided.
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        reference_data = self.get_flattened_property_magnitude(property_metric, reference_calculator, frames)

        if not isinstance(target_calculator, list):
            target_calculators = [target_calculator]
        else:
            target_calculators = target_calculator

        maes = []
        for calc in target_calculators:
            target_data = self.get_flattened_property_magnitude(property_metric, calc, frames)
            
            if isinstance(reference_data, list):  # For forces with variable atom counts
                mae = np.mean([np.mean(np.abs(ref - tar)) for ref, tar in zip(reference_data, target_data)])
            else:
                mae = np.mean(np.abs(reference_data - target_data))
            
            maes.append(mae)

        return np.array(maes)
    
    def get_rmse(self, 
                property_metric: PropertyMetric,
                reference_calculator: 'PropertyCalculator',
                target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                frame_number: Union[int, List[int], slice] = slice(None)
                ) -> np.ndarray:
        """
        Calculate Root Mean Square Error (RMSE) across specified frames.

        Args:
            property_metric (PropertyMetric): The property and metric type to calculate RMSE for.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, List[int], slice], optional): The frame number(s) to calculate RMSE for.
                                                                    Defaults to all frames.

        Returns:
            np.ndarray: RMSE values. The shape depends on the property_metric and number of calculators:
            - For ENERGY and STRESS: (n_calculators,)
            - For FORCES: (n_calculators,)
            Where n_calculators is 1 if a single calculator is provided.

        Raises:
            ValueError: If an invalid property_metric is provided.
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        reference_data = self.get_flattened_property_magnitude(property_metric, reference_calculator, frames)

        if not isinstance(target_calculator, list):
            target_calculators = [target_calculator]
        else:
            target_calculators = target_calculator

        rmses = []
        for calc in target_calculators:
            target_data = self.get_flattened_property_magnitude(property_metric, calc, frames)
            
            if isinstance(reference_data, list):  # For forces with variable atom counts
                mse = np.mean([np.mean((ref - tar) ** 2) for ref, tar in zip(reference_data, target_data)])
                rmse = np.sqrt(mse)
            else:
                rmse = np.sqrt(np.mean((reference_data - target_data) ** 2))
            
            rmses.append(rmse)

        return np.array(rmses)

    def get_correlation(self,
                        property_metric: PropertyMetric,
                        reference_calculator: 'PropertyCalculator',
                        target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                        frame_number: Union[int, List[int], slice] = slice(None)
                        ) -> np.ndarray:
        """
        Calculate the Pearson correlation coefficient for a specified property across frames.

        Args:
            property_metric (PropertyMetric): The property and metric type for which to calculate the correlation.
            reference_calculator (PropertyCalculator): The calculator providing reference property values.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): The calculator(s) providing target property values.
            frame_number (Union[int, List[int], slice], optional): The frame number(s) to calculate correlation for.
                                                                    Defaults to all frames.

        Returns:
            np.ndarray: An array of correlation values. The shape is (n_calculators,),
            where n_calculators is 1 if a single calculator is provided.

        Raises:
            ValueError: If an invalid property_metric is provided.
        """
        frames = normalize_frame_selection(len(self.frames), frame_number)
        reference_data = self.get_flattened_property_magnitude(property_metric, reference_calculator, frames)
        target_calculators = target_calculator if isinstance(target_calculator, list) else [target_calculator]

        correlations = []

        for calc in target_calculators:
            target_data = self.get_flattened_property_magnitude(property_metric, calc, frames)

            if reference_data.size < 2 or target_data.size < 2:
                correlation = np.nan
            else:
                correlation = pearsonr(reference_data, target_data)[0]

            correlations.append(correlation)

        return np.array(correlations)