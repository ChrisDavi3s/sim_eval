from collections import defaultdict
from typing import List, Optional, Union, Dict
import numpy as np
from ase.io import read
from scipy.stats import pearsonr
from .property import Property
from .calculators import PropertyCalculator
from ase.atoms import Atoms


class Frames:
    """
    This class is used to store and manage atomic structures for property calculations.
    It supports reading frames from any format that ASE can read.
    """
    def __init__(self, file_path: str, format: Optional[str] = None, index: Union[int, str] = ':'):
        """
        Initialize Frames object.

        Args:
            file_path (str): Path to the file containing atomic structures.
            format (Optional[str]): File format. If None, ASE will try to guess the format.
            index (Union[int, str]): Which frame(s) to read. Default ':' reads all frames.

        Raises:
            ValueError: If no frames are loaded.
        """
        self.file_path = file_path
        self.frames: List[Atoms] = read(self.file_path, index=index, format=format)
        if not self.frames:
            raise ValueError("No frames were loaded from the file.")
        if not isinstance(self.frames, list):
            self.frames = [self.frames]

    def __len__(self) -> int:
        return len(self.frames)
    
    def get_number_of_atoms(self) -> int:
        """Get the number of atoms in each frame."""
        return len(self.frames[0])
    
    def get_atom_types_and_indices(self) -> Dict[str, List[int]]:
        """
        Get the atom types and their corresponding indices.

        Returns:
            Dict[str, List[int]]: Dictionary with atom types as keys and their indices as values.
        """
        atom_types: Dict[str, List[int]] = defaultdict(list)
        for i, atom in enumerate(self.frames[0]):
            atom_types[atom.symbol].append(i)
        return dict(atom_types)

    def add_method_data(self, method: 'PropertyCalculator') -> None:
        """
        Compute properties using the given method and add them to the frames.

        Args:
            method (PropertyCalculator): The calculator to use for property computation.
        """
        method.compute_properties(self)

    def get_property(self, property: Property, 
                     calculator: 'PropertyCalculator', 
                     frame_number: Union[int, slice] = slice(None)) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get a specific property for given frame(s) and calculator.

        Args:
            property (Property): The type of property to retrieve.
            calculator (PropertyCalculator): The calculator used to compute the property.
            frame_number (Union[int, slice], optional): The frame number(s) to get the property for. Defaults to all frames.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: 
            - For ENERGY: 1D array of shape (n_frames,)
            - For FORCES: 3D array of shape (n_frames, n_atoms, 3) if slice, or 2D array of shape (n_atoms, 3) if int
            - For STRESS: 2D array of shape (n_frames, 6) if slice, or 1D array of shape (6,) if int

        Raises:
            ValueError: If an invalid property type is provided or frame number is out of bounds.
        """
        property_map = {
            Property.ENERGY: lambda frame: frame.info[f'{calculator.name}_total_energy'],
            Property.FORCES: lambda frame: frame.arrays[f'{calculator.name}_forces'],
            Property.STRESS: lambda frame: frame.info[f'{calculator.name}_stress']
        }
        if property not in property_map:
            raise ValueError("Invalid property type")

        get_frame_property = property_map[property]

        self._check_frame_bounds(frame_number)

        if isinstance(frame_number, int):
            return np.array(get_frame_property(self.frames[frame_number]))
        else:
            return np.array([get_frame_property(frame) for frame in self.frames[frame_number]])

    @staticmethod
    def calculate_von_mises_stress(stress_tensor):
        """
        Calculate the von Mises stress from a stress tensor.

        Args:
            stress_tensor (np.ndarray): Stress tensor in Voigt notation [xx, yy, zz, yz, xz, xy].

        Returns:
            np.ndarray: von Mises stress, shape matches input except last dimension is reduced to 1.
        """
        xx, yy, zz, yz, xz, xy = stress_tensor.T
        von_mises = np.sqrt(0.5 * ((xx - yy)**2 + (yy - zz)**2 + (zz - xx)**2 + 6*(yz**2 + xz**2 + xy**2)))
        return  np.array(von_mises).reshape(-1)

    def _check_frame_bounds(self, frame_number: Union[int, slice]):
        """
        Check if the given frame number or slice is within bounds.

        Args:
            frame_number (Union[int, slice]): The frame number(s) to check.

        Raises:
            ValueError: If the frame number is out of bounds.
        """

        if isinstance(frame_number, int):
            if frame_number < 0 or frame_number >= len(self.frames):
                raise ValueError(f"Invalid frame number {frame_number}")
        elif isinstance(frame_number, slice):
            if frame_number.start is not None and frame_number.start < 0:
                raise ValueError(f"Invalid start frame number {frame_number.start}")
            if frame_number.stop is not None and frame_number.stop > len(self.frames):
                raise ValueError(f"Invalid stop frame number {frame_number.stop}")
        else:
            raise ValueError("Invalid frame number")

    def get_property_magnitude(self, property: Property,
                            calculator: 'PropertyCalculator',
                            frame_number: Union[int, slice] = slice(None)) -> np.ndarray:
        data = self.get_property(property, calculator, frame_number)
        
        """
        Get the magnitude of a property for given frame(s) and calculator.

        Args:
            property (PropertyType): The type of property to retrieve.
            calculator (PropertyCalculator): The calculator used to compute the property.
            frame_number (Union[int, slice], optional): The frame number(s) to get the property for. Defaults to all frames.

        Returns:
            np.ndarray: 
            - For ENERGY: 1D array of shape (n_frames,)
            - For FORCES: 2D array of shape (n_frames, n_atoms) 
            - For STRESS: 1D array of shape (n_frames,)
        """

        if property == Property.ENERGY:
            return np.array(data).reshape(-1)
        elif property == Property.FORCES:
            # For each frame, calculate the magnitude of the force for each atom
            if isinstance(frame_number, int):
                return np.linalg.norm(data, axis=-1).reshape(1, -1)
            else:
                return np.linalg.norm(data, axis=-1)
        elif property == Property.STRESS:
            return self.calculate_von_mises_stress(data)
        else:
            raise ValueError("Invalid property type")

    def get_mae(self, property: Property, 
                reference_calculator: 'PropertyCalculator', 
                target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                frame_number: Optional[Union[int, slice]] = slice(None)) -> Union[np.ndarray, List[np.ndarray], float , List[float]]:
        """
        Calculate Mean Absolute Error (MAE) across all specified frames.

        Args:
            property (PropertyType): The type of property to calculate MAE for.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Optional[Union[int, slice]], optional): The frame number(s) to calculate MAE for. Defaults to all frames.

        Returns:
            Union[float, List[np.ndarray], List[float]]: 
            - For ENERGY: float, List[float]
            - For FORCES: ndarray (n_frames, n_atoms), List[ndarray (n_atoms,)]
            - For STRESS: float, List[float]
        """
        reference_data = self.get_property_magnitude(property, reference_calculator, frame_number)
        
        if not isinstance(target_calculator, list):
            target_calculators = [target_calculator]
        else:
            target_calculators = target_calculator
        
        maes = []
        for calc in target_calculators:
            target_data = self.get_property_magnitude(property, calc, frame_number)
            if property == Property.ENERGY:
                mae = np.mean(np.abs(reference_data - target_data))
            elif property == Property.FORCES:
                mae = np.mean(np.abs(reference_data - target_data), axis=0)
            elif property == Property.STRESS:
                mae = np.mean(np.abs(reference_data - target_data))
            maes.append(mae)
        
        return maes[0] if len(maes) == 1 else maes

    def get_rmse(self, property: Property, 
                reference_calculator: 'PropertyCalculator', 
                target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                frame_number: Optional[Union[int, slice]] = slice(None)) -> Union[float, List[np.ndarray], List[float]]:
        """
        Calculate Root Mean Square Error (RMSE) across all specified frames.

        Args:
            property (PropertyType): The type of property to calculate RMSE for.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Optional[Union[int, slice]], optional): The frame number(s) to calculate RMSE for. Defaults to all frames.

        Returns:
            Union[float, List[np.ndarray], List[float]]: 
            - For ENERGY: float, List[float]
            - For FORCES: ndarray (n_atoms,), List[ndarray (n_atoms,)]
            - For STRESS: float, List[float]
        """
        reference_data = self.get_property_magnitude(property, reference_calculator, frame_number)
        
        if not isinstance(target_calculator, list):
            target_calculators = [target_calculator]
        else:
            target_calculators = target_calculator
        
        rmses = []
        for calc in target_calculators:
            target_data = self.get_property_magnitude(property, calc, frame_number)
            if property == Property.ENERGY:
                rmse = np.sqrt(np.mean((reference_data - target_data) ** 2))
            elif property == Property.FORCES:
                rmse = np.sqrt(np.mean((reference_data - target_data) ** 2, axis=0))  # Mean over frames, keeping per-atom dimension
            elif property == Property.STRESS:
                rmse = np.sqrt(np.mean((reference_data - target_data) ** 2))
            rmses.append(rmse)
        
        return rmses[0] if len(rmses) == 1 else rmses
    
    def get_correlation(self, property: Property, 
                        reference_calculator: 'PropertyCalculator', 
                        target_calculator: Union['PropertyCalculator', List['PropertyCalculator']],
                        frame_number: Optional[Union[int, slice]] = slice(None)) -> Union[float, List[np.ndarray], List[float]]:
        """
        Calculate Pearson correlation coefficient across all specified frames.

        Args:
            property (PropertyType): The type of property to calculate correlation for.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Optional[Union[int, slice]], optional): The frame number(s) to calculate correlation for. Defaults to all frames.

        Returns:
            Union[float, List[np.ndarray], List[float]]: 
            - For ENERGY: float, List[float]
            - For FORCES: ndarray (n_atoms,), List[ndarray (n_atoms,)]
            - For STRESS: float, List[float]
        """
        reference_data = self.get_property_magnitude(property, reference_calculator, frame_number)
        
        if not isinstance(target_calculator, list):
            target_calculators = [target_calculator]
        else:
            target_calculators = target_calculator
        
        correlations = []
        for calc in target_calculators:
            target_data = self.get_property_magnitude(property, calc, frame_number)
            if property == Property.ENERGY:
                if reference_data.size < 2 or target_data.size < 2:
                    correlation = np.nan  # Not enough data points to compute correlation
                else:
                    correlation = pearsonr(reference_data, target_data)[0]
            elif property == Property.FORCES:
                if reference_data.shape[0] < 2:
                    correlation = np.full(reference_data.shape[1], np.nan)  # Not enough frames to compute correlation per atom
                else:
                    # Calculate correlation for each atom across frames
                    correlation = np.array([pearsonr(reference_data[:, atom], target_data[:, atom])[0] for atom in range(reference_data.shape[1])])
            elif property == Property.STRESS:
                if reference_data.size < 2 or target_data.size < 2:
                    correlation = np.nan  # Not enough data points to compute correlation
                else:
                    correlation = pearsonr(reference_data, target_data)[0]
            correlations.append(correlation)
        
        return correlations[0] if len(correlations) == 1 else correlations