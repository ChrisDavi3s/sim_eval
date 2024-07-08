from .base_plotter import BasePlotter
from typing import Union, List, Optional
import matplotlib.pyplot as plt
from ..property import Property
import numpy as np
from ..frames import Frames
from ..calculators import PropertyCalculator

class ForcesPlotter(BasePlotter):
    PROPERTY = Property.FORCES

    @classmethod
    def format_metrics(cls, mae, rmse, correlation, add_whitespace=""):
        """
        Format the forces metrics for display.

        Args:
            mae (np.ndarray): Mean Absolute Error of forces.
            rmse (np.ndarray): Root Mean Square Error of forces.
            correlation (np.ndarray): Correlation coefficient of forces.
            add_whitespace (str, optional): Additional whitespace to add. Defaults to "".

        Returns:
            str: Formatted string of forces metrics.
        """
        return (f"{add_whitespace}MAE: {np.mean(mae):.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}RMSE: {np.mean(rmse):.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}Correlation: {np.mean(correlation):.6f}")

    @classmethod
    def print_metrics(cls, frames: Frames, 
                      reference_calculator: PropertyCalculator, 
                      target_calculators: Union[PropertyCalculator, List[PropertyCalculator]],
                      frame_number: Union[int, slice] = slice(None)):
        """
        Print the forces metrics comparing reference and target calculators.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, slice], optional): The frame number(s) to calculate metrics for. Defaults to all frames.
        """

        print(f"\nForces Metrics (vs {reference_calculator.name}):")
        print("----------------")
        for target_calc in (target_calculators if isinstance(target_calculators, list) else [target_calculators]):
            mae = frames.get_mae(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            rmse = frames.get_rmse(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            correlation = frames.get_correlation(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            number_of_atoms = frames.get_number_of_atoms()
            
            print(f"\n  {target_calc.name}:")
            print("    Total:")
            print(cls.format_metrics(mae, rmse, correlation, "      "))
            print(f'      MAE (average per atom): {np.mean(mae) / number_of_atoms:.6f} {cls.PROPERTY.get_units()}')
            print(f'      RMSE (average per atom): {np.mean(rmse) / number_of_atoms:.6f} {cls.PROPERTY.get_units()}')
            
            print('\n    Per Atom Type:')
            atoms_dict = frames.get_atom_types_and_indices()
            for atom_type, indices in atoms_dict.items():
                mae_per_atom = mae[..., indices]
                rmse_per_atom = rmse[..., indices]
                correlation_per_atom = correlation[..., indices]
                print(f"      {atom_type}:")
                print(cls.format_metrics(mae_per_atom, rmse_per_atom, correlation_per_atom, "        "))

    @classmethod
    def plot_box(cls, frames: Frames, reference_calculator: PropertyCalculator,
                 target_calculators: Union[PropertyCalculator, List[PropertyCalculator]],
                 frame_number: Union[int, slice] = slice(None),
                 per_atom: bool = False,
                 group_spacing: float = 1.0,
                 box_spacing: float = 0.25,
                 atom_types: Optional[List[str]] = None):
        """
        Create and display a box plot of force errors with adjustable spacing and atom type selection.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            per_atom (bool, optional): Whether to calculate errors per atom type. Defaults to False.
            group_spacing (float, optional): Spacing between groups (atom types or calculators). Defaults to 1.0.
            box_spacing (float, optional): Spacing between boxes within a group. Defaults to 0.25.
            atom_types (Optional[List[str]], optional): List of atom types to plot. If None, all atom types are plotted. Only used when per_atom is True.
        """
        if not isinstance(target_calculators, list):
            target_calculators = [target_calculators]
        
        data = []
        labels = []
        positions = []
        all_atom_types = frames.get_atom_types_and_indices()
        
        if per_atom and atom_types:
            atom_types_dict = {at: all_atom_types[at] for at in atom_types if at in all_atom_types}
        else:
            atom_types_dict = all_atom_types

        fig_width = 3 + (len(atom_types_dict) * len(target_calculators) * (1 + box_spacing) + (len(atom_types_dict) - 1) * group_spacing) if per_atom else 3 + (len(target_calculators) * (1 + group_spacing))
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        reference_forces = frames.get_property_magnitude(cls.PROPERTY, reference_calculator, frame_number)
        
        if per_atom:
            for i, (atom_type, indices) in enumerate(atom_types_dict.items()):
                for j, target_calc in enumerate(target_calculators):
                    target_forces = frames.get_property_magnitude(cls.PROPERTY, target_calc, frame_number)
                    error = np.abs(reference_forces[..., indices] - target_forces[..., indices])
                    if error.ndim == 2:
                        error = error.flatten()
                    data.append(error)
                    labels.append(f"{atom_type}\n({target_calc.name})")
                    positions.append(i * (len(target_calculators) * (1 + box_spacing) + group_spacing) + j * (1 + box_spacing))
        else:
            for i, target_calc in enumerate(target_calculators):
                target_forces = frames.get_property_magnitude(cls.PROPERTY, target_calc, frame_number)
                error = np.abs(reference_forces - target_forces)
                if error.ndim == 2:
                    error = error.flatten()
                data.append(error)
                labels.append(target_calc.name)
                positions.append(i * (1 + group_spacing))

        ax.boxplot(data, positions=positions, labels=labels, widths=0.8)
        
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
        ax.set_axisbelow(True)
        
        title = 'Forces Error Distribution (per atom type)' if per_atom else 'Total Forces Error Distribution'
        ax.set_title(title)
        ax.set_ylabel(f'Force Error ({cls.PROPERTY.get_units()})')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

