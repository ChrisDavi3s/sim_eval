from .base_plotter import BasePlotter
from typing import Union, List
import matplotlib.pyplot as plt
from ..property import Property
import numpy as np
from ..frames import Frames
from ..calculators import PropertyCalculator

class ForcesPlotter(BasePlotter):
    PROPERTY = Property.FORCES

    @classmethod
    def format_metrics(cls, mae, rmse, correlation, add_whitespace=""):
        return (f"{add_whitespace}MAE: {np.mean(mae):.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}RMSE: {np.mean(rmse):.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}Correlation: {np.mean(correlation):.6f}")

    @classmethod
    def print_metrics(cls, frames, reference_calculator, target_calculators, frame_number: Union[int, slice] = slice(None)):
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
                 per_atom: bool = False):
        if not isinstance(target_calculators, list):
            target_calculators = [target_calculators]

        data = []
        labels = []
        atom_types = frames.get_atom_types_and_indices()
        
        if per_atom:
            fig_width = 3 + (len(target_calculators) * len(atom_types) * 2)
        else:
            fig_width = 3 + (len(target_calculators) * 2)
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        reference_forces = frames.get_property_magnitude(cls.PROPERTY, reference_calculator, frame_number)

        if per_atom:
            for atom_type, indices in atom_types.items():
                for target_calc in target_calculators:
                    target_forces = frames.get_property_magnitude(cls.PROPERTY, target_calc, frame_number)
                    error = np.abs(reference_forces[..., indices] - target_forces[..., indices])
                    if error.ndim == 2:
                        error = error.flatten()
                    data.append(error)
                    labels.append(f"{atom_type} ({target_calc.name})")
        else:
            for target_calc in target_calculators:
                target_forces = frames.get_property_magnitude(cls.PROPERTY, target_calc, frame_number)
                error = np.abs(reference_forces - target_forces)
                if error.ndim == 2:
                    error = error.flatten()
                data.append(error)
                labels.append(target_calc.name)

        ax.boxplot(data, labels=labels)
        
        # Add this line after your boxplot call
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

        # Also, ensure the grid is behind the plot elements
        ax.set_axisbelow(True)

        if per_atom:
            title = 'Forces Error Distribution (per atom type)'
        else:
            title = 'Total Forces Error Distribution'
        
        ax.set_title(title)
        ax.set_ylabel(f'Force Error ({cls.PROPERTY.get_units()})')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
