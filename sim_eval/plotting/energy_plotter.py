from .base_plotter import BasePlotter
from typing import Union
import matplotlib.pyplot as plt
from ..property import Property
from ..frames import Frames
from ..calculators import PropertyCalculator
from typing import List

class EnergyPlotter(BasePlotter):
    PROPERTY = Property.ENERGY

    @classmethod
    def format_metrics(cls, mae, rmse, correlation, add_whitespace=""):
        """
        Format the energy metrics for display.

        Args:
            mae (float): Mean Absolute Error of energy.
            rmse (float): Root Mean Square Error of energy.
            correlation (float): Correlation coefficient of energy.
            add_whitespace (str, optional): Additional whitespace to add. Defaults to "".

        Returns:
            str: Formatted string of energy metrics.
        """
        return (f"{add_whitespace}MAE: {mae:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}RMSE: {rmse:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}Correlation: {correlation:.6f}")

    @classmethod
    def print_metrics(cls, frames, reference_calculator, target_calculators, frame_number: Union[int, slice] = slice(None)):
        """
        Print the energy metrics comparing reference and target calculators.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, slice], optional): The frame number(s) to calculate metrics for. Defaults to all frames.
        """

        print(f"\nEnergy Metrics (vs {reference_calculator.name}):")
        print("---------------")
        for target_calc in (target_calculators if isinstance(target_calculators, list) else [target_calculators]):
            mae = frames.get_mae(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            rmse = frames.get_rmse(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            correlation = frames.get_correlation(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            
            num_atoms = frames.get_number_of_atoms()
        
            print(f"\n  {target_calc.name}:")
            print(cls.format_metrics(mae, rmse, correlation, "    "))
            print(f"    MAE (average per atom): {mae/num_atoms:.6f} {cls.PROPERTY.get_units()}")
            print(f"    RMSE (average per atom): {rmse/num_atoms:.6f} {cls.PROPERTY.get_units()}")

