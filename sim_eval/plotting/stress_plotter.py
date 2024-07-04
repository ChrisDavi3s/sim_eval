from .base_plotter import BasePlotter
from ..property import Property
from typing import Union
from ..frames import Frames
from ..calculators import PropertyCalculator
from typing import List

class StressPlotter(BasePlotter):
    PROPERTY = Property.STRESS

    @classmethod
    def format_metrics(cls, mae, rmse, correlation, add_whitespace=""):
        """
        Format the stress metrics for display.

        Args:
            mae (float): Mean Absolute Error of stress.
            rmse (float): Root Mean Square Error of stress.
            correlation (float): Correlation coefficient of stress.
            add_whitespace (str, optional): Additional whitespace to add. Defaults to "".

        Returns:
            str: Formatted string of stress metrics.
        """
        return (f"{add_whitespace}MAE: {mae:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}RMSE: {rmse:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}Correlation: {correlation:.6f}")

    @classmethod
    def print_metrics(cls, frames: Frames,
                      reference_calculator: PropertyCalculator,
                      target_calculators: Union[PropertyCalculator, List[PropertyCalculator]],
                      frame_number: Union[int, slice] = slice(None)):
        """
        Print the stress metrics comparing reference and target calculators.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, slice], optional): The frame number(s) to calculate metrics for. Defaults to all frames.
        """

        print(f"\nStress Metrics (vs {reference_calculator.name}):")
        print("----------------")
        for target_calc in (target_calculators if isinstance(target_calculators, list) else [target_calculators]):
            mae = frames.get_mae(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            rmse = frames.get_rmse(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            correlation = frames.get_correlation(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            
            print(f"\n  {target_calc.name}:")
            print(cls.format_metrics(mae, rmse, correlation, "    "))
            print("    (All values per structure)")
