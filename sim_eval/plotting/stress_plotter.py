from .base_plotter import BasePlotter
from ..property import Property
from typing import Union

class StressPlotter(BasePlotter):
    PROPERTY = Property.STRESS

    @classmethod
    def format_metrics(cls, mae, rmse, correlation, add_whitespace=""):
        return (f"{add_whitespace}MAE: {mae:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}RMSE: {rmse:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}Correlation: {correlation:.6f}")

    @classmethod
    def print_metrics(cls, frames, reference_calculator, target_calculators, frame_number: Union[int, slice] = slice(None)):
        print(f"\nStress Metrics (vs {reference_calculator.name}):")
        print("----------------")
        for target_calc in (target_calculators if isinstance(target_calculators, list) else [target_calculators]):
            mae = frames.get_mae(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            rmse = frames.get_rmse(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            correlation = frames.get_correlation(cls.PROPERTY, reference_calculator, target_calc, frame_number)
            
            print(f"\n  {target_calc.name}:")
            print(cls.format_metrics(mae, rmse, correlation, "    "))
            print("    (All values per structure)")
