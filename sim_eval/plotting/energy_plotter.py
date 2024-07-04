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
        return (f"{add_whitespace}MAE: {mae:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}RMSE: {rmse:.6f} {cls.PROPERTY.get_units()}\n"
                f"{add_whitespace}Correlation: {correlation:.6f}")

    @classmethod
    def print_metrics(cls, frames: Frames, 
                      reference_calculator : PropertyCalculator, 
                      target_calculators : Union[PropertyCalculator, List[PropertyCalculator]],
                      frame_number: Union[int, slice] = slice(None)):
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

