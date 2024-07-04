from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
from ..calculators import PropertyCalculator
from ..frames import Frames
from ..property import Property

class BasePlotter:
    PROPERTY = None 

    @classmethod
    def _create_scatter_plot(cls, ax, reference_data: np.ndarray, target_data: np.ndarray, 
                             reference_calculator: PropertyCalculator, target_calculator: PropertyCalculator) -> plt.Axes:
        """
        Create a scatter plot comparing reference and target data.

        Args:
            ax (plt.Axes): The matplotlib axes to plot on.
            reference_data (np.ndarray): Data from the reference calculator.
            target_data (np.ndarray): Data from the target calculator.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (PropertyCalculator): The target calculator.

        Returns:
            plt.Axes: The matplotlib axes with the scatter plot.
        """

        ax.scatter(reference_data.flatten(), target_data.flatten(), alpha=0.6, color='black', s=20, edgecolors='none')

        all_data = np.concatenate([reference_data.flatten(), target_data.flatten()])
        min_val, max_val = np.min(all_data), np.max(all_data)
        range_val = max_val - min_val
        buffer = range_val * 0.05
        plot_min, plot_max = min_val - buffer, max_val + buffer

        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=1)
        ax.set_xlabel(f'{reference_calculator.name} ({cls.PROPERTY.get_units()})', fontsize=10)
        ax.set_ylabel(f'{target_calculator.name} ({cls.PROPERTY.get_units()})', fontsize=10)
        
        ax.set_title(f'{cls.PROPERTY.get_name().capitalize()} Comparison', fontsize=12)

        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect('equal', adjustable='box')

        return ax
   
    @classmethod
    def _add_metrics_to_plot(cls, ax, frames, reference_calculator, target_calculator, frame_number):
        """
        Add metric statistics to the plot.

        Args:
            ax (plt.Axes): The matplotlib axes to add text to.
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (PropertyCalculator): The target calculator.
            frame_number (Union[int, slice]): The frame number(s) to calculate metrics for.
        """

        mae = frames.get_mae(cls.PROPERTY, reference_calculator, target_calculator, frame_number)
        rmse = frames.get_rmse(cls.PROPERTY, reference_calculator, target_calculator, frame_number)
        correlation = frames.get_correlation(cls.PROPERTY, reference_calculator, target_calculator, frame_number)
        
        stats_text = cls.format_metrics(mae, rmse, correlation)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))

    @classmethod
    def plot_scatter(cls, frames: Frames, reference_calculator: PropertyCalculator, 
                     target_calculator: PropertyCalculator, frame_number: Union[int, slice] = slice(None),
                     title: str = None, display_metrics: bool = True):
        """
        Create and display a scatter plot comparing reference and target data.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (PropertyCalculator): The target calculator.
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            title (str, optional): The title of the plot. Defaults to None.
            display_metrics (bool, optional): Whether to display metric statistics. Defaults to True.
        """

        if cls.PROPERTY is None:
            raise NotImplementedError("Subclasses must define PROPERTY")

        reference_data = frames.get_property_magnitude(cls.PROPERTY, reference_calculator, frame_number)
        target_data = frames.get_property_magnitude(cls.PROPERTY, target_calculator, frame_number)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax = cls._create_scatter_plot(ax, reference_data, target_data, reference_calculator, target_calculator)
        
        if display_metrics:
            cls._add_metrics_to_plot(ax, frames, reference_calculator, target_calculator, frame_number)

        if title:
            ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_all_scatter(cls, frames: Frames, reference_calculator: PropertyCalculator,
                         target_calculator: PropertyCalculator, frame_number: Union[int, slice] = slice(None),
                         display_metrics: bool = True):
        """
        Create and display scatter plots for energy, forces, and stress.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (PropertyCalculator): The target calculator.
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            display_metrics (bool, optional): Whether to display metric statistics. Defaults to True.
        """

        # Avoid circular import
        from ..plotting import EnergyPlotter, ForcesPlotter, StressPlotter

        
        properties = [Property.ENERGY, Property.FORCES, Property.STRESS]
        plotters = [EnergyPlotter, ForcesPlotter, StressPlotter]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax, property, plotter in zip(axes, properties, plotters):
            reference_data = frames.get_property_magnitude(property, reference_calculator, frame_number)
            target_data = frames.get_property_magnitude(property, target_calculator, frame_number)
            
            plotter._create_scatter_plot(ax, reference_data, target_data, reference_calculator, target_calculator)
            if display_metrics:
                plotter._add_metrics_to_plot(ax, frames, reference_calculator, target_calculator, frame_number)
        
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_box(cls, frames: Frames, reference_calculator: PropertyCalculator, 
                 target_calculators: Union[PropertyCalculator, List[PropertyCalculator]], 
                 frame_number: Union[int, slice] = slice(None),
                 per_atom: bool = False):
        """
        Create and display a box plot of errors.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            per_atom (bool, optional): Whether to calculate errors per atom. Defaults to False.
        """
        if cls.PROPERTY is None:
            raise NotImplementedError("Subclasses must define PROPERTY")

        if not isinstance(target_calculators, list):
            target_calculators = [target_calculators]
        fig_width = 3 + (len(target_calculators) * 2)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        
        data = []
        labels = []

        reference_data = frames.get_property_magnitude(cls.PROPERTY, reference_calculator, frame_number)
        num_atoms = frames.get_number_of_atoms()

        for target_calc in target_calculators:
            target_data = frames.get_property_magnitude(cls.PROPERTY, target_calc, frame_number)
            
            error = np.abs(reference_data - target_data)
            
            if error.ndim == 2:  # Multiple frames
                if not per_atom:
                    error = np.mean(error, axis=0)  # Average over frames if not per_atom
                else:
                    error = error.flatten()  # Keep all frame data if per_atom
            elif error.ndim > 2:
                raise ValueError(f"Unexpected number of dimensions in error: {error.ndim}")
            
            if per_atom:
                error = error / num_atoms
            
            data.append(error)
            labels.append(target_calc.name)

        ax.boxplot(data, labels=labels)

        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

        ax.set_axisbelow(True)

        title_suffix = "per atom" if per_atom else "per structure"
        ax.set_title(f'{cls.PROPERTY.get_name().capitalize()} Error Distribution ({title_suffix})')
        ax.set_ylabel(f'{cls.PROPERTY.get_name().capitalize()} Error ({cls.PROPERTY.get_units()})')

        plt.tight_layout()
        plt.show()

    @classmethod
    def format_metrics(cls, mae, rmse, correlation, add_whitespace=""):
        """
        Format the metrics for display.

        Args:
            mae (float): Mean Absolute Error.
            rmse (float): Root Mean Square Error.
            correlation (float): Correlation coefficient.
            add_whitespace (str, optional): Additional whitespace to add. Defaults to "".

        Returns:
            str: Formatted string of metrics.
        """
        raise NotImplementedError("Subclasses must implement format_metrics method")

    @classmethod
    def print_metrics(cls, frames: Frames, 
                      reference_calculator : PropertyCalculator, 
                      target_calculators : Union[PropertyCalculator, List[PropertyCalculator]],
                      frame_number: Union[int, slice] = slice(None)):
        """
        Print the metrics comparing reference and target calculators.

        Args:
            frames (Frames): The Frames object containing the data.
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
            frame_number (Union[int, slice], optional): The frame number(s) to calculate metrics for. Defaults to all frames.
        """
        raise NotImplementedError("Subclasses must implement print_metrics method")
