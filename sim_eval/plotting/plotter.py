from typing import Union, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import numpy as np
from ..property import Property, PropertyMetric, MetricType
from ..frames import Frames
from ..calculators import PropertyCalculator
from warnings import warn

class Plotter:
    """
    A unified plotter class for visualizing and analyzing various properties of molecular systems.
    """

    @staticmethod
    def _get_property_metric(property_name: str, per_atom: bool) -> PropertyMetric:
        """
        Convert a property name string to a PropertyMetric object.

        Args:
            property_name (str): The name of the property ('energy', 'forces', or 'stress').
            per_atom (bool): Whether the metric should be calculated per atom.

        Returns:
            PropertyMetric: The corresponding PropertyMetric object.

        Raises:
            ValueError: If an invalid property name is provided.
        """
        property_name = property_name.lower()
        metric_type = MetricType.PER_ATOM if per_atom else MetricType.PER_STRUCTURE

        if property_name == 'energy':
            return PropertyMetric(Property.ENERGY, metric_type)
        elif property_name == 'forces':
            return PropertyMetric(Property.FORCES, metric_type)
        elif property_name == 'stress':
            return PropertyMetric(Property.STRESS, metric_type)
        else:
            raise ValueError(f"Invalid property name: {property_name}")

    @staticmethod
    def plot_box(frames: Frames,
                property_name: str,
                reference_calculator: PropertyCalculator,
                target_calculators: Union[PropertyCalculator, List[PropertyCalculator]],
                per_atom: bool = False,
                frame_number: Union[int, slice] = slice(None),
                group_spacing: float = 1.0,
                box_spacing: float = 0.25,
                legend_location: Optional[str] = None,
                group_per_species: bool = False,
                allowed_species: Optional[List[str]] = None) -> Tuple[Figure, Axes]:
        """
        Create scatter plots comparing reference and target data for one or more properties.

        Args:
            frames (Frames): The Frames object containing the data.
            properties (Union[str, List[str]]): The name(s) of the property to plot ('energy', 'forces', or 'stress').
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (PropertyCalculator): The target calculator.
            per_atom (Union[bool, Tuple[bool, ...]], optional): Whether to use per-atom metrics. Defaults to False.
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            title (str, optional): The title of the plot. Defaults to None.
            display_metrics (Union[bool, Tuple[bool, ...]], optional): Whether to display metric statistics. Defaults to True.

        Returns:
            Tuple[Figure, Union[Axes, Tuple[Axes, ...]]]: A tuple containing the Figure and Axes objects of the plot.
        """

        property_metric = Plotter._get_property_metric(property_name, per_atom)

        if not isinstance(target_calculators, list):
            target_calculators = [target_calculators]

        reference_data = frames.get_flattened_property_magnitude(property_metric, reference_calculator, frame_number)

        data = []
        labels = []
        positions = []            

        # Determine if we should group by atom types
        if group_per_species and property_name.lower() == 'forces' and per_atom:
            all_atom_types = frames.get_atom_types_and_indices(frame_number)
            # Filter atom types based on allowed species
            if allowed_species:
                atom_types_dict = {at: all_atom_types[at] for at in allowed_species if at in all_atom_types}
            else:
                atom_types_dict = all_atom_types
        else:
            atom_types_dict = {'All': slice(None)}

        fig_width = 3 + (len(atom_types_dict) * len(target_calculators) * (1 + box_spacing) + (len(atom_types_dict) - 1) * group_spacing)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        for i, (atom_type, indices) in enumerate(atom_types_dict.items()):
            for j, target_calc in enumerate(target_calculators):
                target_data = frames.get_flattened_property_magnitude(property_metric, target_calc, frame_number)
                
                error = np.abs(reference_data[indices] - target_data[indices])
                
                if len(error) > 0:  # Only add data if there are valid errors
                    data.append(error)
                    labels.append(f"{target_calc.name} ({atom_type})" if atom_type != 'All' else target_calc.name)
                    positions.append(i * (len(target_calculators) * (1 + box_spacing) + group_spacing) + j * (1 + box_spacing))

        if not data:
            raise ValueError("No valid data to plot. Check your frame selection and atom types.")

        bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True)

        colors = ['white'] + list(plt.cm.Pastel1(np.linspace(0, 1, len(target_calculators) - 1)))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(target_calculators)])
            patch.set_edgecolor('black')

        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f'{property_name.capitalize()} Error Distribution\nReference: {reference_calculator.name}')
        ax.set_ylabel(f'{property_name.capitalize()} Error ({property_metric.get_units()})')

        # Calculate the centers of the groups for x-ticks
        if group_per_species and property_name.lower() == 'forces' and per_atom:
            group_centers = [np.mean(positions[i:i+len(target_calculators)]) for i in range(0, len(positions), len(target_calculators))]
            ax.set_xticks(group_centers)
            ax.set_xticklabels(list(atom_types_dict.keys()), rotation=45, ha='right')
        else:
            # Calculate the centers for each individual box plot
            box_centers = [np.mean(positions[i:i+1]) for i in range(len(positions))]
            ax.set_xticks(box_centers)
            ax.set_xticklabels(labels, rotation=45, ha='right')

        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black', label=calc.name)
                            for i, calc in enumerate(target_calculators)]
        if legend_location:
            ax.legend(handles=legend_elements, title="Calculators", loc=legend_location)

        ax.set_xlabel('Calculators' if not (group_per_species and property_name.lower() == 'forces' and per_atom) else 'Atom Types')

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_property_distribution(frames: Frames,
                        property_name: str,
                        calculators: Union[PropertyCalculator, List[PropertyCalculator]],
                        per_atom: bool = False,
                        frame_number: Union[int, slice] = slice(None),
                        legend_location: Union[str, None] = None) -> Tuple[Figure, Axes]:
        """
        Create a box plot of property distribution for multiple calculators.

        This is just a box plot of the property values for each calculator. No error metrics are calculated.

        Args:
            frames (Frames): The Frames object containing the data.
            property_name (str): The name of the property to plot ('energy', 'forces', or 'stress').
            calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The calculator(s) to plot.
            per_atom (bool, optional): Whether to calculate distribution per atom. Defaults to False.
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            legend_location (str, optional): Location of the legend. If None, no legend is shown.

        Returns:
            Tuple[Figure, Axes]: A tuple containing the Figure and Axes objects of the plot.
        """

        if not isinstance(calculators, list):
            calculators = [calculators]

        property_metric = Plotter._get_property_metric(property_name, per_atom)

        fig, ax = plt.subplots(figsize=(3 + len(calculators), 6))
        data = []
        labels = []

        for calc in calculators:
            # We need to flatten for forces and per_atom as (calc, atoms) ie (1,208) -> (208,)

            property_data = frames.get_flattened_property_magnitude(property_metric, calc, frame_number)
            data.append(property_data)
            labels.append(calc.name)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = ['white'] + list(plt.cm.Pastel1(np.linspace(0, 1, len(calculators) - 1)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')

        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
        ax.set_axisbelow(True)

        title = f"{property_name.capitalize()} Distribution"
        y_label = f"{property_name.capitalize()} ({property_metric.get_units()})"
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Calculators')

        plt.xticks(rotation=45, ha='right')

        if legend_location:
            legend_elements = [Patch(facecolor=color, edgecolor='black', label=calc.name)
                                for color, calc in zip(colors, calculators)]
            ax.legend(handles=legend_elements, title="Calculators", loc=legend_location)

        plt.tight_layout()
        return fig, ax

    @staticmethod
    def _format_metrics(property_metric: PropertyMetric, mae: float, rmse: float, correlation: float, add_whitespace: str = "") -> str:
        """
        Format the metrics for display.

        Args:
            property_metric (PropertyMetric): The property metric being analyzed.
            mae (float): Mean Absolute Error.
            rmse (float): Root Mean Square Error.
            correlation (float): Correlation coefficient.
            add_whitespace (str, optional): Additional whitespace to add. Defaults to "".

        Returns:
            str: Formatted string of metrics.
        """
        return (f"{add_whitespace}MAE: {mae:.4f} {property_metric.get_units()}\n"
                f"{add_whitespace}RMSE: {rmse:.4f} {property_metric.get_units()}\n"
                f"{add_whitespace}Correlation: {correlation:.4f}")

    @staticmethod
    def print_metrics(frames: Frames,
                    property_name: Union[str, List[str]],
                    reference_calculator: PropertyCalculator,
                    target_calculators: Union[PropertyCalculator, List[PropertyCalculator]],
                    per_atom: Union[bool, Tuple[bool, ...]] = False,
                    frame_number: Union[int, slice] = slice(None),
                    group_per_species: bool = False):
        """
        Print the metrics comparing reference and target calculators for specific properties.

        Args:
        frames (Frames): The Frames object containing the data.
        properties (Union[str, List[str]]): The name(s) of the property(ies) to analyze ('energy', 'forces', or 'stress').
        reference_calculator (PropertyCalculator): The reference calculator.
        target_calculators (Union[PropertyCalculator, List[PropertyCalculator]]): The target calculator(s).
        per_atom (Union[bool, Tuple[bool, ...]], optional): Whether to use per-atom metrics. If a tuple, must match the length of properties. Defaults to False.
        frame_number (Union[int, slice], optional): The frame number(s) to calculate metrics for. Defaults to all frames.
        group_per_species (bool, optional): Whether to group metrics by atom species. Defaults to False. Only works for forces that are per atom.
        """
        if isinstance(property_name, str):
            property_name = [property_name]
        if isinstance(per_atom, bool):
            per_atom = (per_atom,) * len(property_name)
        if not isinstance(target_calculators, list):
            target_calculators = [target_calculators]

        for prop, is_per_atom in zip(property_name, per_atom):
            property_metric = Plotter._get_property_metric(prop, is_per_atom)
            # Grouping only makes sense for forces and when per_atom is True
            should_group = group_per_species and property_metric.property_type == Property.FORCES and is_per_atom

            if group_per_species and property_metric.property_type == Property.FORCES and not is_per_atom:
                warn("Grouping per species requested for forces, but 'per_atom' is False. Metrics will not be grouped by species.")
                should_group = False # Ensure grouping is disabled

            print(f"\n{prop.capitalize()} Metrics (vs {reference_calculator.name}):")
            print("----")

            # Get reference data once per property
            reference_data = frames.get_flattened_property_magnitude(property_metric, reference_calculator, frame_number)

            for target_calc in target_calculators:
                target_data = frames.get_flattened_property_magnitude(property_metric, target_calc, frame_number)

                # Ensure data shapes match before calculating metrics
                if reference_data.shape != target_data.shape:
                     warn(f"Data shape mismatch for {prop} between {reference_calculator.name} ({reference_data.shape}) and {target_calc.name} ({target_data.shape}). Skipping metrics for this calculator.")
                     continue

                # Calculate overall metrics
                mae_overall = np.mean(np.abs(reference_data - target_data))
                rmse_overall = np.sqrt(np.mean((reference_data - target_data)**2))
                # Handle potential NaN in correlation if variance is zero or data is constant
                correlation_overall = 0.0 # Default
                if np.var(reference_data) > 1e-9 and np.var(target_data) > 1e-9 and reference_data.size > 1: # Add tolerance and size check
                    correlation_overall = np.corrcoef(reference_data, target_data)[0, 1]
                    if np.isnan(correlation_overall):
                        correlation_overall = 0.0 # Set to 0 if corrcoef returns NaN

                print(f"\n  {target_calc.name}:")
                print(Plotter._format_metrics(property_metric, mae_overall, rmse_overall, correlation_overall, "    "))

                if should_group:
                    print('\n    Per Atom Type:')
                    # Assuming get_atom_types_and_indices returns Dict[str, List[int]]
                    atoms_dict = frames.get_atom_types_and_indices(frame_number)
                    if not atoms_dict:
                         print("      Could not retrieve atom types and indices for grouping.")
                         continue # Skip species grouping if dict is empty

                    for atom_type, indices in atoms_dict.items():
                        if not indices: # Check if the list of indices is empty
                            print(f"      {atom_type}: No data points found (empty index list).")
                            continue
                        try:
                            ref_subset = reference_data[indices]
                            target_subset = target_data[indices]
                        except IndexError as e:
                            print(f"      {atom_type}: Error indexing data - {e}. Max index in list: {max(indices) if indices else 'N/A'}, Data length: {len(reference_data)}")
                            continue


                        if ref_subset.size == 0: # Check if subset is empty after indexing
                            print(f"      {atom_type}: No data points after indexing.")
                            continue

                        # Calculate metrics for this specific species
                        mae_species = np.mean(np.abs(ref_subset - target_subset))
                        rmse_species = np.sqrt(np.mean((ref_subset - target_subset)**2))

                        # Handle correlation calculation edge cases for subsets
                        correlation_species = 0.0 # Default
                        # Add tolerance and size check for variance
                        if np.var(ref_subset) > 1e-9 and np.var(target_subset) > 1e-9 and ref_subset.size > 1:
                             correlation_species = np.corrcoef(ref_subset, target_subset)[0, 1]
                             if np.isnan(correlation_species):
                                 correlation_species = 0.0 # Handle NaN from corrcoef

                        print(f"      {atom_type}:")
                        print(Plotter._format_metrics(property_metric,
                                                    mae_species,
                                                    rmse_species,
                                                    correlation_species,
                                                    "        ")) # Indent further for species
    @staticmethod
    def plot_scatter(frames: Frames,
                property_name: Union[str, List[str]],
                reference_calculator: PropertyCalculator,
                target_calculator: PropertyCalculator,
                per_atom: Union[bool, Tuple[bool, ...]] = False,
                frame_number: Union[int, slice] = slice(None),
                title: Optional[str] = None,
                display_metrics: Union[bool, Tuple[bool, ...]] = True) -> Tuple[Figure, Union[Axes, Tuple[Axes, ...]]]:
        """
        Create scatter plots comparing reference and target data for one or more properties.

        Args:
            frames (Frames): The Frames object containing the data.
            properties (Union[str, List[str]]): The name(s) of the property to plot ('energy', 'forces', or 'stress').
            reference_calculator (PropertyCalculator): The reference calculator.
            target_calculator (PropertyCalculator): The target calculator.
            per_atom (Union[bool, Tuple[bool, ...]], optional): Whether to use per-atom metrics. Defaults to False.
            frame_number (Union[int, slice], optional): The frame number(s) to plot. Defaults to all frames.
            title (str, optional): The title of the plot. Defaults to None.
            display_metrics (Union[bool, Tuple[bool, ...]], optional): Whether to display metric statistics. Defaults to True.

        Returns:
            Tuple[Figure, Union[Axes, Tuple[Axes, ...]]]: A tuple containing the Figure and Axes objects of the plot.
        """
        if isinstance(property_name, str):
            property_name = [property_name]
        if isinstance(per_atom, bool):
            per_atom = (per_atom,) * len(property_name)
        if isinstance(display_metrics, bool):
            display_metrics = (display_metrics,) * len(property_name)

        fig, axes = plt.subplots(1, len(property_name), figsize=(6 * len(property_name), 5))
        if len(property_name) == 1:
            axes = [axes]

        for ax, property_name, is_per_atom, show_metrics in zip(axes, property_name, per_atom, display_metrics):
            property_metric = Plotter._get_property_metric(property_name, is_per_atom)
            reference_data = frames.get_flattened_property_magnitude(property_metric, reference_calculator, frame_number)
            target_data = frames.get_flattened_property_magnitude(property_metric, target_calculator, frame_number)

            ax.scatter(reference_data, target_data, alpha=0.6, color='black', s=20, edgecolors='none')

            all_data = np.concatenate([reference_data, target_data])
            min_val, max_val = np.min(all_data), np.max(all_data)
            range_val = max_val - min_val
            buffer = range_val * 0.05
            plot_min, plot_max = min_val - buffer, max_val + buffer

            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=1)
            ax.set_xlabel(f'{reference_calculator.name} ({property_metric.get_units()})', fontsize=10)
            ax.set_ylabel(f'{target_calculator.name} ({property_metric.get_units()})', fontsize=10)

            title_name = title if title else f'{property_name.capitalize()} Comparison'
            ax.set_title(title_name, fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_aspect('equal', adjustable='box')

            if show_metrics:
                mae = np.mean(frames.get_mae(property_metric, reference_calculator, target_calculator, frame_number))
                rmse = np.mean(frames.get_rmse(property_metric, reference_calculator, target_calculator, frame_number))
                correlation = np.mean(frames.get_correlation(property_metric, reference_calculator, target_calculator, frame_number))
                stats_text = Plotter._format_metrics(property_metric, mae, rmse, correlation)
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))

        plt.tight_layout()
        return fig, tuple(axes) if len(axes) > 1 else axes[0]
