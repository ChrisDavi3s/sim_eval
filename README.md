# sim_eval

sim_eval (better name suggestions appreciated) is a Python library for analyzing and comparing the results of different molecular dynamics simulations. 

It's particularly useful for benchmarking machine learning potentials (like those from NequIP/Allegro) against reference calculations (such as VASP DFT).

## Features

Load and process molecular dynamics trajectories
Compare energy, forces, and stress predictions from different calculators

Calculate various error metrics (MAE, RMSE, correlation)
Generate publication-quality plots for easy comparison and analysis

**Most plotters will accept MULTIPLE target systems allowing comparison of methods against each other on the SAME plot.**

### Scatter plots
```python
(method) def plot_scatter(
    frames: Frames,
    reference_calculator: PropertyCalculator,
    target_calculator: PropertyCalculator,
    frame_number: int | slice = slice(None),
    title: str = None,
    display_metrics: bool = True
) -> None
```

### Box plots
```python
(method) def plot_box(
    frames: Frames,
    reference_calculator: PropertyCalculator,
    target_calculators: PropertyCalculator | List[PropertyCalculator],
    frame_number: int | slice = slice(None),
    per_atom: bool = False
) -> None
```

### Metrics

```python
(method) def print_metrics(
    frames: Frames,
    reference_calculator: PropertyCalculator,
    target_calculators: PropertyCalculator | List[PropertyCalculator],
    frame_number: int | slice = slice(None)
) -> None
```
## Currently Implemented Comparisons
- Nequip / Allegro
- VASP
- (Anything with an ASE simulator)

## Installation
To install the SimulationBenchmarks library directly from GitHub, run:
```bash
pip install git+https://github.com/ChrisDavi3s/sim_eval.git
```

## Standalone Notebook

For those unwilling to install, please see our standalone notebook for a ready to go implementation of the library. The notebook can be found @ stand_alone_notebook.ipynb.

## Quick Start

Please see examples/example.ipynb.


Here's a simple example to get you started:
```python
from sim_eval import Frames, VASPXMLPropertyCalculator, NequIPPropertyCalculator
from sim_eval.plotting import BasePlotter,EnergyPlotter, ForcesPlotter, StressPlotter

# Load Frames - can be any ASE supported format
frames = Frames('example_frames.extxyz')

# Set up calculators
vasp_calc = VASPXMLPropertyCalculator('DFT (PBE)', '31_vaspout_xml_frames', 'vasprun_frame')
nequip_calc = NequIPPropertyCalculator('Allegro', 'nequip_model.pth')

# Add data from calculators
frames.add_method_data(vasp_calc)
frames.add_method_data(nequip_calc)

#Print Metrics
EnergyPlotter.print_metrics(frames, vasp_calc, nequip_calc)

# Generate plots
EnergyPlotter.plot_scatter(frames, vasp_calc, nequip_calc)
ForcesPlotter.plot_box(frames, vasp_calc, nequip_calc, per_atom=True)
StressPlotter.plot_scatter(frames, vasp_calc, nequip_calc)
BasePlotter.plot_all_scatter(frames, vasp_calc, nequip_calc)
```

## OUTPUTS 

```md

Energy Metrics (vs DFT (PBE)):
---------------

  Allegro:
    MAE: 0.435113 eV
    RMSE: 0.458745 eV
    Correlation: 0.995340
    MAE (average per atom): 0.002092 eV
    RMSE (average per atom): 0.002206 eV

```

### EnergyPlotter.plot_scatter(frames, vasp_calc, nequip_calc)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/cfa9bd21-7cd7-4602-a61a-f9e07802a1f1)

### ForcesPlotter.plot_box(frames, vasp_calc, nequip_calc, per_atom=True)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/549bad78-18cc-4b4b-a251-6b386f84ec11)

### StressPlotter.plot_scatter(frames, vasp_calc, nequip_calc)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/38edbb8f-402c-40a4-9b8d-66f1bb9ce8a5)

### BasePlotter.plot_all_scatter(frames, vasp_calc, nequip_calc)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/9e9bc19d-18b6-4e25-a533-b8cd4063b593)

### Plotting two different nequip models against a VASP run:

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/60a1fb75-46b7-4c02-82a5-9ee00f4e3895)




