# sim_eval

sim_eval (better name suggestions appreciated) is a Python library for analyzing and comparing the results of different molecular dynamics simulations. 

It's particularly useful for benchmarking machine learning potentials (like those from NequIP/Allegro) against reference calculations (such as VASP DFT).

## Features

Load and process molecular dynamics trajectories
Compare energy, forces, and stress predictions from different calculators

Calculate various error metrics (MAE, RMSE, correlation)
Generate publication-quality plots for easy comparison and analysis

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

# Generate plots
EnergyPlotter.plot_scatter(frames, vasp_calc, nequip_calc)
ForcesPlotter.plot_box(frames, vasp_calc, nequip_calc, per_atom=True)
StressPlotter.plot_scatter(frames, vasp_calc, nequip_calc)
BasePlotter.plot_all_scatter(frames, vasp_calc, nequip_calc)
```
