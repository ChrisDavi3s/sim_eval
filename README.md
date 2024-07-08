# sim_eval

sim_eval (better name suggestions appreciated) is a Python library for analyzing, quantifying, and comparing molecular simulation calculation errors.

It excels in benchmarking machine learning potentials (like those from NequIP/Allegro) against reference calculations (such as VASP DFT).

## Use Case

If you:
  - (a) some structures (called frames in code) and,
  - (b) have some associated calculations / a trained model that can calculate properties for those atoms
    
... look no further!

## Features

- Load and process molecular dynamics generated frames

- Compare energy, forces, and stress predictions from different calculators against some reference benchmark method

- Calculate various error metrics (MAE, RMSE, correlation)
- Generate publication-quality plots for easy comparison and analysis

**Most plotters will accept MULTIPLE target systems allowing comparison of methods against each other on the SAME plot.**

## Currently Implemented Comparisons
- Nequip / Allegro
  \```python
  NequIPPropertyCalculator()
  \```
- VASP (single OUTCAR file)
  \```python
  VASPOUTCARPropertyCalculator()
  \```
- VASP (multiple OUTCAR files in a directory)
  \```python
  VASPOUTCARDirectoryPropertyCalculator()
  \```
- VASP (a folder of sorted XML outputs)
  \```python
  VASPXMLDDirectoryPropertyCalculator()
  \```
- CHGnet
  \```python
  CHGNetPropertyCalculator()
  \```
- MACE (UNTESTED)
  \```python
  MACEPropertyCalculator()
  \```
- (Anything with an ASE calculator) see https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html

## Installation
To install the SimulationBenchmarks library directly from GitHub, run:
```bash
pip install git+https://github.com/ChrisDavi3s/sim_eval.git
```

## Standalone Notebook

For those unwilling to install, please see the standalone notebook for a ready to go implementation of the library. The notebook can be found @ stand_alone_notebook.ipynb.

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

### EnergyPlotter.print_metrics(frames, vasp_calc, nequip_calc)
(A good model vs a bad model)
```md

Energy Metrics (vs DFT (PBE)):
---------------

  Allegro:
    MAE: 0.435113 eV
    RMSE: 0.458745 eV
    Correlation: 0.995340
    MAE (average per atom): 0.002092 eV
    RMSE (average per atom): 0.002206 eV

  Allegro (2):
    MAE: 735.355491 eV
    RMSE: 735.355548 eV
    Correlation: 0.990630
    MAE (average per atom): 3.535363 eV
    RMSE (average per atom): 3.535363 eV
```

### EnergyPlotter.plot_scatter(frames, vasp_calc, nequip_calc)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/ec69853b-e819-46e2-85a5-22fbc29c9f77)


### ForcesPlotter.plot_box(frames, vasp_calc, [nequip_calc, second_nequip_calc], per_atom=True, group_spacing=1, box_spacing=0.2 , atom_types=['Li', 'P', 'S'])
![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/1c3d5d48-3de7-491a-9ed5-52ae19d0dd63)

### StressPlotter.plot_scatter(frames, vasp_calc, nequip_calc)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/5ecc06cc-0352-4e09-8ee0-b025a2d6ce3a)

### BasePlotter.plot_all_scatter(frames, vasp_calc, nequip_calc)

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/53e28a17-f518-472d-be77-74a208875d8b)

### Plotting two different nequip models against a VASP run:

![image](https://github.com/ChrisDavi3s/sim_eval/assets/9642076/60a1fb75-46b7-4c02-82a5-9ee00f4e3895)


## Quick API overview: 

### Loading Frames

```python
class Frames(
    file_path: str,
    format: str | None = None,
    index: int | str = ':'
)
```

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

### Force box plots (ovverwrites plot_box)
```python
(method) def plot_box(
    frames: Frames,
    reference_calculator: PropertyCalculator,
    target_calculators: PropertyCalculator | List[PropertyCalculator],
    frame_number: int | slice = slice(None),
    per_atom: bool = False,
    group_spacing: float = 1,         # Spacing between groups of boxes (ie different calculators)
    box_spacing: float = 0.25,        # Spacing between boxes in a group (ie atom type)
    atom_types: List[str] | None = None # List of atom types to plot
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
## Testing

Github has a CI pipeline that runs the tests on every push. The tests are also run on every pull request (I think?).

Test are written using the unittest module. To run the tests, navigate to the root directory of the project and run:

```bash
python -m unittest discover
```

or use the interface in your favourite IDE.


## Contributing

Pull requests are welcome. I am happy to help with any issues you may have. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. (I am aware the test coverage is not great, I am working on it!)


❤️
