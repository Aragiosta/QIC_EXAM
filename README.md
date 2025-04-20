# SSH chain of interacting hardcore bosons using dipolar (r^-3) hoppings
This repository holds the code of the final project for the Quantum Information and Computing course held at the University of Padua.
The aim of the project was to simulate a 1d chain of hardcore bosons with hoppings which decay as $R^{-3}$ and respect the chiral symmetry of the SSH model.

The structure of the repository is as follows:

## Python module qic_ssh:

'init_ED.py'
'init_MPS.py'
'plots.py': Plotting routines
'setup.py': Initializes the hopping matrix, holds sim. Parameters.
'utils.py': fFunctions for ED calculation of 2-point correlators, string parameters, ent. Spectrum, number of excitations

## Python scripts 'main_ed.py', 'main_mps.py':

Initializes, launches and prints to file the simulation for given size, model...

## Bash script 'handler.sh':
Iterates through different sizes, models, configurations and organizes the printed data.
