# Assignment 3
## Structure:
- src/: Source Python files:
  - src/assignment3-2.py: The CLI handler of running the Helmholtz module.
  - src/helmholtz.py: A module solving the Helmholtz equations for a
  predefined 

### Data files:

### Result files:
- results/
  - results/results.txt: Contains all the gathered signal strengths for all the possible router locations.


## Usage:
To create the python environment use:
```bash
python3 -m venv venv
```
Then use source:
```bash
source venv/bin/activate
```
And finally install all packages:
```bash
venv/bin/pip3 install -r requirements.txt
```

### Assignment 1:
Finite element method:

To run the simulation for Re=500, run the Jupyter notebook finite_element.ipynb. Run the cells individually to run the simulation for the other Reynolds numbers.

### Assignment 2:
To run the simulations for all 7125 router positions:
```bash
python3 src/assignment3-2.py --output-dir results --results-file results/results.txt
```

Since running all simulations can take up to 2 hours, to verify a single position, run:
```bash
python3 src/assignment3-2.py --output-dir results --results-file results/results.txt --x <x_position> --y <y_position>
```



