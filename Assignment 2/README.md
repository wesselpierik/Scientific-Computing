# Assignment 2
## Structure:
- src/: Source Python files:
  - src/assignment2-1.py: Plotting functionality for the the first Diffusion Limited Aggregation assignment.
  - src/assignment2-2.py: File for the Monte Carlo simulation of DLA.
  - src/assignment2-3.py: File for the Gray-Scott simulation.
  - src/dla.py: DLA logic of the first assignment.
  - src/gray_scott.py: The logic for the Gray-Scott models.
  - src/plots.py: Contains custom settings for plots.
  - src/utils.py: Contains several global definitions and paths.

### Data files:
  - data/no_mp.csv: Timing results of experimenting with a grid of 100 by 100 and using no parallelization through either numba or multiprocessing.
  - data/no_mp_large.csv: Same as above using a grid of 1000 by 1000.
  - data/manual_mp.csv: Timing results of experimenting with a grid of 100 by 100 and 16 manual multiprocessing workers.
  - data/manual_mp_large.csv: Same as above using a grid of 1000 by 1000.
  - data/numba_mp.csv: Timing results of experimenting with a grid of 100 by 100 and setting the DLA's stepping iterator to use Numba's built in parallelization.
  - data/numba_mp_large.csv: Same as above using a grid of 1000 by 1000.

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
To plot the growth pattern and path use:
```bash
python3 src/assignment2-1.py eta none
```
```bash
python3 src/assignment2-1.py eta_path none
```

To plot the omega plot, use:
```bash
python3 src/assignment2-1.py omega none
```

To gather the timing data:
```bash
python3 src/assignment2-1.py <gather_small/gather_large> <none/numba/manual>
```
and then plot using:
```bash
python3 src/assignment2-1.py <plot_small/plot_large>
```

### Assignment 2:
To run the simulation and to get the plots for subquestions C and D:
```bash
python3 src/assignment2-2.py 
```


### Assignment 3:
To run the simulation and save the snapshot at timestep 50000:
```bash
python3 src/assignment2-3.py E
```


