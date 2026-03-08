import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numba

from plots import rcCustom, rcCustom_wide

def parse_args() -> argparse.Namespace:
    """Function to parse the command line arguments.

    Returns:
        argparse.Namespace: namespace containing the parsed arguments
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "option", help="Determine the code to run", choices=["C", "D"]
    )
    return parser.parse_args()

@numba.njit
def random_seed(seed):
    np.random.seed(seed)

@numba.njit
def get_neighbours(c: np.ndarray, N:int, location):
    y, x = location

    top = c[y - 1, x] if y > 0 else 0
    bottom = c[y + 1, x] if y < N - 1 else 0
    left = c[y, x - 1] if x > 0 else 0
    right = c[y, x + 1] if x < N - 1 else 0

    return int(top), int(bottom), int(right), int(left)

@numba.njit
def get_new_location(c, location, N):
    y, x = location
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    allowed_locations = []

    for dy, dx in directions:
        new_y = y + dy
        new_x = x + dx

        # Boundary conditions for upper and lower boundaries
        if new_y < 0 or new_y > N - 1:
            continue

        # Periodic boundary conditions for left and right boundaries
        if new_x < 0:
            new_x = N - 1

        elif new_x > N - 1:
            new_x = 0

        # Check if new location is not on cluster
        if c[new_y, new_x] != 2:
            allowed_locations.append((new_y, new_x))

    # Remove walker if all neighbours are part of the cluster
    if len(allowed_locations) == 0:
        return None
    
    new_location = allowed_locations[np.random.randint(len(allowed_locations))]
    return new_location

@numba.njit
def single_walker(c:np.ndarray, N:int) -> np.ndarray:
    # Generate walker on random point at the top of the grid
    location = (0, np.random.randint(0, N))
    y, x = location
    c[y, x] = 1

    while c[y, x] != 2:
        c[y, x] = 0

        # Get values of neighbours
        neighbour_val = get_neighbours(c, N, location)

        if 2 in neighbour_val:
            c[y, x] = 2
            break
        
        location = get_new_location(c, location, N)
        y, x = location
        if location is None:
            return c
        else:
            c[y, x] = 1

    return c

@numba.njit
def single_walker_stick(c:np.ndarray, N:int, p_s:float) -> np.ndarray:
    # Generate walker on random point at the top of the grid
    location = (0, np.random.randint(0, N))
    y, x = location
    c[y, x] = 1

    while c[y, x] != 2:
        c[y, x] = 0

        # Get values of neighbours
        neighbour_val = get_neighbours(c, N, location)

        if 2 in neighbour_val:
            if np.random.rand() <= p_s:
                c[y, x] = 2
                break
            else:
                location = get_new_location(c, location, N)

        else:
            location = get_new_location(c, location, N)
        

        if location is None:
            return c
        else:
            y, x = location
            c[y, x] = 1
    
    return c

def main(): 
    args = parse_args()
    option = args.option
    random_seed(0)

    # Grid size
    N = 100

    if option == "C":
        # Create the grid
        c = np.zeros((N, N))

        # Initial stationary point at the bottom of the grid
        c[-1, int(N/2)] = 2

        for walker in range(1000):                
            c = single_walker(c, N)

        #######

        # Average over 10 iterations
        # Array with all final clusters for each iteration
        c_all = np.zeros((N, N, 10))

        for i in range(10):
            print(f"Iteration {i}")
            # Create the grid
            c_i = np.zeros((N, N))

            # Initial stationary point at the bottom of the grid
            c_i[-1, int(N/2)] = 2

            for walker in range(1000):
                c_i = single_walker(c_i, N)

            c_all[:, :, i] = c_i
        
        # Average over all iterations
        c_avg = np.mean(c_all, axis=2)

        fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained')
        ax = axs[0]
        ax.imshow(c)
        ax.set_title(f"Final cluster with size 1000")

        ax = axs[1]
        ax.imshow(c_avg)
        ax.set_title(f"Average cluster for a cluster size of 1000, averaged over 10 iterations")

        for ax in axs.ravel():
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.show()

    elif option == "D":
        p_s = np.array([0.4, 0.6, 0.8, 1.0])
        c_sticking_prob = []
        tot_cluster_members = []

        for p in range(len(p_s)):
            # Array with all final clusters for each iteration
            c_all = np.zeros((N, N, 10))

            for i in range(10):
                print(f"Iteration {i} for sticking probability {p_s[p]}")
                # Create the grid
                c_i = np.zeros((N, N))

                # Initial stationary point at the bottom of the grid
                c_i[-1, int(N/2)] = 2

                for walker in range(1000):                        
                    c_i = single_walker_stick(c_i, N, p_s[p])

                c_all[:, :, i] = c_i
            
            # Average over all iterations
            c_avg = np.mean(c_all, axis=2)
            tot_cluster_members.append(np.sum(c_avg) / 2)
            c_sticking_prob.append(c_avg)
            
        # Show final clusters
        fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
        
        ax = axs[0,0]
        ax.imshow(c_sticking_prob[0])
        ax.set_title(f"Sticking probability = {p_s[0]}")

        ax = axs[0,1]
        ax.imshow(c_sticking_prob[1])
        ax.set_title(f"Sticking probability = {p_s[1]}")

        ax = axs[1,0]
        ax.imshow(c_sticking_prob[2])
        ax.set_title(f"Sticking probability = {p_s[2]}")

        ax = axs[1,1]
        ax.imshow(c_sticking_prob[3])
        ax.set_title(f"Sticking probability = {p_s[3]}")

        for ax in axs.ravel():
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.show()

if __name__ == "__main__":
    main()
