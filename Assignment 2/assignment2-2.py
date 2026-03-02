import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numba

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

# @numba.njit
def get_neighbours(c: np.ndarray, N:int, location):
    y, x = location

    top = c[y - 1, x] if y > 0 else 0
    bottom = c[y + 1, x] if y < N - 1 else 0
    left = c[y, x - 1] if x > 0 else 0
    right = c[y, x + 1] if x < N - 1 else 0

    return int(top), int(bottom), int(right), int(left)

# @numba.njit
def get_new_location(location, N, neighbour_val=None):
    y, x = location
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random_direction = np.random.randint(4)

    if neighbour_val is None:
        # Select random direction
        direction = directions[random_direction]
        dy, dx = direction
        # print(direction)

        new_y = y + dy
        new_x = x + dx

    else:
        while neighbour_val[random_direction] == 2:
            # print(f"get new direction, {random_direction}, {neighbour_val[random_direction]}")
            random_direction = np.random.randint(4)

        direction = directions[random_direction]
        dy, dx = direction
        new_y = y + dy
        new_x = x + dx

    if new_y < 0 or new_y > N - 1:
        return None

    elif new_x < 0:
        new_x = N - 1

    elif new_x > N - 1:
        new_x = 0

    new_location = (new_y, new_x)

    return new_location

def single_walker(c:np.ndarray, N:int) -> np.ndarray:
    # Generate walker on random point at the top of the grid
    location = (0, np.random.randint(0, N))
    c[location] = 1

    while c[location] != 2:
        c[location] = 0

        # Get values of neighbours
        neighbour_val = get_neighbours(c, N, location)
        # print(neighbour_val)

        if 2 in neighbour_val:
            c[location] = 2
            # print("walker is now part of cluster")
            # print(c)
            break
        
        location = get_new_location(location, N)
        
        if location is None:
            # print("Remove walker")
            return c
        else:
            c[location] = 1

    return c

def single_walker_stick(c:np.ndarray, N:int, p_s:float) -> np.ndarray:
    # Generate walker on random point at the top of the grid
    location = (0, np.random.randint(0, N))
    c[location] = 1

    while c[location] != 2:
        c[location] = 0

        # Get values of neighbours
        neighbour_val = get_neighbours(c, N, location)
        # print(neighbour_val)

        if 2 in neighbour_val:
            if np.random.rand() <= p_s:
                c[location] = 2
                # print("walker is now part of cluster")
                # print(c)
                break
            else:
                location = get_new_location(location, N, neighbour_val)

        else:
            location = get_new_location(location, N)
        
        if location is None:
            # print("Remove walker")
            return c
        else:
            c[location] = 1

        # print(c)
    
    return c

def main(): 
    args = parse_args()
    option = args.option

    np.random.seed(0)

    # Grid size
    N = 100

    if option == "C":
        # Create the grid
        c = np.zeros((N, N))

        # Initial stationary point at the bottom of the grid
        c[-1, int(N/2)] = 2

        # first_row = c[0,]
        # walker = 0
        # while 0 in first_row:
        #     walker += 1
        #     if walker % 5000 == 0: 
        #         print(f"Random walker number {walker + 1}")
                
        #     c = single_walker(c, N)
        #     first_row = c[0,]

        for walker in range(65000):
            if walker % 5000 == 0: 
                print(f"Random walker number {walker}")
                
            c = single_walker(c, N)
            first_row = c[0,]

        # Show final cluster
        plt.imshow(c)
        plt.title(f"Final cluster after {walker + 1} random walkers")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        

    elif option == "D":
        print("Part D")
        p_s = np.array([0.4, 0.6, 0.8, 1.0])
        # p_s = np.array([0.2, 1.0])
        c_sticking_prob = []
        tot_cluster_members = []

        for p in range(len(p_s)):
            # Create the grid
            c = np.zeros((N, N))

            # Initial stationary point at the bottom of the grid
            c[-1, int(N/2)] = 2

            for walker in range(65000):
                if walker % 5000 == 0: 
                    print(f"Random walker number {walker}")
                    
                c = single_walker_stick(c, N, p_s[p])

            tot_cluster_members.append(np.sum(c) / 2)
            c_sticking_prob.append(c)

        # Show final clusters
        fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')
        fig.suptitle(f"Final clusters after {walker + 1} random walkers, for different sticking probabilities")
        
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

        # for i in range(len(p_s)):
        #     ax = axs[i]
        #     ax.imshow(c_sticking_prob[i])
        #     ax.set_title(f"Sticking probability = {p_s[i]}")
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        plt.show()

        plt.figure()
        plt.plot(p_s, tot_cluster_members)
        plt.grid(True)
        plt.xlabel(r"Sticking probability $p_s$")
        plt.ylabel("Final cluster size")
        plt.show()


if __name__ == "__main__":
    main()
