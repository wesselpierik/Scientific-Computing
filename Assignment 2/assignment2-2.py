import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

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

def get_neighbours(c: np.ndarray, N:int, location):
    if location[0] == 0:
        top = 0
        bottom = c[location[0] + 1, location[1]]
        right = c[location[0], location[1] + 1]
        left = c[location[0], location[1] - 1]

    elif location[0] == N - 1:
        top = c[location[0] - 1, location[1]]
        bottom = 0
        right = c[location[0], location[1] + 1]
        left = c[location[0], location[1] - 1]

    elif location[1] == 0:
        top = c[location[0] - 1, location[1]]
        bottom = c[location[0] + 1, location[1]]
        right = c[location[0], location[1] + 1]
        left = 0

    elif location[1] == N - 1:
        top = c[location[0] - 1, location[1]]
        bottom = c[location[0] + 1, location[1]]
        right = c[location[0], 0]
        left = c[location[0], location[1] - 1]

    else:
        top = c[location[0] - 1, location[1]]
        bottom = c[location[0] + 1, location[1]]
        right = c[location[0], location[1] + 1]
        left = c[location[0], location[1] - 1]

    return int(top), int(bottom), int(right), int(left)

def get_new_location(location, N):
    neighbour = np.random.choice(["top", "bottom", "left", "right"])
    print(neighbour)

    if neighbour == "top":
        if location[0] == 0:
            return None            
        else:
            location = (location[0] - 1, location[1])

    elif neighbour == "bottom":
        if location[0] == N - 1:
            return None
        else:
            location = (location[0] + 1, location[1])
    
    elif neighbour == "left":
        if location[1] == 0:
            location = (location[0], N - 1)
        else:
            location = (location[0], location[1] - 1)

    else:
        if location[1] == N - 1:
            location = (location[0], 0)
        else:
            location = (location[0], location[1] + 1)

    return location

def single_walker(c: np.ndarray, N:int) -> np.ndarray:
    # Generate walker on random point at the top of the grid
    location = (0, np.random.randint(0, N - 1))
    c[location] = 1

    while c[location] != 2:
        c[location] = 0

        # Get values of neighbours
        neighbour_val = get_neighbours(c, N, location)
        # print(neighbour_val)

        if 2 in neighbour_val:
            c[location] = 2
            print("walker is now part of cluster")
            break
        
        location = get_new_location(location, N)
        
        if location is None:
            print("Remove walker")
            return c
        else:
            c[location] = 1

        # print(c)
    
    return c

def main(): 
    args = parse_args()
    option = args.option

    # np.random.seed(0)

    # Grid size
    N = 10

    # Create the grid
    c = np.zeros((N, N))

    # Initial stationary point at the bottom of the grid
    c[-1, int(N/2)] = 2

    print(c)

    if option == "C":
        c = single_walker(c, N)
        print(c)

    elif option == "D":
        print("Running code for part D")

if __name__ == "__main__":
    main()
