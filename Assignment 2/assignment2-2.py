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
    y, x = location

    top = c[y - 1, x] if y > 0 else 0
    bottom = c[y + 1, x] if y < N - 1 else 0
    left = c[y, x - 1] if x > 0 else 0
    right = c[y, x + 1] if x < N - 1 else 0

    return int(top), int(bottom), int(right), int(left)

def get_new_location(location, N):
    y, x = location

    # Select random direction
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    direction = directions[np.random.randint(4)]
    dy, dx = direction
    # print(direction)

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

def single_walker(c: np.ndarray, N:int) -> np.ndarray:
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
            print("walker is now part of cluster")
            print(c)
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

    np.random.seed(0)

    # Grid size
    N = 100

    # Create the grid
    c = np.zeros((N, N))

    # Initial stationary point at the bottom of the grid
    c[-1, int(N/2)] = 2

    # print(c)

    if option == "C":
        walker = 0
        first_row = c[0,]

        while 0 in first_row:
            print(f"Walker n: {walker}")
            c = single_walker(c, N)
            # print(c)
            first_row = c[0,]
            walker += 1

        

    elif option == "D":
        print("Running code for part D")

if __name__ == "__main__":
    main()
