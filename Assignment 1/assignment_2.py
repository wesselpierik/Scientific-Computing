import math
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

def main():
    # boundary conditions
    c_y0 = 0
    c_y1 = 1

    # initial conditions
    c_t0 = 0

    # interval lengths
    N = 10
    t0 = 0
    tN = 10
    delta_x = 1 / N
    delta_y = 1 / N
    delta_t = 0.001

    # parameters
    D = 1

    # array with x and y
    c = np.zeros((N, N))
    c[0,] = c_y1

    stability = stability_condition(delta_t, delta_x, D)

    for t in np.arange(t0, tN, delta_t):
        c = concentration_timestep(c, delta_x, delta_t, D, N)
        print(c)




def stability_condition(delta_t, delta_x, D):
    stability_check = (4 * delta_t * D) / (delta_x**2)
    if stability_check <= 1:
        print(f"The stability condition is satisfied, value is: {stability_check}")
        return True

    print(f"The stability condition is not satisfied, value is: {stability_check}")
    return False

def concentration_timestep(c, delta_x, delta_t, D, N):
    for x in range(N):
        for y in range(1, N - 1):
            # Boundary condition
            if x == N - 1:
                c[y, x] = c[y, x] + delta_t * D / (delta_x ** 2) * (c[y, 1] + c[y, x-1] + c[y+1, x] + c[y-1, x] - 4*c[y, x])

            elif x == 0:
                c[y, x] = c[y, x] + delta_t * D / (delta_x ** 2) * (c[y, x+1] + c[y, -2] + c[y+1, x] + c[y-1, x] - 4*c[y, x])

            else:
                c[y, x] = c[y, x] + delta_t * D / (delta_x ** 2) * (c[y, x+1] + c[y, x-1] + c[y+1, x] + c[y-1, x] - 4*c[y, x])

    return c

if __name__ == "__main__":
    main()
