import numpy as np
import matplotlib.pyplot as plt

# boundary conditions
c_y0 = 0
c_y1 = 1

# initial conditions
c_t0 = 0

# interval lengths
N = 10
t0 = 0
tN = 100
delta_x = 1 / N
delta_y = 1 / N
delta_t = 0.0001

# parameters
D = 1

# array with x and y
c = np.zeros((N, N))
c[0,] = c_y1
print(c)

def stability_condition(delta_t, delta_x, D):
    stability_check = (4 * delta_t * D) / (delta_x ** 2)
    if stability_check <= 1:
        print(f"The stability condition is satisfied, value is: {stability_check}")
        return True
    
    print(f"The stability condition is not satisfied, value is: {stability_check}")
    return False

def concentration_timestep():
    for x, y in range(delta_x):

        # Boundary condition
        if y == 0 or y == delta_x - 1:
            continue

        if x == delta_x - 1:
            c[y, x] = c[y, x] + delta_t * D / (delta_x ** 2) * (c[y, 0] + c[y, x-1] + c[y+1, x] + c[y-1, x] - 4*c[y, x])

        else:
            c[y, x] = c[y, x] + delta_t * D / (delta_x ** 2) * (c[y, x+1] + c[y, x-1] + c[y+1, x] + c[y-1, x] - 4*c[y, x])

    return c