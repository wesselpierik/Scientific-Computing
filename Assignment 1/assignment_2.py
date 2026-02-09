import numpy as np
import matplotlib.pyplot as plt

# boundary conditions
c_y0 = 0
c_y1 = 1

# initial conditions
c_t0 = 0

# interval lengths
N = 1000
delta_x = 1 / N
delta_y = 1 / N
delta_t = 0.0001

# parameters
D = 1

def stability_condition(delta_t, delta_x, D):
    stability_check = (4 * delta_t * D) / (delta_x ** 2)
    if stability_check <= 1:
        print(f"The stability condition is satisfied, value is: {stability_check}")
        return True
    
    print(f"The stability condition is not satisfied, value is: {stability_check}")
    return False
