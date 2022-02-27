import math

depth_div = 8
width_coeff = 1.8

def _calculate_width(x):

    x *= width_coeff
    new_x = max(depth_div, int(x + depth_div / 2) // depth_div * depth_div)
    if new_x < 0.9 * x:
        new_x += depth_div
    return int(new_x)

result = _calculate_width(32)

print(result)