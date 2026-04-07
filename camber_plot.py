import matplotlib.pyplot as plt
import numpy as np
import sys

def camber_equation(xc, p):
    
    term1 = np.exp(-(xc**2))
    term2 = (1 - np.exp(-1)) * xc
    
    zc_over_c = p * (term1 + term2 - 1)
    
    return zc_over_c


range = np.linspace(0, 1, int(sys.argv[1]))
zc = camber_equation(range, float(sys.argv[2]))


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range, zc, color='blue', linewidth=2, label='z_c(x)')

ax.set_title(f"Camber Line with p={sys.argv[2]}")
ax.set_xlabel("x/c")
ax.set_ylabel("z/c(x/c)")
ax.grid(True)
ax.legend()

plt.show()
