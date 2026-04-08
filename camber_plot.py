import matplotlib.pyplot as plt
import numpy as np

PRECISION = 1000
P = 0.3

def camber_equation(xc, p):
    
    term1 = np.exp(-(xc**2))
    term2 = (1 - np.exp(-1)) * xc
    
    zc_over_c = p * (term1 + term2 - 1)
    
    return zc_over_c


range = np.linspace(0, 1, PRECISION)
zc = camber_equation(range, P)


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(range, zc, color='blue', linewidth=2, label='z_c(x)')

ax.set_title(f"Camber Line with p={P}")
ax.set_xlabel("x/c")
ax.set_ylabel("z/c(x/c)")
ax.grid(True)
ax.legend()

plt.show()
