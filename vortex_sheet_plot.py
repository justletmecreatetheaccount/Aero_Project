import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

PRECISION = 30
PLOT_PRECISION = 1000
P = 0.3
C = 75.3 / 30.9
ALPHA = 0.01472
SPEED = 62.6

coefficients = np.zeros(PRECISION)

def dz(theta, p):
    exp = np.exp(- ((1 - np.cos(theta))/2)**2)
    term1 = exp * p * (np.cos(theta) - 1)
    return term1 + p - p/np.e

def gamma(theta, coefficients, speed):
    term1 = coefficients[0] * (1 - np.cos(theta))/(np.sin(theta))
    term2 = 0
    for i in range(1, coefficients.size):
        term2 = term2 + coefficients[i] * np.sin(i * theta)
    return 2 * speed * (term1 + term2)

dz_integral, _ = integrate.quad(lambda theta: dz(theta, P), 0, np.pi)
coefficients[0] = ALPHA - (1/np.pi) * dz_integral

for i in range(1, PRECISION):
    dz_cos_integral, _ = integrate.quad(lambda theta: dz(theta, P) * np.cos(i * theta), 0, np.pi)
    coefficients[i] = (2/np.pi) * dz_cos_integral

print("A1 =", coefficients[1])
print("A2 =", coefficients[2])
print("C_mac =", (np.pi/4)*(coefficients[2]-coefficients[1]))


r = np.linspace(0, np.pi - 0.1, PLOT_PRECISION)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot( (C/2)*(1 - np.cos(r)), gamma(r, coefficients, SPEED), color='blue', linewidth=2, label=r'$\gamma(x)$')

ax.set_title(f"$\gamma(x)$ with p={P}, speed={SPEED}")
ax.set_xlabel("x")
ax.set_ylabel(r'$\gamma(x)$')
ax.grid(True)
ax.legend()

plt.show()

