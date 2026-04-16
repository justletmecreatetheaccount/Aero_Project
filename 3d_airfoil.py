import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

WING_SPAN = 75.3
ASPECT_RATIO = 30.9
MASS = 930
AIR_DENSITY = 0.08
DGACD = 0.05 #dimesionless gravtity to aerodynamic center distance
CRUISE_SPEED = 62.5
NUMBER_OF_COEFFICIENTS = 20
REFERENCE_THICKNESS = 0.12 #dimensionless

TARGET_ZERO_LIFT_ANGLE = -3 * np.pi / 180
ZETAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def chord_distribution(_y):
    return WING_SPAN / ASPECT_RATIO

def elliptic_gamma_distribution(y, gamma_0):
    return gamma_0 * (1 - ((2 * y)/WING_SPAN)**2)**(1/2)

def thickness_distribution_slope_theta(reference_thickness, theta):
    # If theta is extremely close to zero, use the mathematical limit
    if theta < 1e-8:
        return (reference_thickness / 0.2) * (0.10497 * np.sqrt(2))

    return reference_thickness / 0.2 * np.sin(theta) * (
        0.0259 * np.cos(theta)**3 + 
        0.0289125 * np.cos(theta)**2 + 
        0.040275 * np.cos(theta) + 
        (0.10497 / np.sqrt(1 - np.cos(theta))) - 
        0.158088
    )

def get_camber_coefficients(number_of_coefficients, lift_coefficient, target_alpha_zero_lift):
    if (number_of_coefficients < 3):
        print("LESS THAN 3 COEFFICIENTS !!!")
    coefficients = np.zeros(number_of_coefficients)
    
    term1 = (4/np.pi) * lift_coefficient * DGACD
    
    coefficients[0] = term1 - 2 * target_alpha_zero_lift
    coefficients[1] = 2 * (term1 - 3 * target_alpha_zero_lift)
    coefficients[2] = 3 * (term1 - 2 * target_alpha_zero_lift)

    return coefficients

def get_camber_coefficient_at_y_locked_alpha_0(number_of_coefficients, gamma_0, target_alpha_zero_lift, y):
    slice_circulation = elliptic_gamma_distribution(y, gamma_0)
    slice_chord = chord_distribution(y)
    slice_lift_coefficient = (slice_circulation * 2) / (slice_chord * CRUISE_SPEED)
    return get_camber_coefficients(number_of_coefficients, slice_lift_coefficient, target_alpha_zero_lift)

def get_camber_coefficient_at_y_locked_alpha_w(number_of_coefficients, gamma_0, target_alpha_w, alpha_induced, lift_slope, y):
    slice_circulation = elliptic_gamma_distribution(y, gamma_0)
    slice_chord = chord_distribution(y)
    slice_lift_coefficient = (slice_circulation * 2) / (slice_chord * CRUISE_SPEED)
    print(slice_lift_coefficient)
    target_alpha_zero_lift = target_alpha_w - alpha_induced - slice_lift_coefficient/lift_slope
    return get_camber_coefficients(number_of_coefficients, slice_lift_coefficient, target_alpha_zero_lift)


def get_thickness_coefficients(number_of_coefficients, reference_thickness):
    coefficients = np.zeros(number_of_coefficients)

    coefficients[0] = (1/np.pi) * integrate.quad(lambda theta: thickness_distribution_slope_theta(reference_thickness, theta), 0, np.pi)[0]

    for i in range(1, number_of_coefficients):
        coefficients[i] = (2/np.pi) * integrate.quad(lambda theta: np.cos(i * theta) * thickness_distribution_slope_theta(reference_thickness, theta), 0, np.pi)[0]

    return coefficients

def get_thick_lift_slope(thickness_coefficients):
    return 2*np.pi + 4 * np.pi * np.sum(thickness_coefficients[1:])

def get_zero_lift_angle(camber_coefficients, thickness_coefficients):
    num = camber_coefficients[1] + 2 * np.sum((camber_coefficients * thickness_coefficients)[1:])
    den = 2 + 4 * np.sum(thickness_coefficients[1:])
    return camber_coefficients[0] - num/den

def get_airfoil_lift(alpha_zero_lift, alpha_effective, thickness_coefficients):
    lift_slope = get_thick_lift_slope(thickness_coefficients)
    return lift_slope * (alpha_effective - alpha_zero_lift)

def get_induced_alpha(lift_coefficient):
    return lift_coefficient / (np.pi * ASPECT_RATIO) #pg 444

def get_gamma_0(lift_coefficient, surface):
    return (lift_coefficient * surface * CRUISE_SPEED * 2) / (np.pi * WING_SPAN)


def get_camber_line_coordinates(C_coeffs, x_c_array):
    """Integrates the slope to find the physical z/c coordinates of the camber line."""
    theta_array = np.arccos(1 - 2 * x_c_array)
    z_c = np.zeros_like(x_c_array)
    
    for i, theta in enumerate(theta_array):
        # The critical 0.5 * sin(phi) term ensures integration is with respect to x, not theta.
        z_c[i], _ = integrate.quad(lambda phi: (C_coeffs[0] + C_coeffs[1]*np.cos(phi) + C_coeffs[2]*np.cos(2*phi)) * 0.5 * np.sin(phi), 0, theta)
    return z_c

def naca_0012_thickness(x_c, reference_thickness):
    """Standard analytical formula for NACA 4-digit thickness distribution."""
    scale = reference_thickness / 0.20
    # clip x_c at 0 to prevent math domain errors at the absolute leading edge
    x_safe = np.clip(x_c, 0, 1)
    return scale * (0.2969*np.sqrt(x_safe) - 0.1260*x_safe - 0.3516*x_safe**2 + 0.2843*x_safe**3 - 0.1015*x_safe**4)

def plot_airfoils(zetas, camber_coeffs, reference_thickness):
    num_plots = len(zetas)
    # Create a vertical stack of subplots, dynamically sizing the figure height
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
    
    # Handle the edge case where only one zeta is passed
    if num_plots == 1:
        axes = [axes]
        
    # Generate common x-coordinates
    x_c = np.linspace(0, 1, 250)
    thickness = naca_0012_thickness(x_c, reference_thickness)

    # Loop through each spanwise station
    for i, zeta in enumerate(zetas):
        ax = axes[i]
        
        # 1. Integrate the specific camber coefficients for this station
        z_c = get_camber_line_coordinates(camber_coeffs[i], x_c)
        
        # 2. Apply standard thickness approximation (perpendicular to chord)
        z_upper = z_c + thickness / 2
        z_lower = z_c - thickness / 2
        
        # 3. Plotting
        color = plt.cm.viridis(i / max(1, len(zetas) - 1))
        
        ax.plot(x_c, z_upper, color=color, linewidth=1.5, label='Surface')
        ax.plot(x_c, z_lower, color=color, linewidth=1.5)
        ax.plot(x_c, z_c, color=color, linestyle='--', linewidth=0.8, alpha=0.6, label='Camber Line')
        ax.axhline(0, color='black', linewidth=0.8, linestyle=':', label='Chord Line')
        
        ax.set_title(f'Station $\\zeta = {zeta:.2f}$')
        ax.set_ylabel("z/c") 
        
        # Ensure the airfoil is not visually distorted
        ax.axis('equal') 
        ax.grid(True, alpha=0.3)
        
        # Add the legend only to the top plot to prevent clutter
        if i == 0:
            ax.legend(loc='upper right')

    # Add the x-axis label only to the bottom plot
    axes[-1].set_xlabel("x/c")
    
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

def plot_camber_lines_only(zetas, camber_coeffs):
    plt.figure(figsize=(12, 6))
    
    # Generate common x-coordinates
    x_c = np.linspace(0, 1, 250)

    # Loop through each spanwise station
    for i, zeta in enumerate(zetas):
        # Integrate the specific camber coefficients for this station
        z_c = get_camber_line_coordinates(camber_coeffs[i], x_c)
        
        # Cycle colors smoothly from purple to yellow
        color = plt.cm.viridis(i / max(1, len(zetas) - 1))
        
        # Plot only the camber line
        plt.plot(x_c, z_c, color=color, linewidth=2, label=f'$\zeta = {zeta:.2f}$')

    # Add the reference chord line
    plt.axhline(0, color='black', linewidth=1, linestyle=':', label='Chord Line')
    
    plt.title("Isolated Camber Lines Across Spanwise Stations")
    plt.xlabel("x/c")
    plt.ylabel("z/c") 
    
    # PRO TIP: If you comment out the line below, matplotlib will auto-scale the y-axis,
    # massively exaggerating the vertical twist and making the reflex camber highly visible!
    plt.axis('equal') 
    
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#QUESTION 1

camber_coefficients = np.zeros((len(ZETAS), NUMBER_OF_COEFFICIENTS))

necessary_lift_coefficient = (2 * MASS * 9.81) / (AIR_DENSITY * CRUISE_SPEED**2 * (WING_SPAN**2 / ASPECT_RATIO))

thickness_coefficients = get_thickness_coefficients(NUMBER_OF_COEFFICIENTS, REFERENCE_THICKNESS)

thick_lift_slope = get_thick_lift_slope(thickness_coefficients)

induced_alpha = get_induced_alpha(necessary_lift_coefficient)

surface, _ = integrate.quad(lambda y: chord_distribution(y), -WING_SPAN/2, WING_SPAN/2)

gamma_0 = get_gamma_0(necessary_lift_coefficient, surface)

camber_coefficients[0] = get_camber_coefficient_at_y_locked_alpha_0(NUMBER_OF_COEFFICIENTS, gamma_0, TARGET_ZERO_LIFT_ANGLE, 0)

true_root_alpha_L0 = get_zero_lift_angle(camber_coefficients[0], thickness_coefficients)
slice_circulation = elliptic_gamma_distribution(0, gamma_0)
slice_chord = chord_distribution(0)
slice_lift_coefficient = (slice_circulation * 2) / (slice_chord * CRUISE_SPEED)
root_alpha_w = (slice_lift_coefficient/thick_lift_slope) + true_root_alpha_L0 + induced_alpha

for i in range(len(ZETAS)):
    camber_coefficients[i] = get_camber_coefficient_at_y_locked_alpha_w(NUMBER_OF_COEFFICIENTS, gamma_0, root_alpha_w, induced_alpha, thick_lift_slope, ZETAS[i]*(WING_SPAN/2))




# Execution
plot_airfoils(ZETAS, camber_coefficients, REFERENCE_THICKNESS)

camber_coefficients = np.zeros((len(ZETAS), NUMBER_OF_COEFFICIENTS))

necessary_lift_coefficient = (2 * MASS * 9.81) / (AIR_DENSITY * CRUISE_SPEED**2 * (WING_SPAN**2 / ASPECT_RATIO))

thickness_coefficients = get_thickness_coefficients(NUMBER_OF_COEFFICIENTS, REFERENCE_THICKNESS)
thick_lift_slope = get_thick_lift_slope(thickness_coefficients)
induced_alpha = get_induced_alpha(necessary_lift_coefficient)

surface, _ = integrate.quad(lambda y: chord_distribution(y), -WING_SPAN/2, WING_SPAN/2)
gamma_0 = get_gamma_0(necessary_lift_coefficient, surface)

# --- THE GEOMETRY ANCHOR ---
# To force a symmetric (0-camber) tip where cl=0, alpha_w must perfectly equal alpha_i.
global_alpha_w = induced_alpha

# Run the unified loop for all stations using the new tip-anchored alpha_w
for i in range(len(ZETAS)):
    camber_coefficients[i] = get_camber_coefficient_at_y_locked_alpha_w(
        NUMBER_OF_COEFFICIENTS, 
        gamma_0, 
        global_alpha_w, 
        induced_alpha, 
        thick_lift_slope, 
        ZETAS[i]*(WING_SPAN/2),
    )

# Execution
plot_airfoils(ZETAS, camber_coefficients, REFERENCE_THICKNESS)
plot_camber_lines_only(ZETAS, camber_coefficients)
