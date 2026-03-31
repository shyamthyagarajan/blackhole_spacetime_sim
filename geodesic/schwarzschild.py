import numpy as np
from scipy.integrate import solve_ivp

# Constants
G_CONST = 6.67E-11
SPEED_OF_LIGHT = 3E8

# Key Assumption of theta being pi/2 (equatorial plane)

def schwarzschild_ODE_solver(tau_max, tau_steps, y_0, M, E, L):
    '''
    y_0 = [t_0, phi_0, r_0]
    '''
    tau_arr = np.linspace(0, tau_max, tau_steps)
    ivp_soln = solve_ivp(schwarzschild_geodesic, (0, tau_max), y0=y_0, args = (M, E, L), dense_output=True)
    t, phi, r = ivp_soln.sol(tau_arr)
    return np.array([t, phi, r])

def schwarzschild_geodesic(tau, y, M, E, L):
    '''
    y = [t, phi, r]
    '''
    t, phi, r = y
    t_dot = E / (1 - (2*G_CONST*M)/(SPEED_OF_LIGHT**2 * r))
    phi_dot = L / r**2
    radicand = E**2 - (1 - (2*G_CONST*M)/(SPEED_OF_LIGHT**2 * r)) * (1 + (L/r)**2)
    r_dot = np.sqrt(np.maximum(radicand, 0))
    return np.array([t_dot, phi_dot, r_dot])

if __name__ == "__main__":
    M = 20 * 1.989E30 # kg (20 solar masses)
    r = 7*G_CONST*M / (SPEED_OF_LIGHT**2)
    E = (1 - (2*G_CONST*M)/(SPEED_OF_LIGHT**2 * r)) / np.sqrt(1 - (3*G_CONST*M)/(SPEED_OF_LIGHT**2 * r))
    L = np.sqrt(G_CONST*M*r / (SPEED_OF_LIGHT**2)) / np.sqrt(1 - (3*G_CONST*M)/(SPEED_OF_LIGHT**2 * r))
    t_0 = 0
    phi_0 = 0
    r_0 = r
    y_0 = [t_0, phi_0, r_0]

    tau_max = 1e9
    tau_steps = 10000

    result = schwarzschild_ODE_solver(tau_max, tau_steps, y_0, M, E, L)
    t, phi, r_sol = result

    print(f"t range: {t[0]:.3f} to {t[-1]:.3f}")
    print(f"r range: {r_sol.min():.3f} to {r_sol.max():.3f}")
    print(f"phi range: {phi[0]:.3f} to {phi[-1]:.3f}")