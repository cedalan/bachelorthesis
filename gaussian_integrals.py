import numpy as np
from scipy.integrate import simps
def gaussian_product(a, b, c, d):
    """
    takes two functions e^-a(x - c)^2 and e^-b(x - d)^2
    
    rewrites the product to e^-(x - x_prime)^2 * e^(-p + x_prime^2)
    
    which is then returned as x_prime and f, where f = e^(-p + x_prime^2)
    """
    x_prime = (a*c + b*d)/(a + b)
    p = (a*c**2 + b*d**2)/(a + b)
    f = np.exp(-p + x_prime**2)
    return f, x_prime

def integrate_gaussian(a = 1):
    """
    This function will integrate a given gaussian gaussian e^-a(x + b)^2 from x between (-inf, inf)
    """
    return np.sqrt(np.pi / a)

def convert_and_integrate(a, b, c, d):
    weight, new_x = gaussian_product(a, b, c, d)
    return integrate_gaussian() * weight

def slater_inner(x, alpha, beta, x0):
    return np.exp(- alpha * np.abs(x))*np.exp(-beta * np.abs(x - x0))

def integrate_slater(x, alpha, beta, x0):
    STOS = slater_inner(x, alpha, beta, x0)
    STOS = simps(STOS, x)
    return STOS

def initialize_conditions(alpha_min=0.1, alpha_max=3, beta_min=0.1, beta_max=3, x0_min=0.1, x0_max=5, N=100, random = True):
    if random:
        alphas = np.random.rand(N) * (alpha_max - alpha_min) + alpha_min
        betas = np.random.rand(N) * (beta_max - beta_min) + beta_min
        x0 = np.random.rand(N) * (x0_max - x0_min) + x0_min
    return alphas, betas, x0