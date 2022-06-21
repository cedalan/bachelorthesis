#Author: Christian Elias Anderssen Dalan, April 2022
import numpy as np
from scipy.integrate import quad

def gaussian_product(a, b, x_a, x_b):
    """
    Takes two functions $e^{-a(x - x_a)^2}$ and e^{-b(x - x_b)^2}
    
    Rewrites their product into $e^{-\mu x_{ab}^2} \cdot e^{-p(x - x_p)^2}$
    
    Where:
    $\mu = \frac{ab}{a + b}$ (reduced exponent)
    $x_{ab} = x_a - x_b$ (relative separation)
    $p = a + b$ (total exponent)
    $x_p = \frac{ax_a + bx_b}{p}$ ("Center of mass")
    """
    
    mu = a * b / (a + b)
    x_ab = x_a - x_b
    p = a + b
    x_p = (a * x_a + b * x_b) / p 
    
    preexp_fac = np.exp(-mu * x_ab**2)
    
    return preexp_fac, p, x_p

def integrate_gaussian(a = 1):
    """
    This function will integrate a given gaussian gaussian e^-a(x + b)^2 from x between (-inf, inf)
    """
    return np.sqrt(np.pi / a)

def integrate_gaussian_product(a, b, xa, xb):
    """
    Converts a gaussian product of the form:
    $e^{-a(x - c)^2} e^{-b(x - d)^2}$
    using the gaussian_product()- function.
    See gaussian_product for a more in depth explanation.
    """
    w, p, x_p = gaussian_product(a, b, xa, xb)
    return integrate_gaussian_(p) * w

def slater_inner(x, alpha, beta, x0):
    """
    A function that returns the integrand of $\langle S_1 | S_2 \rangle$ 
    
    Parameters:
    
    - Alpha: Exponent of orbital located at x = 0
    - Beta: Exponent of orbital centered at x = x_0
    - $x_0$: Center of the second slater-type orbital
    """
    return np.exp(- alpha * np.abs(x))*np.exp(-beta * np.abs(x - x0))

def integrate_slater(alpha, beta, x0):
    """
    A function that calculates the overlap integral between two slater type orbitals, one centered at x = 0 and one at x = $x_0$.
    
    See slater_inner() for an explanation on parameters
    
    Returns:
    
    -STOS: An array (or int) of the overlap integrals.
    """
    if len(alpha) == 1:
        STOS = quad(slater_inner, -np.inf, np.inf, args=(alpha, beta, x0), epsabs=1e-14)
    else:
        STOS = np.zeros(len(alpha))
        for i in range(len(alpha)):
            STOS[i], _ = quad(slater_inner, -np.inf, np.inf, args=(alpha[i], beta[i], x0[i]), epsabs=1e-14) #INCREASE
    return STOS

def normalize_overlap_GTO(alphas, betas): #Normalizes the overlap of two gaussians with exponents alpha and beta
    constants = (4 * alphas * betas/np.pi**2)**(1/4)
    return constants

def normalize_overlap_STO(alphas, betas): #Normalizes the overlap of two slater-like with exponents alpha and beta
    constants = np.sqrt(alphas * betas)
    return constants