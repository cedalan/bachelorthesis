import numpy as np
from scipy.special import yn, struve, erfc, kn

def sto(x, a=1, c=0):
    N = np.sqrt(a)
    return N * np.exp(-a * abs(x-c))


def gto(x, a=1):
    N = (2 * a / np.pi) ** (1 / 4)
    return N * np.exp(-a * x ** 2)


def ddx_sto(x, a):
    """
    Deprecated!!!
    """
    N = np.sqrt(a)
    return N * a ** 2 * np.exp(-a * abs(x))


def sto_overlap(a1, a2):
    """
    Analytical solution of the integral <s1|s2>, where
            * s1 = N1*exp(-a1*|x|), N1 = sqrt(a1)
            * s2 = N2*exp(-a2*|x|), N2 = sqrt(a2)
    """
    return 2 * np.sqrt(a1 * a2) / (a1 + a2)

def gto_overlap(a1, a2):
    N1 = (2 * a1 / np.pi) ** (1 / 4)
    N2 = (2 * a2 / np.pi) ** (1 / 4)
    return N1*N2*np.sqrt(np.pi/(a1+a2))

def sto_ddx(a1, a2):
    """
    Analytical solution of the integral <s1|d^2/dx^2|s2>, where
            * s1 = N1*exp(-a1*|x|), N1 = sqrt(a1)
            * s2 = N2*exp(-a2*|x|), N2 = sqrt(a2)
    """
    return -np.sqrt(a1 * a2**2) / (a1 + a2)

def gto_ddx(a1, a2):
    """
    Analytical solution of the integral <g1|d^2/dx^2|g2> where,
            * g1 = N1*exp(-a1*x^2), N1 = (2 * a1 / np.pi) ** (1 / 4)
            * g2 = N2*exp(-a2*x^2), N1 = (2 * a2 / np.pi) ** (1 / 4)
    """
    N1 = (2 * a1 / np.pi) ** (1 / 4)
    N2 = (2 * a2 / np.pi) ** (1 / 4)
    
    return N1*N2*(-2*a1*a2*np.sqrt(np.pi)/(a1+a2)**(3/2))
    

def sto_core(a1, a2, eps=1):
    """
    Analytical solution of the integral -<s1|1/sqrt(x^2+eps^2)|s2>, where
            * s1 = N1*exp(-a1*|x|), N1 = sqrt(a1)
            * s2 = N2*exp(-a2*|x|), N2 = sqrt(a2)
    """
    return (
        -np.sqrt(a1 * a2)
        * np.pi
        * (
            struve(0, np.sqrt(eps) * (a1 + a2))
            - yn(0, np.sqrt(eps) * (a1 + a2))
        )
    )

def gto_core(a1, a2, eps=1):
    N1 = (2 * a1 / np.pi) ** (1 / 4)
    N2 = (2 * a2 / np.pi) ** (1 / 4)
    return -np.exp((a1+a2)/2)*kn(0, (a1+a2)/2)*N1*N2

def sto_gto(a, b):
    """
    Analytical solution of the integral <s|g>, where
            * s = Ns*exp(-a*|x|), Ns = sqrt(a1)
            * g = Ng*exp(-b*|x|), Ng = (2 * b / np.pi) ** (1 / 4)
    """
    return (
        1
        / np.sqrt(b)
        * np.sqrt(np.pi)
        * np.exp(a ** 2 / (4 * b))
        * erfc(a / (2 * np.sqrt(b)))
        * (2 * b / np.pi) ** (1 / 4)
        * np.sqrt(a)
    )