import numpy as np
from scipy.special import yn, struve, erfc, kn

def sto_overlap(a1, a2):
    """
    Analytical solution of the integral $\langle s_1 | s_2 \rangle$ where
    $$s_1 = N_1 exp(-a_1 |x|), N1 = \sqrt{a_1}$$
    $$s_2 = N_2 exp(-a_2 |x|), N2 = \sqrt{a_2}$$
    """
    N1 = np.sqrt(a1)
    N2 = np.sqrt(a2)
    return 2 * N1 * N2 / (a1 + a2)

def gto_overlap(a1, a2):
    """
    Analytical solution of the integral $\langle g_1 | g_2 \rangle$, where
    $$g_1 = N_1 exp(-a_1 x^2), N_1 = (\frac{2a_1}{\pi})^(1/4)$$
    $$g_2 = N_2 exp(-a_2 x^2), N_2 = (\frac{2a_2}{\pi})^(1/4)$$
    """
    N1 = (2 * a1 / np.pi) ** (1 / 4)
    N2 = (2 * a2 / np.pi) ** (1 / 4)
    return N1*N2*np.sqrt(np.pi/(a1+a2))