import numpy as np

def kinetic_gto(a, b):
    """
    Returns $\langle g_1 | \hat{H}| g_2 \rangle$, where $g_1 = e^{-a * }$ and $g_2 = e^{-b * x^2}$
    """
    return - a * b * np.sqrt(np.pi) / (np.sqrt(a + b) * (a + b))

def kinetic_sto(a, b):
    """
    Returns $\langle s_1 | \hat{H}| s_2 \rangle$, where $s_1 = e^{-a * \abs{x}}$ and $s_2 = e^{-b * \abs{x}}$
    """
    return - b**2 / (a + b)