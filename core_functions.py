import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from scipy.integrate import simps
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from gaussian_integrals import *
from core_integrals import *

def create_grid(alpha_min = 0.1, alpha_max = 5, beta_min = 0.1, beta_max = 5, N = 10, a = 0.1, return_singles = False, non_uniform = False):
    if non_uniform:
        alpha_beta = np.linspace(alpha_min**a, alpha_max**a,N)**(a**-1)
        
        grid = np.zeros((N**2, 2))
        
        for i in range(len(alpha_beta)):
            for j in range(len(alpha_beta)):
                grid[i*N+j,0] = alpha_beta[i]
                grid[i*N+j,1] = alpha_beta[j]
        if return_singles:
            return alpha_beta, grid
    else:
        alphas = np.linspace(alpha_min, alpha_max, N)
        betas = np.linspace(beta_min, beta_max, N)

        grid = np.zeros((N**2,2))

        for i in range(len(alphas)):
            for j in range(len(betas)):
                grid[i*N+j,0] = alphas[i]
                grid[i*N+j,1] = betas[j]
        
        if return_singles:
            return alphas, betas, grid
    return grid

def create_test(alpha_min = 0.1, alpha_max = 5, beta_min = 0.1, beta_max = 5, delta = 10e-2, N = 10000):
    alphas = np.random.rand(N) * (alpha_max - delta) + alpha_min
    betas = np.random.rand(N) * (beta_max - delta) + beta_min
    
    grid = np.zeros((N, 2))
    
    for i in range(N):
        grid[i, 0] = alphas[i]
        grid[i, 1] = betas[i]
    return grid

def train_network(training_data, GTOS, STOS, kernel = RBF()):
    fitting_data = STOS - GTOS
    
    gpr = GaussianProcessRegressor(kernel = kernel, random_state=42).fit(training_data, fitting_data)
    
    return gpr

def optimize_length_parameter(l_min = 0.1, l_max = 1, N_l = 100, N_grid = 10, N_test = 10000):
    training_grid = create_grid(N = N_grid)
    test_grid = create_test(N = N_test)
    
    ls = np.linspace(l_min, l_max, N_l)
    
    max_score = 0
    max_l = 0
    
    score_list = []
    
    for l in ls:
        sto_core_potential = sto_core(training_grid[:,0], training_grid[:,1], 1)
        gto_core_potential = gto_core(training_grid[:,0], training_grid[:,1], 1)

        RBF_kernel = RBF(length_scale=l*np.ones(shape=(2), dtype=float))
        gpr = train_network(training_grid, gto_core_potential, sto_core_potential, kernel = RBF_kernel)
        
        sto_core_test = sto_core(test_grid[:,0], test_grid[:,1], 1)
        gto_core_test = gto_core(test_grid[:,0], test_grid[:,1], 1)

        core_diff_test = sto_core_test - gto_core_test

        accuracy = gpr.score(test_grid, core_diff_test)
        
        score_list.append(accuracy)
        if accuracy > max_score:
            max_l = l
            max_score = accuracy
    return max_score, max_l, ls, score_list

def calculate_error(N_grid = 10, N_samples = 100, uniform_grid = True, plot_errors = False, N_l = 100):
    if uniform_grid:
        grid = create_grid(N = N_grid)
    else:
        grid = create_grid(N = N_grid, non_uniform=True)
        
    GTOS = gto_core(grid[:,0], grid[:,1], 1)
    STOS = sto_core(grid[:,0], grid[:,1], 1)

    length_info = optimize_length_parameter(N_grid = N_grid, N_l = N_l)

    RBF_kernel = RBF(length_scale=length_info[1]*np.ones(shape=(2), dtype=float))

    gpr = train_network(grid, GTOS, STOS, RBF_kernel)
    
    test_grid = create_test(N = N_samples)
    
    GTOS = gto_core(test_grid[:,0], test_grid[:,1], 1)
    STOS = sto_core(test_grid[:,0], test_grid[:,1], 1)
    
    true_dI = STOS - GTOS
    
    predicted_dI = gpr.predict(test_grid)
    
    dI_dI = np.abs(true_dI - predicted_dI)
    
        
    if plot_errors:
        sample_range = np.linspace(1, N_samples, N_samples)
        
        plt.figure(figsize=(10, 8))
        plt.plot(sample_range, dI_dI, label="Error in dI")
        plt.title(f"Error in dI for {N_samples} random (alpha, beta)")
        plt.yscale("log")
        plt.ylabel("Error")
        plt.xlabel("Sample number")
        plt.grid()
        plt.show()
    return dI_dI, np.mean(dI_dI), np.max(dI_dI)

