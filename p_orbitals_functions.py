import numpy as np
import braketlab as bk
import sympy as sp

def create_grid(N_grid = 10):
    A_min = -0.1
    A_max = -5
    B_min = 0.1
    B_max = 5
    c_min = -2
    c_max = 2
    
    A = np.linspace(A_min, A_max, N_grid)
    B = np.linspace(B_min, B_max, N_grid)
    c = np.linspace(c_min, c_max, N_grid)
    
    grid = np.zeros((N_grid**3, 3))
    
    for i in range(N_grid):
        for j in range(N_grid):
            for k in range(N_grid):
                grid[i * N_grid**2 + j * N_grid + k, 0] = A[i]
                grid[i * N_grid**2 + j * N_grid + k, 1] = B[j]
                grid[i * N_grid**2 + j * N_grid + k, 2] = c[k]
                
    return grid

def integrate_grid(grid, alpha = 1, m = 1, l = 1):
    x = sp.symbols("x")
    
    GTOS = np.zeros(len(grid))
    STOS = np.zeros(len(grid))
    for i, triplet in enumerate(grid):
        #potential = -1 / sp.sqrt((x-triplet[2])**2 + 1)
        potential = bk.ket(-1 / sp.sqrt((x-triplet[2])**2 + 1))
        
        #GTOS:
        phi_A = bk.ket( (x-triplet[0])*sp.exp(-alpha*(x-triplet[0])**2) )
        phi_A = phi_A * (phi_A.bra@phi_A)**-.5
        phi_B = bk.ket((x-triplet[1])*sp.exp(-alpha*(x-triplet[1])**2) )
        phi_B = phi_B * (phi_B.bra@phi_B)**-.5
        phi_B = phi_B * potential
        
        #print(phi_A.bra@phi_B)
        GTOS[i] = phi_A.bra@phi_B
        
        #STOS:
        phi_A = bk.ket( (x-triplet[0])*sp.exp(-alpha*sp.Abs(x-triplet[0])) )
        phi_A = phi_A * (phi_A.bra@phi_A)**-.5
        phi_B = bk.ket( (x-triplet[1])*sp.exp(-alpha*sp.Abs(x-triplet[1])) )
        phi_B = phi_B * (phi_B.bra@phi_B)**-.5
        phi_B = phi_B * potential
        
        STOS[i] = phi_A.bra@phi_B
        
        #print(f"GTO: {GTOS[i]}, STO: {STOS[i]}")
    return GTOS, STOS

def create_test(regressor, N_test = 100, alpha = 1, m = 1, l = 1):
    grid = np.zeros((N_test,3))
    x = sp.symbols("x")
    
    A = np.random.rand(N_test)*-4.9 - 0.1
    B = np.random.rand(N_test)*4.9 + 0.1
    c = np.random.rand(N_test)*4 - 2
    
    grid[:,0] = A; grid[:,1] = B; grid[:,2] = c
    
    true_dI = np.zeros(N_test)
    for i, triplet in enumerate(grid):
        #potential = -1 / sp.sqrt((x-triplet[2])**2 + 1)
        potential = bk.ket(-1 / sp.sqrt((x-triplet[2])**2 + 1))
        
        #GTOS:
        phi_A = bk.ket( (x-triplet[0])*sp.exp(-alpha*(x-triplet[0])**2) )
        phi_A = phi_A * (phi_A.bra@phi_A)**-.5
        phi_B = bk.ket((x-triplet[1])*sp.exp(-alpha*(x-triplet[1])**2) )
        phi_B = phi_B * (phi_B.bra@phi_B)**-.5
        phi_B = phi_B * potential
        
        #print(phi_A.bra@phi_B)
        GTO = phi_A.bra@phi_B
        
        #STOS:
        phi_A = bk.ket( (x-triplet[0])*sp.exp(-alpha*sp.Abs(x-triplet[0])) )
        phi_A = phi_A * (phi_A.bra@phi_A)**-.5
        phi_B = bk.ket( (x-triplet[1])*sp.exp(-alpha*sp.Abs(x-triplet[1])) )
        phi_B = phi_B * (phi_B.bra@phi_B)**-.5
        phi_B = phi_B * potential
        
        STO = phi_A.bra@phi_B
        
        true_dI[i] = STO - GTO
        
    predicted_dI = regressor.predict(grid)
    
    return grid, predicted_dI, true_dI