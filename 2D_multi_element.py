# -*- coding: utf-8 -*-
"""
@author: svermani

"""

import numpy as np
from Element_2D import *
import scipy as sc
plt.rcParams.update({'figure.max_open_warning': 0})

#### Exact Solutions and Source Term

def phi(x,y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def u(x,y):
    u_x = np.pi*np.cos(np.pi*x)*np.sin(2*np.pi*y)
    u_y = 2*np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*y)
    return u_x, u_y

def f(x,y):
    return -np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y) - 4*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y)


p = 2 # Polynomial degree

K_x = K_y = 2 # No. of elements per axis
K = K_x * K_y # Total no. of elements

X_dim = (0,2) # x-coordinate range
Y_dim = (0,2) # y-coordinate range

h_x = (X_dim[1]-X_dim[0])/(K_x)
h_y = (Y_dim[1]-Y_dim[0])/(K_y)

E = incidence_matrix_2d(p)

#### Assembling Global A matrix (LHS) and b matrix (RHS)

matrix_list_A = []
matrix_list_B = []
domain = []

for j in range(K_y):
    domain_y = (j*h_y,(j+1)*h_y)
    for i in range(K_x):
        domain_x = (i*h_x,(i+1)*h_x)
        # print("domain_y is ", domain_y)
        # print("domain_x is ", domain_x)
        
        domain.append([domain_x,domain_y])
        A,B = solver_2d(p,domain_x,domain_y,E,f)
        matrix_list_A.append(A)
        matrix_list_B.append(B)


big_A = sc.linalg.block_diag(*matrix_list_A)

# Connectivity Matrix (C matrix)

GLobal_num = np.reshape(np.arange(K*(2*p*(p+1) + p**2)),((2*p*(p+1))+p**2,K),order='F') # Numering edges and phi's as column vectors

all_elms = np.reshape(np.arange(K),(K_x,K_y)) # Numbering of elements

num_of_lmbd = ((K_x-1)*K_y + (K_y-1)*K_x)*p  # Numbering of lambdas

C_1 = np.zeros((num_of_lmbd,K*((p*2*(p+1))+p**2))) # preliminary C matrix with only 0s

x_edges_c = C_gathering_x(all_elms,K_x,K_y,GLobal_num,p)

y_edges_c = C_gathering_y(all_elms,K_x,K_y,GLobal_num,p)          

C_final = C_gather(C_1,x_edges_c,y_edges_c,num_of_lmbd)

Big_b = np.concatenate(matrix_list_B)

fill_b = np.zeros((num_of_lmbd))

fill = np.zeros((num_of_lmbd,num_of_lmbd))

Final_A = np.block([[big_A,C_final.T],[C_final,fill]])

Final_B = np.concatenate((Big_b,fill_b))


# Solve system and reconstruction 

soln = np.linalg.solve(Final_A,Final_B)


## Results
L2_error_list = []
for h in range(K):
    ux = soln[GLobal_num[0,h]:GLobal_num[p*(p+1),h]]
    uy = soln[GLobal_num[p*(p+1),h]:GLobal_num[2*p*(p+1),h]]
    x_dom = domain[h][0] 
    y_dom = domain[h][1]
    qx, qy = reconstruct(u,p,ux,uy,100,x_dom,y_dom,h,plot=True)
    
    # L2_error_list.append(L2_error_squared)

# L2_error = np.sqrt(np.sum(L2_error_list))






