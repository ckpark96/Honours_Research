# -*- coding: utf-8 -*-
"""
@author: svermani

"""

import numpy as np
from Element_2D_1 import *
import scipy as sc
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy.stats import linregress
import time as time


t_s = time.time()


# Exact Solutions and Source Term

def phi(x,y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def u(x,y):
    u_x = np.pi*np.cos(np.pi*x)*np.sin(2*np.pi*y)
    u_y = 2*np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*y)
    return u_x, u_y

def f(x,y):
    return -np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y) - 4*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y)



p = 10
#K = 4 
# Test = [4]
Test = [4,16,25,36,49,64,81,100]
# 		121,144]
		#,169,196]

total_error_x = np.zeros(len(Test))
# total_error_y = []

h_x_all = []
h_y_all = []

for idx in range(len(Test)):
    K = Test[idx]
    K_x = K_y = int(np.sqrt(K))
    
    X_dim = (0,1)
    Y_dim = (0,1)
    
    h_x = (X_dim[1]-X_dim[0])/(K_x)
    h_y = (Y_dim[1]-Y_dim[0])/(K_y)
    
    h_x_all.append(h_x)
    h_y_all.append(h_y)
    
    E = incidence_matrix_2d(p)
    
    # Assembling (Global A matrix)
    
    matrix_list_A = []
    matrix_list_B = []
    domain = []
    
    for j in range(K_y):
        domain_y = (j*h_y,(j+1)*h_y)
        for i in range(K_x):
            domain_x = (i*h_x,(i+1)*h_x)
            
            domain.append([domain_x,domain_y])
            A,B = solver_2d(p,domain_x,domain_y,E,f)
            matrix_list_A.append(A)
            matrix_list_B.append(B)
    
    
    big_A = sc.linalg.block_diag(*matrix_list_A)
    
    #Global Numbering
    
    GLobal_num = np.reshape(np.arange(K*(2*p*(p+1) + p**2)),((2*p*(p+1))+p**2,K),order='F')
    
    all_elms = np.reshape(np.arange(K),(K_x,K_y))
    
    
    # Connectivity Matrix 
    
    num_of_lmbd = ((K_x-1)*K_y + (K_y-1)*K_x)*p 
    
    C_1 = np.zeros((num_of_lmbd,K*((p*2*(p+1))+p**2)))
    
    x_edges_c = C_gathering_x(all_elms,K_x,K_y,GLobal_num,p)
    
    y_edges_c = C_gathering_y(all_elms,K_x,K_y,GLobal_num,p)          
    
    C_final = C_gather(C_1,x_edges_c,y_edges_c,num_of_lmbd)
    
    # Getting together everything into final system
    
    Big_b = np.concatenate(matrix_list_B)
    
    fill_b = np.zeros((num_of_lmbd))
    
    fill = np.zeros((num_of_lmbd,num_of_lmbd))
    
    Final_A = np.block([[big_A,C_final.T],[C_final,fill]])
    
    Final_B = np.concatenate((Big_b,fill_b))
    
    
    # Solve system and reconstruction 
    
    soln = np.linalg.solve(Final_A,Final_B)
    
    
    ## Results and L2 Error
    
    
    
    
    Error_x = np.zeros(K)
    Error_y = []
    
    
    
    for h in range(K):
        ux_coeff = soln[GLobal_num[0,h]:GLobal_num[p*(p+1),h]]
        uy_coeff = soln[GLobal_num[p*(p+1),h]:GLobal_num[2*p*(p+1),h]]
        x_dom = domain[h][0] 
        y_dom = domain[h][1]
        error_x, err_intg = L2_error(u,p,ux_coeff,uy_coeff,100,x_dom,y_dom,h,plot=False)
    
        # Err_x = np.zeros((len(w_int),len(w_int)))
        # for m in range(len(w_int)):
        #     for g in range(len(w_int)):
        #         Err_x[m,g] = error_x[m,g]*w_int[g]*w_int[m]
                
                
        
        # Error_x[h] = np.sum(Err_x,dtype="float64")
        Error_x[h] = err_intg
        
    l2_error_x = np.sqrt(np.sum(Error_x,dtype="float64"))
    
    # l2_error_y = np.sqrt(np.sum(Error_y))
    
    total_error_x[idx]= l2_error_x
    
    # total_error_y.append(l2_error_y)
    
    print('K =',K,'Done')
    

#%%
r = lambda f,p: f - f % p
	
slope, intercept, r_value, p_value, std_err = linregress(np.log(h_x_all),np.log(total_error_x))
print("Slope: ",slope,"\n")
print("R2 value: ",r_value)


plt.semilogy(h_x_all,total_error_x)
plt.xlabel("hx")
plt.ylabel("L2 Error")
plt.xlim((max(h_x_all),min(h_x_all)))
plt.title("L2 Error Analysis for p = "+ str(p)+", (Slope = " +str(r(slope,0.001)) + ", R2 = "+ str(r(r_value,0.001))+")" )



tf = time.time()

print("Completed in ",tf-t_s," seconds")



