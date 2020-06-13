# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:02:17 2019

@author: changkyupark
"""

import numpy as np
import bf_polynomials as bf
from scipy.integrate import nquad
from matplotlib import pyplot as plt

def phi(x,y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def u(x,y):
    u_x = np.pi*np.cos(np.pi*x)*np.sin(2*np.pi*y)
    u_y = 2*np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*y)
    return u_x, u_y

def f(x,y):
    return -np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y) - 4*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y) 


def incidencemat(p):
    
    # qx part of the matrix   
    ones = np.eye(p)
    subqx = np.zeros((p,(p+1)))
    np.fill_diagonal(subqx, -1)
    np.fill_diagonal(subqx[:,1:],1)
    qx = np.kron(ones, subqx) 

    # qy part of the matrix
    qy = np.zeros((p*p,p*(p+1)))
    np.fill_diagonal(qy, -1)
    np.fill_diagonal(qy[:,p:],1) 

    E = np.hstack([qx,qy])
    return E
 
def massmat(p):
    
    # reference: s,t ; physical: x,y
    s_int,sw_int = bf.gauss_quad(p+1) # p+1 nodes for integration and their weights
    s_nodes = bf.lobatto_quad(p)[0] # p+1 gauss-lobatto nodes
    t_int,tw_int = bf.gauss_quad(p+1)
    t_nodes = bf.lobatto_quad(p)[0]
    
    e_j = bf.edge_basis(t_nodes, t_int)
    e_l = bf.edge_basis(s_nodes, s_int)
    h_i = bf.lagrange_basis(s_nodes, s_int)
    h_k = bf.lagrange_basis(t_nodes, t_int)
    
    b_x = np.kron(e_j,h_i) # node goes first in this numbering system 
    b_y = np.kron(h_k,e_l)
    int_w = np.kron(sw_int,tw_int)
    
    Mx = np.einsum('iw,jw,w->ij', b_x, b_x, int_w, optimize = 'optimal')*hx/hy
    My = np.einsum('iw,jw,w->ij', b_y, b_y, int_w, optimize = 'optimal')*hy/hx
    
    M = np.zeros((2*p*(p+1), 2*p*(p+1)))
    M[:p*(p+1), :p*(p+1)] = Mx
    M[p*(p+1):, p*(p+1):] = My

    return M


def Amatrix(p):
    E = incidencemat(p)
    M = massmat(p)
    zeros = np.zeros((E.shape[0],E.shape[0]))
    A = np.block([[M, E.T],[E, zeros]])
    return A


def bmatrix(p):
    s_nodes = bf.lobatto_quad(p)[0]
    t_nodes = bf.lobatto_quad(p)[0]
    x_transition = lambda u: 0.5*(1-u)*x_min + 0.5*(1+u)*x_max
    y_transition = lambda u: 0.5*(1-u)*y_min + 0.5*(1+u)*y_max
    x_nodes = x_transition(s_nodes)
    y_nodes = y_transition(t_nodes)
    f_k = []
    for j in range(p): 
        for i in range(p):         
            f_ij = nquad(f, [[x_nodes[i], x_nodes[i+1]],[y_nodes[j], y_nodes[j+1]]])[0]
            f_k.append(f_ij)
    f_k = np.array(f_k)
    zeros = np.zeros((2*p*(p+1)))
    bmat = np.block([zeros,f_k])
    return bmat


def solution(p,A,b):
    
    # reconstruction
    s_nodes = bf.lobatto_quad(p)[0]
    t_nodes = bf.lobatto_quad(p)[0]
    S = np.linspace(-1,1,100)
    T = np.linspace(-1,1,100)
    h_iS = bf.lagrange_basis(s_nodes, S)
    e_iS = bf.edge_basis(s_nodes, S)
    h_iT = bf.lagrange_basis(t_nodes, T)
    e_iT = bf.edge_basis(t_nodes, T)
    
    soln = np.linalg.solve(A,b)
    u_i = soln[:2*p*(p+1)]
    u_x = u_i[:p*(p+1)]
    u_y = u_i[p*(p+1):2*p*(p+1)+1]   

    basis_function_x = np.kron(e_iT, h_iS)
    basis_function_y = np.kron(h_iT, e_iS)
    
    Rc_x = np.einsum('i, in -> n', u_x, basis_function_x)*2/hy
    Rc_y = np.einsum('i, in -> n', u_y, basis_function_y)*2/hx
    
    reshaper = lambda u: np.reshape(u,(np.size(S),np.size(T)), order = 'C')
    
    S = np.linspace(x_min,x_max,100)
    T = np.linspace(y_min,y_max,100)
    exact_x, exact_y = u(*np.meshgrid(S,T))
    
    fig = plt.figure()
    fig.suptitle('P = '+str(p))
    
    plt.subplot(321)
    plt.contourf(S,T,reshaper(Rc_x))
    plt.title('u_x approx')
    plt.colorbar()
    
    plt.subplot(322)
    plt.contourf(S,T,reshaper(Rc_y))
    plt.title('u_y approx')
    plt.colorbar()
    
    plt.subplot(323)
    plt.contourf(S,T,exact_x)
    plt.title('u_x exact')
    plt.colorbar()
    
    plt.subplot(324)
    plt.contourf(S,T,exact_y)
    plt.title('u_y exact')
    plt.colorbar()
   
    plt.subplot(325)
    plt.contourf(S,T,reshaper(Rc_x)-exact_x)
    plt.title('u_x error')
    plt.colorbar()
    
    plt.subplot(326)
    plt.contourf(S,T,reshaper(Rc_y)-exact_y)
    plt.title('u_y error')
    plt.colorbar()
    
    plt.show()
    return
    

# x_min, x_max = 0, 2
# hx = x_max - x_min
# y_min, y_max = 0, 2
# hy = y_max - y_min


# solution(p,Amatrix(p),bmatrix(p))  
    
# Global A matrix without C matrix implemented
p = 3
K1D = 3 # must be 3 or higher
K = K1D **2 #square: 4,9,16,...

global_x_min, global_x_max = -1, 1
global_y_min, global_y_max = -1, 1

global_hx = (global_x_max - global_x_min) / K1D
global_hy = (global_y_max - global_y_min) / K1D

pre_global_A = np.zeros(((2*p*(p+1)+p*p)*K, (2*p*(p+1)+p*p)*K))
pre_global_b = np.zeros(((2*p*(p+1)+p*p)*K))

for i in range(K):
    
    x_min, x_max = global_x_min + i*global_hx, global_x_min + (i+1)*global_hx
    y_min, y_max = global_y_min + i*global_hy, global_y_min + (i+1)*global_hy
    hx = x_max - x_min
    hy = y_max - y_min
    
    pre_global_A[(2*p*(p+1)+p*p)*i:(2*p*(p+1)+p*p)*(i+1),(2*p*(p+1)+p*p)*i:(2*p*(p+1)+p*p)*(i+1)] = Amatrix(p)    
    pre_global_b[(2*p*(p+1)+p*p)*i:(2*p*(p+1)+p*p)*(i+1)] = bmatrix(p)


# Connectivity Matrix (C matrix)

sub_C = []

# Elements at corners
cornerA = np.zeros((4*(K1D-2)*3*p,2*p*(p+1))) # total no. of lambdas , no. of sides per element
cornerB = np.copy(cornerA)
cornerC = np.copy(cornerA)
cornerD = np.copy(cornerA)

for i in range(p):
    cornerA[i, p**2+i] = 1 # horizontal side
    cornerA[2*(K1D-2)*3*p+i, (p+1)**2-1+i*(p+1)] = 1 # vertical side
    
    cornerB[(K1D-2)*2*p+i, p**2+i] = 1
    cornerB[2*(K1D-2)*3*p+(K1D-2)*p+i, (p+1)*(p+i)] = -1
    
    cornerC[K1D*p+i, i] = -1
    cornerC[2*(K1D-2)*3*p+(K1D-1)**2*p+i, (p+1)**2-1+i*(p+1)] = 1

    cornerD[2*(K1D-2)*3*p-p+i, i] = -1
    cornerD[4*(K1D-2)*3*p-p+i, (p+1)*(p+i)] = -1
    

# Elements on edges
total = np.ones((4*(K1D-2)))



# Solve system and reconstruction 

    
    