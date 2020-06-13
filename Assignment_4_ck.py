# -*- coding: utf-8 -*-

import numpy as np
import bf_polynomials as bf
from matplotlib import pyplot as plt

def f(x,alpha):
    return -1*(alpha**2)*((np.pi)**2)*np.sin(alpha*np.pi*x)

def discpoisson(func, p, alpha, K):
    #gaussquad p
    #f
    
    h = 1/K
    xi_int,w_int = bf.gauss_quad(p) #?
    xi_nodes = bf.lobatto_quad(p)[0]
    
    dxdxi = h/2
    
    A = np.zeros((K*(2*p+1),K*(2*p+1))) # K number of 2N+1 square matrices
    b = np.zeros(((2*p+1)*K,1))
    
    #building incidence matrix
    E = np.zeros((p,p))
    np.fill_diagonal(E,-1)
    np.fill_diagonal(E[:,1:], 1)
    extracol = np.zeros((p,1))
    extracol[-1,:] = 1
    E = np.hstack((E,extracol))
    
#    run = 0
    for i in range(K):
        #transformation, ref to phy
        x_int = h/2*xi_int + h/2 + h*i 
            #weights?
        x_nodes = h/2*xi_nodes + h/2 + h*i
        
        f_k = w_int*func(x_int,alpha) #shape (1,p+5)
        
        e_i = bf.edge_basis(x_nodes,x_int)
        e_j = bf.edge_basis(x_nodes,x_int)        
        h_i = bf.lagrange_basis(x_nodes,x_int)
        h_j = bf.lagrange_basis(x_nodes,x_int)
        
        #mass matrices
        Mku = dxdxi*np.einsum("ik, jk, k -> ij", h_i, h_j, w_int, optimize="optimal")
        Mkphi = dxdxi*np.einsum("ik, jk, k -> ij", e_i, e_j, w_int, optimize="optimal")
        
        #building the main sub-matrix equation
        MkphiE = Mkphi@E
        emptysq = np.zeros((p,p))
        emptycol = np.zeros((p+1,1))
        
        M = np.hstack((np.vstack((Mku,MkphiE)), np.vstack((MkphiE.T,emptysq))))
        f_k = f_k.reshape((1,p))
        
        F = np.vstack((emptycol,-1*Mkphi@np.transpose(f_k)))
        
        start = (2*p+1)*i
        end = (2*p+1)*(i+1)
#        
        A[ start:end , start:end ] = M
        b[ start:end , : ] = F
        
#        soln = np.linalg.solve(A,b)
        
        return A
        
       
    
    

    

discpoisson(f, 3, 1, 3)