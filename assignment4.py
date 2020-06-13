# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:08:34 2019

@author: changkyupark
"""

import numpy as np
import bf_polynomials as bf
from matplotlib import pyplot as plt

alpha = 1

def f(x):
    return -1*(alpha**2)*((np.pi)**2)*np.sin(alpha*np.pi*x)

p = 5
K = 2
h = 1/K

xi_int,w_int = bf.gauss_quad(p+1) # produces N+1 nodes for integration and their weights
xi_nodes = bf.lobatto_quad(p)[0] # produces N+1 gauss-lobatto nodes

dxdxi = h/2

A = np.zeros((K*(2*p+1)+K+1,K*(2*p+1)+K+1)) # K number of 2p+1 square matrices & K+1 for b.c.
b = np.zeros(((2*p+1)*K+K+1,1))

#building incidence matrix
E = np.zeros((p,p))
np.fill_diagonal(E,-1)
np.fill_diagonal(E[:,1:], 1)
extracol = np.zeros((p,1))
extracol[-1,:] = 1
E = np.hstack((E,extracol))

for i in range(K):
    x_int = h/2*xi_int + h/2 + h*i # transformation from reference to physical
    x_int -= x_int[0] # let every new transformation start at 0 instead of where prev left off
    x_nodes = h/2*xi_nodes + h/2 + h*i #transformation
    segment_length = xi_nodes[1:] - xi_nodes[:-1]
    
    # integration
    f_k = np.zeros(p)
    for k in range(1):
        print(k)
        x_int_i = x_int * segment_length[k]/2
        x_int_i = x_int_i + x_nodes[k]
        print('xnodes',x_nodes)
        print('diff', x_nodes[k+1] - x_nodes[k])
        print(x_int_i)
        f_k[k] = np.sum((x_nodes[k+1] - x_nodes[k])*0.5*w_int*f(x_int_i))
    
    e_i = bf.edge_basis(xi_nodes,xi_int)
    e_j = e_i
    h_i = bf.lagrange_basis(xi_nodes,xi_int)
    h_j = h_i
    
    # mass matrices
    Mku = dxdxi*np.einsum("ik, jk, k -> ij", h_i, h_j, w_int, optimize="optimal")
    Mkphi = 1/dxdxi*np.einsum("ik, jk, k -> ij", e_i, e_j, w_int, optimize="optimal")
    
    # building the main sub-matrix equation
    MkphiE = Mkphi@E
    emptysq = np.zeros((p,p))
    emptycol = np.zeros((p+1,1))
    
    M = np.hstack((np.vstack((Mku,MkphiE)), np.vstack((MkphiE.T,emptysq))))
    
    f_k = f_k.reshape((1,p))
    
    F = np.vstack((emptycol,-1*Mkphi@np.transpose(f_k)))
    
    start = (2*p+1)*i
    end = (2*p+1)*(i+1)
    print(M.shape)
    A[ start:end , start:end ] = M
    b[ start:end , : ] = F 

# connectors
N = np.zeros((2,p+1))
N[0,0] = 1
N[-1,-1] = -1

for m in range(K):
    start = (2*p+1)*m
    end = (2*p+1)*m + p +1
    
    A[K*(2*p+1)+m : K*(2*p+1)+m+2 , start:end] = N
    A[start:end , K*(2*p+1)+m : K*(2*p+1)+m+2] = N.T
    
# boundary conditions
A[K*(2*p+1),K*(2*p+1)] = 1
A[K*(2*p+1),0] = 0
b[-1] = alpha*np.pi*np.cos(alpha*np.pi*1)

soln = np.linalg.solve(A,b)
soln = soln.reshape((2*p+1)*K+K+1) # making into 1-D

# reconstruction
xi = np.linspace(-1,1,1000)
h_ix =  bf.lagrange_basis(xi_nodes,xi) #####
e_ix = bf.edge_basis(xi_nodes,xi)

for n in range(K):
    u_h =  np.einsum('i,ik->k',soln[(2*p+1)*n:(2*p+1)*n+p+1],h_ix,optimize='optimal')
    phi_h = np.einsum('i,ik->k',soln[(2*p+1)*n+p+1:(2*p+1)*n+2*p+1],1/dxdxi*e_ix,optimize='optimal')
    
    X = h/2*xi + h/2 + h*n
    
#    plt.figure()
#    plt.plot(X,phi_h,label='approx')
#    plt.plot(X,-np.sin(X*alpha*np.pi),label='exact')
#    plt.title('phi %d' %n)
#    plt.legend()
#
#    plt.figure()
#    plt.plot(X,u_h,label='approx')
#    plt.plot(X,-alpha*np.pi*np.cos(alpha*X*np.pi),label='exact')
#    plt.title('u %d' %n)
#    plt.legend()
    