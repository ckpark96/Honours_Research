# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:57:08 2019

@author: svermani
"""

import numpy as np
import Tools as tl
import bf_polynomials as bf
from scipy import linalg
from matplotlib import pyplot as plt

alpha = 1

def f(x):
    fn = -((alpha)**2)*(np.pi**2)*(np.sin(alpha*np.pi*x)) 
    return fn

p = 10
quad_deg = p+1
K = 3
h = 1/K

#Numbering and Gathering Matrix
#GM = np.arange(K*((2*p)+1),dtype=int)
#GM = GM.reshape((K,(2*p)+1)).transpose()
#
#sub = np.arange(K*((2*p)+1),K*((2*p)+1)+4, dtype=int)
#sub = sub.reshape((K,3)).transpose()
#for m in range(1,K):
#    sub[-2,m] = sub[-1,m-1]

num_dofs_per_element = p+1+p
total_internal_dofs = num_dofs_per_element * K
GM_upper = np.arange(total_internal_dofs).reshape((num_dofs_per_element, K), 
                     order='F')
GM_down_1 = np.arange(total_internal_dofs, total_internal_dofs+K)[np.newaxis,:]
GM_down_2 = GM_down_1 + 1

GM = np.vstack((GM_upper, GM_down_1, GM_down_2))


#Assembling

A = np.zeros((np.max(GM)+1,np.max(GM)+1))
B = np.zeros(np.max(GM)+1)

for e in range(K):
    A_k, B_k = tl.element_matrix(f,p,quad_deg,e*h,(e+1)*h)
    for i in range((2*p)+3):
        B[GM[i,e]] += B_k[i]
        for j in range((2*p)+3):
            A[GM[i,e],GM[j,e]] += A_k[i,j]
            

#Boundary conditions
A[GM[-2,0],0] = 0
A[GM[-2,0],GM[-2,0]] = 1
B[-1] = alpha*np.pi*np.cos(alpha*np.pi*1)

#%% Solving the system
    
soln = linalg.solve(A,B)

#soln = np.zeros(K*((2*p)+3))

#M_phi = tl.element_matrix(f,p,quad_deg,e*h,(e+1)*h)[2]

#%% Reconstruction



for k in range(K):
    a,b = k*h, (k+1)*h
    
    xi = np.linspace(-1,1,int(1000/K))
    
    dxdxi = (b-a)*0.5
    nodes = bf.lobatto_quad(p)[0]
    
#    nodes = 0.5*(1-nodes)*a+0.5*(1+nodes)*b
    e_ix = bf.edge_basis(nodes,xi)
    print(GM[p+1,k])
    print(GM[2*p,k]+1)
    phi_h = np.einsum('i,ik->k',soln[GM[p+1,k]:GM[2*p,k]+1],1/dxdxi*e_ix,optimize='optimal')
    
    h_ix =  bf.lagrange_basis(nodes,xi)
    
    u_h =  np.einsum('i,ik->k',soln[GM[0,k]:GM[p+1,k]],h_ix,optimize='optimal')
      
    X = 0.5*(1-xi)*a+0.5*(1+xi)*b
    plt.figure()
    plt.plot(X,phi_h)
    plt.plot(X,-np.sin(X*alpha*np.pi))
    plt.title('Phi')
    
    plt.figure()
    plt.plot(X,u_h)
    plt.plot(X,-alpha*np.pi*np.cos(alpha*X*np.pi))
    plt.title('u')
    