# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 01:18:49 2019

@author: svermani
"""

import numpy as np
import bf_polynomials as bf
import networkx as nx
from scipy import linalg
from matplotlib import pyplot as plt

alpha = 1

def f(x):
    fn = ((alpha)**2)*(np.pi**2)*(np.sin(alpha*np.pi*x)) 
    return fn

p = 5
quad_deg = p+5

# Mass matrices and source matrix

x_int,w_int = bf.gauss_quad(quad_deg)

nodes = bf.lobatto_quad(p)[0]

h_i = bf.lagrange_basis(nodes,x_int)
e_i = bf.edge_basis(nodes,x_int)

f_i = np.zeros(p)

for i in range(p):
    f_i[i] = np.sum(w_int*f(x_int)*e_i[i,:])
    

M_u = np.einsum("ik, jk, k -> ij", h_i, h_i, w_int, optimize="optimal")
M_phi = np.einsum("ik, jk, k -> ij", e_i, e_i, w_int, optimize="optimal")

#Incidence Matrix

edges = []
nodal_pt = list(range(p+1))
for e in range(p):
    edges.append([e,e+1])

G = nx.DiGraph()
G.add_nodes_from(nodal_pt)
G.add_edges_from(edges)

E = nx.incidence_matrix(G,oriented=True) # this returns a scipy sparse matrix
E = np.transpose(E.toarray())

#%%Solving the system
A1 = np.hstack((M_u, np.transpose(M_phi@E)))
A2 = np.hstack((M_phi@E, np.zeros((p,p))))
A = np.vstack((A1, A2))

#A = np.array([[M_u , np.transpose(np.matmul(M_phi,E))],[(np.matmul(M_phi,E)),0]])
B = np.concatenate((np.zeros(p+1),f_i))


soln = linalg.solve(A,B)

#%% Reconstruction

X = np.linspace(-1,1,100)

e_ix = bf.edge_basis(nodes,X)

phi_h = np.einsum('i,ik->k',soln[p+1:],e_ix,optimize='optimal')

h_ix = bf.lagrange_basis(nodes,X)

u_h =  np.einsum('i,ik->k',soln[:p+1],h_ix,optimize='optimal')

plt.figure()
plt.plot(X,phi_h)
plt.plot(X,-np.sin(X*alpha*np.pi))

plt.figure()
plt.plot(X,u_h)
plt.plot(X,-alpha*np.pi*np.cos(alpha*X*np.pi))

print(u_h)



 

