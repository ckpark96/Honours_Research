# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:11:30 2019

@author: Sanjay
"""

import bf_polynomials as bf
import scipy.integrate
import numpy as np
from matplotlib import pyplot as plt

def f(y,x):
    func = 2*(np.pi)*(np.cos(np.pi*x))*(np.cos(np.pi*y))
    
    return func

p = 5
x_nodes = bf.lobatto_quad(p)[0]
y_nodes =  bf.lobatto_quad(p)[0]

x = np.linspace(-1,1,5)
y = np.linspace(-1,1,5)
X,Y = np.meshgrid(x,y)
#Using nodal basis functions

f_exact = f(Y,X)
print(f_exact)

h_i = bf.lagrange_basis(x_nodes,x)
h_j = bf.lagrange_basis(y_nodes,y)

nodes = np.meshgrid(x_nodes,y_nodes)

f_ij = f(*nodes)



#f_i = np.array([f(y_nodes[i],x_nodes[i]) for i in range(p+1)])
#
f_h = np.einsum('ij,im,jn->mn', f_ij, h_i,h_j, optimize='optimal')
#
#
#
#
#
#plt.figure()
#cp = plt.contourf(X,Y,f_h)
#plt.colorbar(cp)
#plt.title('Discretized '+ 'p='+str(p))
#
#plt.figure()
#g = plt.contourf(X,Y,f(Y,X))
#plt.colorbar(g)
#plt.title('Exact')
#
## Using Edge basis functions
#
#def g(y,x):
#    res = 2*np.sin(np.pi*x)*np.cos(np.pi*y)
#    return res

F_ij = np.zeros((p,p))
for i in range(p):
    for j in range(p):
        F_ij[i,j] = scipy.integrate.nquad(f, 
            [[y_nodes[j],y_nodes[j+1]] , 
            [x_nodes[i], x_nodes[i+1]]] )[0]
    
e_i = bf.edge_basis(x_nodes,x)
e_j = bf.edge_basis(y_nodes,y)



#
e_recon= np.einsum('ij,im,jn->mn', F_ij, e_i,e_j, optimize='optimal')
plt.figure()
cp = plt.contourf(X,Y,e_recon)
plt.colorbar(cp)
plt.title('Discretized '+ 'p='+str(p))
#


## Vector
#
#def u(y,x):
#    return np.sin(np.pi*x)*np.cos(np.pi*y)
#
#def v(y,x):
#    return np.cos(np.pi*x)*np.sin(np.pi*y)
#
#def cof_u(y,x):
#    return (np.sin(np.pi*x)*np.sin(np.pi*y))/np.pi
#
#def cof_v(y,x):
#    return x* np.cos(np.pi*x)* np.sin(np.pi*y)
#
#
#u_i = []
#
#for i in range(len(x)):
#    u_i.append(cof_u(y[i],x[i]) - cof_u(y[i-1],x[i]))
#
#u_i = np.array(u_i)
#
#v_i =[]
#
#for i in range(len(x)):
#    v_i.append(cof_v(y[i],x[i]) - cof_u(y[i],x[i-1]))
#
#v_i = np.array(v_i)  
#
#
#u_h = np.einsum('i,ij,ik->i', u_i, h_i,e_j)
#
#v_h =  np.einsum('i,ij,ik->i', v_i, e_i,h_j)
#
#
#plt.figure()
#plt.plot(x,u_h)
#plt.plot(x,u(y,x))
#
#plt.figure()
#plt.plot(x,v_h)
#plt.plot(x,v(y,x))

