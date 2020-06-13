import bf_polynomials as bf
import numpy as np

import matplotlib.pyplot as plt

#1-D

p = 12 #order of quadrature

#gauss-lobatto nodes
xi = bf.lobatto_quad(p)[0]
#
##------------lagrange-------------
#g_i = np.array([np.sin(np.pi*i) for i in xi])
#
#x = np.linspace(-1,1,100)
##lagrange basis functions
#h_i = bf.lagrange_basis(xi,x)
##for x=None, checked for identity matrix
#
#print(g_i)
#
#gh = np.einsum('i,ij->j', g_i, h_i)
##gh = np.sum(g_i[:, np.newaxis] * h_i, axis=0)
#
##print(gh)
#
#g_exact = np.sin(np.pi*x)
##print(g_exact)
#
#plt.figure()
#plt.plot(x, g_exact)
#plt.plot(x, gh)

#-------------edge-----------------

#edges
edg = []
for i in range(1,len(xi)):
    diff = xi[i] - xi[i-1]
    edg.append(diff)
edg = np.array(edg) 
#the x coordinates are the difference between consecutive lobatto nodes

def g(xi):
    return np.sin(np.pi*xi) #test function


#coefficients
#for i in range(1,len(xi))
f_i = g(xi)[1:] - g(xi)[0:-1] #diff between consecutive y coordinates
print(f_i)
#integral

xe = np.linspace(-1,1,1000) #testing x values

#edge basis functions
e_i = bf.edge_basis(xi,xe) #edge polynomials values with given nodes

fh = np.einsum('i,ij->j', f_i, e_i)
print(fh)

f_exact = np.pi*np.cos(np.pi*xe)

plt.figure()
plt.plot(xe, f_exact)
plt.plot(xe, fh)