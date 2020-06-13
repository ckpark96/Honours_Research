# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:10:42 2019

@author: chang
"""

import Assignment_2_ck as as2
import numpy as np
import bf_polynomials as bf
import matplotlib.pyplot as plt

def f(x):
    return np.heaviside(x,0)

def L2(f,n):
    #reduction 
    
    xi = bf.lobatto_quad(n)[0]
    
    fi = np.zeros(n)
    for i in range(n):
        fi[i]= as2.numerical_integration_1d(f, n+3, a=xi[i], b= xi[i+1])
        
#    x,w = bf.lobatto_quad(n+100)
    
    xplot = np.linspace(-1,1,10)
    #reconstruction
    #edge basis functions
    e_i = bf.edge_basis(xi,xplot)
    #print(e_i)
    
    fh = np.einsum('i,ij->j', fi, e_i)
    #fh = sum(sum(np.multiply(f_i,e_i.T)))
    #print(fh)
    
    
    f_exact = f(xplot)
    
    plt.figure()
#    plt.plot(x, abs(f_exact - fh)
    plt.plot(xplot,fh)
    plt.plot(xplot, f_exact)
    
    
    return
#    return np.sqrt(np.sum((f_exact - fh)**2*w)) #L2 error

print(L2(f,6))

ei = np.zeros(19)
#for i in range(1,20):
#    ei[i-1] = L2(f,i)

#plt.figure()
#plt.plot(range(1,20),np.log10(ei))

