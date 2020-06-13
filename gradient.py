# -*- coding: utf-8 -*-
"""
Created on May 2019
@author: changkyu
"""
import bf_polynomials as bf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import Assignment_1_Changkyu_corrected as as1

def centdiff(f1,f2,dx):
    return (f2 - f1) / (2*dx)

def disc2grad(g_h, p, n):
    x = np.linspace(-1,1,n)
    y = x
    u_h = np.zeros((len(x)-1, len(y)-1))

    dx = x[1]-x[0]
    for j in range(len(y)-1):
        for i in range(1,len(x)-1):
            u_h[i,j] = centdiff(g_h[i-1,j], g_h[i+1,j], dx)
            
    dy = y[1]-y[0]
    v_h = np.zeros((len(x)-1, len(y)-1))
    for i in range(len(x)-1):
        for j in range(1,len(y)-1):
            v_h[i,j] = centdiff(g_h[i,j-1], g_h[i,j+1], dy)
            
    xg = []    
    yg = []
    for i in range(len(x)-1):
        xg.append((x[i+1]+x[i])/2)
    for j in range(len(y)-1):
        yg.append((y[j+1]+y[j])/2)
    
    XG,YG = np.meshgrid(np.array(xg),np.array(yg))
    
    plt.figure()
    uh = plt.contourf(XG,YG,u_h)
    plt.colorbar(uh)
    plt.title('u_h Discretized->grad, '+ 'p='+str(p))  

    plt.figure()
    vh = plt.contourf(XG,YG,v_h)
    plt.colorbar(vh)
    plt.title('v_h Discretized->grad, '+ 'p='+str(p))   
    
disc2grad(as1.lagrange_2d(as1.f2,5), 5,100)

def grad2disc(func,p,n):
    
    x = np.linspace(-1,1,n)
    y = np.linspace(-1,1,n+1)
    
    u = np.zeros((len(x)-1, len(y)-2))
    dx = x[1]-x[0]
    for j in range(len(y)-1):
        for i in range(1,len(x)-1):
            u[i,j] = centdiff(func(x[i-1],y[j]), func(x[i+1],y[j]), dx)
    
    v = np.zeros((len(x)-1, len(y)-2))
    dy = y[1]-y[0]
    for i in range(len(x)-1):
        for j in range(1,len(y)-2):
            v[i,j] = centdiff(func(x[i],y[j-1]), func(x[i],y[j+1]), dy)


    nodes = bf.lobatto_quad(p)[0]
    
    h_i = bf.lagrange_basis(nodes, x)
    h_j = bf.lagrange_basis(nodes, y)
    e_i = bf.edge_basis(nodes,x)
    e_j = bf.edge_basis(nodes,y)
    
    u_ij = np.zeros((p+1, p))
    for i in range(p+1):
        for j in range(p):
            u_ij[i,j] = (eta[j+1]-eta[j])/6*(func(xi[i],eta[j])+4*func(xi[i],0.5*(eta[j]+eta[j+1]))+func(xi[i],eta[j+1])) 
#            u_ij[i,j] = scipy.integrate.quad(lambda y: u(nodes[i], y), nodes[j], nodes[j+1])[0] #this fixes x value as nodes[i]
#    
#    u_h = np.einsum('ij,im,jn->mn', u_ij, h_i, e_j, optimize='optimal')
#    
#    plt.figure()
#    U = plt.contourf(X, Y ,u_h)
#    plt.colorbar(U)
#    plt.title('Discretized u, '+ 'p='+str(p))
#    
#    v_ij = np.zeros((p,p+1))
#    for i in range(p):
#        for j in range(p+1):
#            v_ij[i,j] = scipy.integrate.quad(lambda x: v(x, nodes[j]), nodes[i], nodes[i+1])[0]
#    
#    v_h = np.einsum('ij,im,jn->mn', v_ij, e_i, h_j, optimize='optimal')
#
#    plt.figure()
#    V = plt.contourf(X, Y ,v_h)
#    plt.colorbar(V)
#    plt.title('Discretized v, '+ 'p='+str(p))
#    
#    
    return
#
#grad2disc(as1.f2,5,100)
