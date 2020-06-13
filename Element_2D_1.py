# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:27:21 2019

@author: sanjay
"""


import numpy as np
import bf_polynomials as bf
from matplotlib import pyplot as plt
import scipy.integrate as scin
import Tools as tl
import math



#  Incidence Matrix 

def incidence_matrix_2d(p):
    qy = np.zeros((p*p,p*(p+1)))
    np.fill_diagonal(qy, -1)
    np.fill_diagonal(qy[:,p:],1)        
    
    ones = np.eye(p) # qx part of the matrix
    subqx = np.zeros((p,(p+1)))
    np.fill_diagonal(subqx, -1)
    np.fill_diagonal(subqx[:,1:],1)
    qx = np.kron(ones, subqx) 
    
    E = np.hstack([qx,qy])
    
    return E


# General Parameters
    
def solver_2d(p,domain_x,domain_y,E,f):

    x_nodes = bf.lobatto_quad(p)[0]
    y_nodes = bf.lobatto_quad(p)[0]
    
    x_int,w_int_x = bf.gauss_quad(p+10)
    y_int,w_int_y = bf.gauss_quad(p+10)
    
    int_X = np.kron(w_int_x,w_int_x)
    int_Y = np.kron(w_int_y,w_int_y)
    
    # Domain and Transformations
    
    a,b = domain_x
    c,d = domain_y
    
    
    x_mapped_nodes = 0.5*(1-x_nodes)*a + 0.5*(1+x_nodes)*b
    y_mapped_nodes = 0.5*(1-y_nodes)*c + 0.5*(1+y_nodes)*d
    
    hx = b-a
    hy= d-c
    
    
    # Source term Reduction
    
    F = []
    
    for j in range(p): 
        for i in range(p):         
           f_ij = scin.nquad(f, [[x_mapped_nodes[i], x_mapped_nodes[i+1]],[y_mapped_nodes[j], y_mapped_nodes[j+1]]])[0]
           F.append(f_ij)
    
    F = np.array(F)
    
    b1 = np.zeros((2*p*(p+1)))
    
    B = np.concatenate((b1,F))
    
    # Mass Matrix
    
    h_i,e_i,h_j,e_j = tl.basis_functions(p,p,None,x_int,y_int,1) #i: x direction; j: y direction
    
    
    b_x = np.kron(e_j,h_i)
    b_y = np.kron(h_j,e_i)
    
    Mx =  np.einsum('iw,jw,w ->ij',b_x,b_x,int_X,optimize='optimal')*hx/hy
    My =  np.einsum('iw,jw,w ->ij',b_y,b_y,int_Y,optimize='optimal')*hy/hx
    
    Me = np.zeros((p*(p+1),p*(p+1)))
    
    M = np.block([[Mx, Me],[Me,My]])
            
        
#    # Solution
#    
    a1 = np.zeros((E.shape[0],E.shape[0]))
        
    A = np.block([[M,E.T],[E,a1]])
    

    return A,B

# Reconstruction

def reconstruct(u,p,u_x,u_y,n,domain_x,domain_y,Element,plot):
    a,b = domain_x
    c,d = domain_y
    x = np.linspace(a,b,n)
    y = np.linspace(c,d,n)
    X,Y = np.meshgrid(x,y)
    
    hx = b-a
    hy= d-c
    
    h_i2,e_i2,h_j2,e_j2 = tl.basis_functions(p,p,'Df',x,y,n)
    
    
    s_x = np.kron(e_j2,h_i2)
    s_y = np.kron(h_j2,e_i2)
    
    
    q_x = np.einsum('i, im -> m',u_x,s_x)*(2/hy)
    
    q_x = np.reshape(q_x,(np.size(x),np.size(y)),order='C')
    
    q_y = np.einsum('i, im -> m',u_y,s_y)*(2/hx)
    
    q_y = np.reshape(q_y,(np.size(x),np.size(y)),order='C')
    
    u_x_exact = u(X,Y)[0]
    u_y_exact = u(X,Y)[1]
        
    # plot
    if plot==True:
     
        fig = plt.figure()
        fig.suptitle('P = '+str(p) + '    Element No.='+ str(Element))
        
        plt.subplot(321) 
        cp = plt.contourf(X,Y,u_x_exact)
        plt.colorbar(cp)
        plt.title('U_x Exact')
        
        
        plt.subplot(322)
        cp =plt.contourf(X,Y,q_x)
        plt.colorbar(cp)
        plt.title('U_x Approx')
        
        plt.subplot(323)
        cp =plt.contourf(X,Y,u_y_exact)
        plt.colorbar(cp)
        plt.title('U_y Exact')
        
        plt.subplot(324)
        cp =plt.contourf(X,Y,q_y)
        plt.colorbar(cp)
        plt.title('U_y Approx')
        
        plt.subplot(325)
        cp =plt.contourf(X,Y,(u(X,Y)[0]-q_x))
        plt.colorbar(cp)
        plt.title('Difference U_x')
        
        plt.subplot(326)
        cp =plt.contourf(X,Y,(u(X,Y)[1]-q_y))
        plt.colorbar(cp)
        plt.title('Difference U_y')
    
    return q_x,q_y,u_x_exact,u_y_exact,X,Y

    


def inner_elem(N):
    all_elm = np.arange(N)
    res = []
    check = int(np.sqrt(N))
    
    if check <=2:
        return res
    
    else:
        check_2 = all_elm[:check]
        cand = check_2[1:check-1]
        res = list(check_2[1:check-1])
        for i in range(1,check):
            for j in range(len(cand)):
                res.append(cand[j] + (i*check))
        
        return res

def C_gathering_x(all_elements,K_x,K_y,Global_num,p):
    count = []
    already_done = []
    for g in range(K_x):
        for h in range(K_y):
            current = all_elements[g,h]
            next_elm = current + 1
            previous = current - 1
            for t in range(p):
                edge_1 = p+t*(p+1)
                edge_2 = t*(p+1)
                if h< K_y-1:
                    if Global_num[edge_1,current] not in already_done:
                        count.append([Global_num[edge_1,current],Global_num[edge_2,next_elm]])
                        already_done.append(Global_num[edge_1,current])
                        already_done.append(Global_num[edge_2,next_elm])
                elif h==K_y-1 and next_elm in all_elements:
                    if Global_num[edge_1,previous] not in already_done:
                        count.append([Global_num[edge_1,previous],Global_num[edge_2,current]])
                        already_done.append(Global_num[edge_1,previous])
                        already_done.append(Global_num[edge_2,current])
                else:
                    continue
    
    return count

    

def C_gathering_y(all_elements,K_x,K_y,Global_num,p):
    count = []

    for g in range(K_x):
        for h in range(K_y):
            current = all_elements[g,h]
            next_elm = current + K_x
            previous = current - K_x
            for t in range(p):
                edge_1 = p*(p+1) + (p**2) + t 
                edge_2 = p*(p+1) + t
                if g< K_y-1:
                    count.append([Global_num[edge_1,current],Global_num[edge_2,next_elm]])
                elif g==K_y-1 and next_elm in all_elements:
                    count.append([Global_num[edge_1,previous],Global_num[edge_2,current]])
                else:
                    continue
    
    return count
        

def C_gather(C,x_edges_num,y_edges_num,numb_lmbd):
    for s in range(numb_lmbd):
        numb = int(numb_lmbd)/2
        if s < numb:
            g = s
            check = x_edges_num
        else:
            g = int(s- numb)
            check = y_edges_num
        
        edge_1 = check[g][0]
        edge_2 = check[g][1]
        C[s,edge_1] =  -1
        C[s,edge_2] = 1     
        
    return C




def Vand_matrix(x_nodes,N):
    A_matrix = np.ones((N,N))
    for i in range(1,N):
        A_matrix[i,:] = x_nodes**i
    
    return A_matrix
    
def intg_weights(x_nodes,N):
    
    
    B_matrix = np.ones((N,1))
    x_0 = x_nodes[0]
    x_1 = x_nodes[-1]
    A_inv = np.linalg.inv(Vand_matrix(x_nodes, N))
    for k in range(1,N+1):
              
        B_matrix[k-1] = ((x_1**k) - (x_0**k))/k
    
    weights = (A_inv.dot(B_matrix)).T
    
    return weights[0]

def integ_1d(f_x_data,w_x):
    
    integ = np.dot(f_x_data,w_x)
    
    return integ

def integ_2d(f_data,w_x,w_z):
    
    weights = np.stack((w_x,w_z))
    integ = f_data.dot(weights[0,:]).dot(weights[1,:])
    
    return integ

def linear_transform(x_nodes,w,c,d):
    """
    x_nodes : standard domain
    c,d : desired domain
    
    output:
    x_new_nodes = transformed nodes to c,d domain
    w_new = transformed weights
    """
    
    x_new_nodes = ((d-c)/2)*(x_nodes+1) + c
    w_new = (d-c)*0.5*w
    
    return x_new_nodes,w_new

#Error
    
def L2_error(u,p,u_x,u_y,n,domain_x,domain_y,Element,plot):
    a,b = domain_x
    c,d = domain_y
    x_org = np.linspace(a,b,p+12)
    y_org = np.linspace(c,d,p+12)
    x = bf.lobatto_quad(p+12)[0]
    y = x
    # print(len(x))
    X,Y = np.meshgrid(x_org,y_org)
    
    hx = b-a
    hy= d-c
    # x_nodes = bf.lobatto_quad(p)[0]
    # y_nodes = x_nodes
    
    # h_i2 = bf.lagrange_basis(x_nodes,x)
    # e_i2 = bf.edge_basis(x_nodes,x)
    
    # h_j2 = bf.lagrange_basis(y_nodes,y)
    # e_j2 = bf.edge_basis(y_nodes,y)
    h_i2,e_i2,h_j2,e_j2 = tl.basis_functions(p,p,"Df",x,y,p+12)
    
    
    s_x = np.kron(e_j2,h_i2)
    s_y = np.kron(h_j2,e_i2)
    
    
    q_x = np.einsum('i, im -> m',u_x,s_x)*(2/hy)
    
    q_x = np.reshape(q_x,(np.size(x_org),np.size(y_org)),order='C')

    
    # print(len(s_x))
    # print(len(s_y))
    # print(len(u_x))
    # print(len(u_y))
    
    
    # q_y = np.einsum('i, im -> m',u_y,s_y)*(2/hx)
    
    # q_y = np.reshape(q_y,(np.size(x),np.size(y)),order='C')
    
    u_x_exact = u(X,Y)[0]
    # u_y_exact = u(x,y)[1]
        
    # w_int = intg_weights(x_org, p+10)
    
    
    try_nodes = np.linspace(-1,1,p+12)
    w_int = intg_weights(try_nodes, p+12)

    error_x = (u_x_exact - q_x)**2
    
    new_nodes,new_w = linear_transform(try_nodes,w_int,a,b)
    err_intg = integ_2d(error_x,new_w, new_w)
    # error_y = (u_y_exact - q_y)**2   
    
    # fig = plt.figure()
    # fig.suptitle('P = '+str(p) + '    Element No.='+ str(Element))
    
    # # plt.subplot(221) 
    # cp = plt.contourf(X,Y,u_x_exact)
    # plt.colorbar(cp)
    # plt.title('U_x Exact')
    
    
    # plt.subplot(222)
    # cp =plt.contourf(X,Y,q_x)
    # plt.colorbar(cp)
    # plt.title('U_x Approx')
    
    # plt.subplot(223)
    # cp =plt.contourf(X,Y,u_x_exact-q_x)
    # plt.colorbar(cp)
    # plt.title('Diff')
    
    return error_x, err_intg

