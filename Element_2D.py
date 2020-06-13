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

    x_nodes = bf.lobatto_quad(p)[0] # outputs p+1 number of nodes
    y_nodes = bf.lobatto_quad(p)[0]
    
    x_int,w_int_x = bf.gauss_quad(p+1) # outputs p+1 number of nodes for integral
    y_int,w_int_y = bf.gauss_quad(p+1)
    
    int_X = np.kron(w_int_x,w_int_x) # Kronecker product
    int_Y = np.kron(w_int_y,w_int_y)
    
    #### Domain and Transformations
    
    a,b = domain_x
    c,d = domain_y
    
    x_mapped_nodes = 0.5*(1-x_nodes)*a + 0.5*(1+x_nodes)*b  # map from lobatto domain (-1,1) into (a,b)
    y_mapped_nodes = 0.5*(1-y_nodes)*c + 0.5*(1+y_nodes)*d
    
    hx = b-a
    hy= d-c
    
    #### Source term Reduction
    
    F = []
    
    for j in range(p): 
        for i in range(p):         
           f_ij = scin.nquad(f, [[x_mapped_nodes[i], x_mapped_nodes[i+1]],[y_mapped_nodes[j], y_mapped_nodes[j+1]]])[0] # integrate over looping integral ranges
           F.append(f_ij)
    
    F = np.array(F) # array of integral values over every element
    
    b1 = np.zeros((2*p*(p+1))) # RHS top rows of 0
    
    B = np.concatenate((b1,F))
    
    #### Mass Matrix
    
    h_i,e_i,h_j,e_j = tl.basis_functions(p,p,None,x_int,y_int,1) #i: x direction; j: y direction
    
    b_x = np.kron(e_j,h_i)
    b_y = np.kron(h_j,e_i)
    
    Mx =  np.einsum('iw,jw,w ->ij',b_x,b_x,int_X,optimize='optimal')*hx/hy
    My =  np.einsum('iw,jw,w ->ij',b_y,b_y,int_Y,optimize='optimal')*hy/hx
    
    Me = np.zeros((p*(p+1),p*(p+1)))
    M = np.block([[Mx, Me],[Me,My]])
        
    #### Solution
   
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
    hy = d-c
    
    h_i2,e_i2,h_j2,e_j2 = tl.basis_functions(p,p,'Df',x,y,n)
    
    
    s_x = np.kron(e_j2,h_i2)
    s_y = np.kron(h_j2,e_i2)
    
    
    q_x = np.einsum('i, im -> m',u_x,s_x)*(2/hy)
    
    q_x = np.reshape(q_x,(np.size(x),np.size(y)),order='C')
    
    q_y = np.einsum('i, im -> m',u_y,s_y)*(2/hx)
    
    q_y = np.reshape(q_y,(np.size(x),np.size(y)),order='C')
    
    
    # plot
    if plot==True:
     
        fig = plt.figure()
        fig.suptitle('P = '+str(p) + '    Element No.='+ str(Element))
        
        plt.subplot(321) 
        cp = plt.contourf(X,Y,u(X,Y)[0])
        plt.colorbar(cp)
        plt.title('U_x Exact')
        
        plt.subplot(322)
        cp =plt.contourf(X,Y,q_x)
        plt.colorbar(cp)
        plt.title('U_x Approx')
        
        plt.subplot(323)
        cp =plt.contourf(X,Y,u(X,Y)[1])
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
        
    # x_int, w_int_x = bf.gauss_quad(p+1) # outputs p+1 number of nodes for integral
    # y_int, w_int_y = bf.gauss_quad(p+1)
    # print('w_int_x is', w_int_x)
    Diff_x = u(X,Y)[0]-q_x
    Diff_y = u(X,Y)[1]-q_y
    print("Diff_x =", Diff_x)
    print("Shape of Diff_x =", np.shape(Diff_x))
        
    # L2_error_x_squared = np.sum(Diff_x ** 2 * w_int_x * w_int_y)
    # L2_error_y_squared = np.sum(Diff_y ** 2 * w_int_x * w_int_y)
    # L2_error_squared = L2_error_x_squared + L2_error_y_squared
    
    return q_x, q_y


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

# finds the element pairs for x-directional flux that are in 'contact' with each other which the pair is basically representing the same thing
def C_gathering_x(all_elements,K_x,K_y,Global_num,p):
    count = []
    already_done = []
    for g in range(K_x):
        for h in range(K_y):
            current = all_elements[g,h]
            # print('current is', current)
            next_elm = current + 1
            previous = current - 1
            for t in range(p):
                edge_1 = p+t*(p+1)
                edge_2 = t*(p+1)
                # print('edge 1 is', edge_1)
                # print('edge 2 is', edge_2)
                if h< K_y-1:
                    if Global_num[edge_1,current] not in already_done:
                        # print('Global num [edge1 curent] is', Global_num[edge_1,current])
                        # print('Global num [edge2 next elm] is', Global_num[edge_2,next_elm])
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
    # print('x count is', count)
    return count

    
# same but for y-directional flux
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
    # print('y count is', count)
    return count
        
# finds where to insert 1 or -1 in the C matrix
def C_gather(C,x_edges_num,y_edges_num,numb_lmbd):
    # print('numb_lmbd is', numb_lmbd)
    for s in range(numb_lmbd):
        # print('s is', s)
        numb = int(numb_lmbd)/2
        # print('numb is', numb)
        if s < numb:
            g = s
            check = x_edges_num
        else:
            g = int(s - numb)
            check = y_edges_num
        
        edge_1 = check[g][0]
        edge_2 = check[g][1]
        C[s,edge_1] =  -1
        C[s,edge_2] = 1     
        
    return C
        
        
    