import numpy as np
import scipy as sp
import bf_polynomials as bf
from matplotlib import pyplot as plt
import scipy.integrate as scin
plt.rcParams.update({'figure.max_open_warning': 0}) # prevents max_open_warning 


#### Exact Solutions and Source Term

def phi(x,y):
    return np.sin(np.pi*x)*np.sin(2*np.pi*y)

def u(x,y):
    u_x = np.pi*np.cos(np.pi*x)*np.sin(2*np.pi*y) # dphi/dx
    u_y = 2*np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*y) # dphi/dy
    return u_x, u_y

def f(x,y):
    return -np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y) \
        - 4*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y)


####  Incidence Matrix 

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

def basis_functions(px,py,d,x,y,n):
    x_nodes = bf.lobatto_quad(px)[0]
    y_nodes = bf.lobatto_quad(py)[0]
    
    if d=='Df':
        x,y = np.linspace(-1,1,n),np.linspace(-1,1,n)
    elif d ==None:
        x=x
        y=y
    
    h_i = bf.lagrange_basis(x_nodes,x)
    e_i = bf.edge_basis(x_nodes,x)
    
    h_j = bf.lagrange_basis(y_nodes,y)
    e_j = bf.edge_basis(y_nodes,y)

    return h_i,e_i,h_j,e_j

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
    
    h_i,e_i,h_j,e_j = basis_functions(p,p,None,x_int,y_int,1) #i: x direction; j: y direction
    
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
    
    h_i2,e_i2,h_j2,e_j2 = basis_functions(p,p,'Df',x,y,n)
    
    
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
        
    # Diff_x = u(X,Y)[0]-q_x
    # Diff_y = u(X,Y)[1]-q_y
    
    return q_x, q_y

#### finds the element pairs for x-directional flux that are in 'contact' with each other which the pair is basically representing the same thing

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
        if s < numb: # x edges
            g = s
            check = x_edges_num
        else: # y edges
            g = int(s - numb)
            check = y_edges_num
        
        # mark the paired edges 1 and -1
        edge_1 = check[g][0]
        edge_2 = check[g][1]
        C[s,edge_1] =  -1
        C[s,edge_2] = 1     
        
    return C

p = 2 # Polynomial degree

K_x = K_y = 2 # No. of elements per axis
K = K_x * K_y # Total no. of elements

X_dim = (-1,1) # x-coordinate range
Y_dim = (-1,1) # y-coordinate range

h_x = (X_dim[1]-X_dim[0])/(K_x) # x directional size of each element
h_y = (Y_dim[1]-Y_dim[0])/(K_y) # y directional size of each element

E = incidence_matrix_2d(p)

#### Assembling Global A matrix (LHS) and b matrix (RHS)

matrix_list_A = []
matrix_list_B = []
domain = []

for j in range(K_y):
    domain_y = (j*h_y,(j+1)*h_y)
    for i in range(K_x):
        domain_x = (i*h_x,(i+1)*h_x)
        # print("domain_y is ", domain_y)
        # print("domain_x is ", domain_x)
        
        domain.append([domain_x,domain_y])
        A,B = solver_2d(p,domain_x,domain_y,E,f)
        print(B)
        matrix_list_A.append(A)
        matrix_list_B.append(B)


big_A = sp.linalg.block_diag(*matrix_list_A)

# Connectivity Matrix (C matrix)

GLobal_num = np.reshape(np.arange(K*(2*p*(p+1) + p**2)),((2*p*(p+1))+p**2,K),order='F') # Numering edges and phi's as column vectors

all_elms = np.reshape(np.arange(K),(K_x,K_y)) # Numbering of elements

num_of_lmbd = ((K_x-1)*K_y + (K_y-1)*K_x)*p  # Numbering of lambdas

C_1 = np.zeros((num_of_lmbd,K*((p*2*(p+1))+p**2))) # preliminary C matrix with only 0s (for the shape)

x_edges_c = C_gathering_x(all_elms,K_x,K_y,GLobal_num,p)

y_edges_c = C_gathering_y(all_elms,K_x,K_y,GLobal_num,p)          

C_final = C_gather(C_1,x_edges_c,y_edges_c,num_of_lmbd)

Big_b = np.concatenate(matrix_list_B)

fill_b = np.zeros((num_of_lmbd))

fill = np.zeros((num_of_lmbd,num_of_lmbd))

Final_A = np.block([[big_A,C_final.T],[C_final,fill]])

Final_B = np.concatenate((Big_b,fill_b))


# Solve system and reconstruction 

soln = np.linalg.solve(Final_A,Final_B)


## Results
L2_error_list = []
for h in range(K):
    ux = soln[GLobal_num[0,h]:GLobal_num[p*(p+1),h]]
    uy = soln[GLobal_num[p*(p+1),h]:GLobal_num[2*p*(p+1),h]]
    x_dom = domain[h][0] 
    y_dom = domain[h][1]
    qx, qy = reconstruct(u,p,ux,uy,100,x_dom,y_dom,h,plot=False)
    
    # L2_error_list.append(L2_error_squared)

# L2_error = np.sqrt(np.sum(L2_error_list))