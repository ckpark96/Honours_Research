import bf_polynomials as bf
import numpy as np
import networkx as nx



def incidence_matrix(p):

    edges = []
    nodal_pt = list(range(p+1))
    for e in range(p):
        edges.append([e,e+1])
    
    G = nx.MultiGraph()
    G.add_nodes_from(nodal_pt)
    G.add_edges_from(edges)
    
    E = nx.incidence_matrix(G,oriented=True) # this returns a scipy sparse matrix
    E = np.transpose(E.toarray())
    
    return E

def mapping(xi, a, b):
    return 0.5*(1-xi)*a+0.5*(1+xi)*b




def element_matrix(f,p,quad_deg,a,b):
    x_int_unmapped,w_int = bf.gauss_quad(quad_deg)
    nodes = bf.lobatto_quad(p)[0]
    
    x_int = 0.5*(1-x_int_unmapped)*a+0.5*(1+x_int_unmapped)*b
#    w_int = w_int*(b-a)*0.5
    
    dxdxi = (b-a)*0.5
    
    
#    nodes = 0.5*(1-nodes)*a+0.5*(1+nodes)*b
    
    h_i = bf.lagrange_basis(nodes,x_int_unmapped)
    e_i = bf.edge_basis(nodes,x_int_unmapped)
    
    f_i = np.zeros(p)
    
    segment_length = nodes[1:] - nodes[:-1]
    mapped_nodes = 0.5*(1-nodes)*a+0.5*(1+nodes)*b
    x_int = 0.5*(1-x_int_unmapped)*a+0.5*(1+x_int_unmapped)*b
    x_int -= x_int[0]
    for i in range(p):
        x_int_i = x_int * segment_length[i] / 2
        x_int_i = x_int_i + mapped_nodes[i] 
        
    #    f_i[i] = np.sum(w_int*f(0.5*(1-x_int)*a+0.5*(1+x_int)*b)*(b-a)*0.5*e_i[i,:])
        f_i[i] = np.sum((mapped_nodes[i+1]-mapped_nodes[i])*0.5*w_int*f(x_int_i))
        
        # print(x_int_i)
    M_u = (dxdxi)* np.einsum("ik, jk, k -> ij", h_i, h_i, w_int, optimize="optimal")
    M_phi =(1/dxdxi)* np.einsum("ik, jk, k -> ij", e_i, e_i, w_int, optimize="optimal")
#    print(M_phi)
#    print(M_u)
    #Incidence Matrix
    
    E = incidence_matrix(p)
    
    N = np.zeros((2,p+1))
    
    N[0,0] = 1
    N[1,-1] = -1
    
    A = np.block([[M_u, np.transpose(M_phi@E),np.transpose(N)],
                   [M_phi@E, np.zeros((p,p)), np.zeros((p,2))],
                   [N, np.zeros((2,p)), np.zeros((2,2))]])

    B = np.concatenate((np.zeros(p+1),-M_phi@f_i,np.zeros(2))) 
    
    return A,B


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def incidence_matrix_2d(p_x,p_y):
#    
#
    edges = []
    extra = []
    nodal_pt = list(range(p_x*p_y))
    for g in range(p_y):
        for e in range(p_x-1):
                edges.append([e+g*p_y,e+g*p_y+1])
                extra.append([e+g*p_y+1,e+g*p_y])
    
    for m in range(p_y-1):
        for l in range(p_x):
            edges.append([l+m*p_y,l+(m+1)*p_y])
            extra.append([l+(m+1)*p_y,l+m*p_y])
            
    
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodal_pt)
    G.add_edges_from(edges)
#    G.add_edges_from(extra)
    
    E = nx.incidence_matrix(G,oriented=True) # this returns a scipy sparse matrix
    E = np.transpose(E.toarray())
    
#    nx.draw(G)
#    print((G.edges()))
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