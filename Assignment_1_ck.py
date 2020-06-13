# -*- coding: utf-8 -*-
"""
Created on Apr 2019
@author: changkyu
"""
import bf_polynomials as bf
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib

IPython_default = plt.rcParams.copy()

def f1(x):
    return np.pi*(np.cos(np.pi*x))

def f2(x):
    return np.pi*(np.sin(np.pi*x))


def lagrange_1d(func,n): #n: order of quadrature
    #reduction
    xi = bf.lobatto_quad(n)[0] #Gauss-Lobatto nodes. n+1 nodes
    xe = np.linspace(-1,1,1000)        
    g_i = np.array([func(xi[i]) for i in range(n+1)])
    
    #reconstruction
    h_i = bf.lagrange_basis(xi,xe) #lagrange basis polynomials
    gh = np.einsum('i,ij->j', g_i, h_i) #multiply the two and sum over i-dimension
    g_exact = np.array([func(xe[i]) for i in range(len(xe))])
    print(g_i.shape)
    print(h_i.shape)
    
    # plt.figure()
    texpsize= [26,28,30]
    SMALL_SIZE  = texpsize[0]
    MEDIUM_SIZE = texpsize[1]
    BIGGER_SIZE = texpsize[2]
    plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    matplotlib.rcParams['lines.linewidth']  = 1.5
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor']   = 'white'
    matplotlib.rcParams["legend.fancybox"]  = False
    
    ## Graph
    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    ax[0,0].plot(xe, g_exact, label="Exact")
    ax[0,0].plot(xe, gh, label="Discretized")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    fig.savefig("lagrange_1D_deg.png", bbox_inches='tight')                                    ## Insert save destination
    
    ## If you want to see the figure, else disable last two lines.
    fig.tight_layout()
    plt.legend()
    plt.show()
    
    return

# lagrange_1d(f1,7)

def edge_1d(func,n):
    #could be improved with better integral method
    #reduction
    xi,wi = bf.lobatto_quad(n)
    #using simpsons rule
    f_i = np.array([(xi[i+1]-xi[i])/6*(func(xi[i])+4*func(0.5*(xi[i]+xi[i+1]))+func(xi[i+1])) for i in range(len(xi)-1)])
    print(np.sum(f_i))
    xe = np.linspace(-1,1,1000)
    
    #reconstruction
    e_i = bf.edge_basis(xi,xe)
    fh = np.einsum('i,ij->j', f_i, e_i)
    f_exact = func(xe)
    
    # plt.figure()
    texpsize= [26,28,30]
    SMALL_SIZE  = texpsize[0]
    MEDIUM_SIZE = texpsize[1]
    BIGGER_SIZE = texpsize[2]
    plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    matplotlib.rcParams['lines.linewidth']  = 1.5
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor']   = 'white'
    matplotlib.rcParams["legend.fancybox"]  = False
    
    ## Graph
    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    ax[0,0].plot(xe, f_exact, label="Exact")
    ax[0,0].plot(xe, fh, label="Discretized")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    fig.savefig("edge_1D_deg.png", bbox_inches='tight')                                    ## Insert save destination
    
    ## If you want to see the figure, else disable last two lines.
    fig.tight_layout()
    plt.legend()
    plt.show()
    
    return 

# edge_1d(f2, 8)


def lagrange_2d(func,p):
    x = np.linspace(-1,1,100)
    y = x #square
    xe,ye = np.meshgrid(x,y)
    f_exact = func(xe,ye)
    
    # plt.figure()
    # texpsize= [26,28,30]
    # SMALL_SIZE  = texpsize[0]
    # MEDIUM_SIZE = texpsize[1]
    # BIGGER_SIZE = texpsize[2]
    # plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    # plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    # matplotlib.rcParams['lines.linewidth']  = 1.5
    # matplotlib.rcParams['figure.facecolor'] = 'white'
    # matplotlib.rcParams['axes.facecolor']   = 'white'
    # matplotlib.rcParams["legend.fancybox"]  = False
    
    ## Graph
    # fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    # cp = ax[0,0].contour(xe,ye,f_exact)
    cp = plt.contour(xe,ye,f_exact)
    # ax[0,0].grid(True,which="major",color="#999999")
    # ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    # ax[0,0].minorticks_on()
    # ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    # ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    # ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    # fig.savefig("edge_1D_deg.png", bbox_inches='tight')           
    plt.colorbar(cp)
    
    
    # xi = bf.lobatto_quad(p)[0]
    # eta = xi #square
    
    # h_i = bf.lagrange_basis(xi,x)
    # h_j = bf.lagrange_basis(eta,y)
    
    # nodes = np.meshgrid(xi,eta)
    # f_ij = func(*nodes)
    
    # f_h = np.einsum('ij,im,jn->mn', f_ij,h_i,h_j, optimize='optimal')
    
    # texpsize= [26,28,30]
    # SMALL_SIZE  = texpsize[0]
    # MEDIUM_SIZE = texpsize[1]
    # BIGGER_SIZE = texpsize[2]
    # plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    # plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    # matplotlib.rcParams['lines.linewidth']  = 1.5
    # matplotlib.rcParams['figure.facecolor'] = 'white'
    # matplotlib.rcParams['axes.facecolor']   = 'white'
    # matplotlib.rcParams["legend.fancybox"]  = False
    
    # ## Graph
    # fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    # cpp = ax[0,0].contourf(xe,ye,f_h)
    # ax[0,0].grid(True,which="major",color="#999999")
    # ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    # ax[0,0].minorticks_on()
    # ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    # ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    # ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    # fig.savefig("edge_1D_deg.png", bbox_inches='tight')  
    # plt.colorbar(cpp)
    
    # plt.figure()
    # texpsize= [26,28,30]
    # SMALL_SIZE  = texpsize[0]
    # MEDIUM_SIZE = texpsize[1]
    # BIGGER_SIZE = texpsize[2]
    # plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    # plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    # cp = plt.contourf(xe,ye,f_h)
    # plt.colorbar(cp)
    
    # fig.tight_layout()
    plt.show()
    return

def f2(x,y):
    return 2*np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)

# lagrange_2d(f2,7)

#def simpsons2d(func,x1,x2,y1,y2):
#    return (y2-y1)/6*(func(x1,y1) + 4*func(x1, 0.5*(y1+y2)) + func(x1,y2))

def centdiff(f1,f2,dx):
    return (f2 - f1) / (2*dx)


def edge_2d(func,p):
    x = np.linspace(-1,1,100)
    y = x
    xe,ye = np.meshgrid(x,y)
    
    g_exact = func(xe,ye)    

    
    # texpsize= [26,28,30]
    # SMALL_SIZE  = texpsize[0]
    # MEDIUM_SIZE = texpsize[1]
    # BIGGER_SIZE = texpsize[2]
    # plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    # plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    # matplotlib.rcParams['lines.linewidth']  = 1.5
    # matplotlib.rcParams['figure.facecolor'] = 'white'
    # matplotlib.rcParams['axes.facecolor']   = 'white'
    # matplotlib.rcParams["legend.fancybox"]  = False
    
    # ## Graph
    # fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    # cpp = ax[0,0].contourf(xe,ye,g_exact)
    # fig.savefig("Edge_2D_discr_Exact.png", bbox_inches='tight')  
    # plt.colorbar(cpp)
    
    
    
    
    xi = bf.lobatto_quad(p)[0]
    eta = xi
    
    e_i = bf.edge_basis(xi,x)
    e_j = bf.edge_basis(eta,y)

    g_ij = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            g_ij[i,j] = scipy.integrate.nquad(func, [[xi[i], xi[i+1]], [eta[j], eta[j+1]]] )[0]
#            g_ij[i,j] = (eta[j+1]-eta[j])/6*(func(xi[i],eta[j])+4*func(xi[i],0.5*(eta[j]+eta[j+1]))+func(xi[i],eta[j+1])) 
#    for i in range(p):
#        g_ij_i = []
#        for j in range(p):                
#            g_ij_i.append((eta[j+1]-eta[j])/6*(func(xi[i],eta[j])+4*func(xi[i],0.5*(eta[j]+eta[j+1]))+func(xi[i],eta[j+1])))
#        g_ij_i = np.array(g_ij_i)
#    g_ij = np.concatenate([[g_ij_i for i in range(p)]],axis=0)
    
    g_h = np.einsum('ij,im,jn->mn', g_ij,e_i,e_j, optimize='optimal')

    
    texpsize= [26,28,30]
    SMALL_SIZE  = texpsize[0]
    MEDIUM_SIZE = texpsize[1]
    BIGGER_SIZE = texpsize[2]
    plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    matplotlib.rcParams['lines.linewidth']  = 1.5
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor']   = 'white'
    matplotlib.rcParams["legend.fancybox"]  = False
    
    ## Graph
    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    cp = ax[0,0].contourf(xe,ye,g_h)
    fig.savefig("Edge_2D_discr_deg.png", bbox_inches='tight')  
    plt.colorbar(cp)
    
    plt.show()
    return g_h




# edge_2d(f2,8)

def u(x,y):
    return -2*(np.pi)**2*np.sin(np.pi*x)*np.cos(np.pi*y)
def v(x,y):
    return -2*(np.pi)**2*np.cos(np.pi*x)*np.sin(np.pi*y)

def vector(u,v,p):
    x = np.linspace(-1,1,50)
    y = np.linspace(-1,1,51) #why 1 more index?
    X, Y = np.meshgrid(x,y, indexing='ij')
    
    # plt.figure()
    # U = plt.contourf(X, Y, u(X,Y))
    # plt.colorbar(U)
    # plt.title('Exact u')
    
    # texpsize= [26,28,30]
    # SMALL_SIZE  = texpsize[0]
    # MEDIUM_SIZE = texpsize[1]
    # BIGGER_SIZE = texpsize[2]
    # plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    # plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    # matplotlib.rcParams['lines.linewidth']  = 1.5
    # matplotlib.rcParams['figure.facecolor'] = 'white'
    # matplotlib.rcParams['axes.facecolor']   = 'white'
    # matplotlib.rcParams["legend.fancybox"]  = False
    
    # ## Graph
    # fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    # cp = ax[0,0].contourf(X, Y, u(X,Y))
    # fig.savefig("vector_u_exact.png", bbox_inches='tight')  
    # plt.colorbar(cp)
    
    # plt.figure()
    # V = plt.contourf(X, Y, v(X,Y))
    # plt.colorbar(V)
    # # plt.title('Exact v')   
    
    
    texpsize= [26,28,30]
    SMALL_SIZE  = texpsize[0]
    MEDIUM_SIZE = texpsize[1]
    BIGGER_SIZE = texpsize[2]
    plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    matplotlib.rcParams['lines.linewidth']  = 1.5
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor']   = 'white'
    matplotlib.rcParams["legend.fancybox"]  = False
    
    ## Graph
    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    cp = ax[0,0].contourf(X, Y, v(X,Y))
    fig.savefig("vector_v_exact.png", bbox_inches='tight')  
    plt.colorbar(cp)
    
#    plt.figure()
#    plt.quiver(x, y, u(X,Y).T, v(X,Y).T)
#    plt.title('Exact')


    
    nodes = bf.lobatto_quad(p)[0]
    
    h_i = bf.lagrange_basis(nodes, x)
    h_j = bf.lagrange_basis(nodes, y)
    e_i = bf.edge_basis(nodes,x)
    e_j = bf.edge_basis(nodes,y)
    
    u_ij = np.zeros((p+1, p))
    for i in range(p+1):
        for j in range(p):
            u_ij[i,j] = scipy.integrate.quad(lambda y: u(nodes[i], y), nodes[j], nodes[j+1])[0] #this fixes x value as nodes[i]
    
    u_h = np.einsum('ij,im,jn->mn', u_ij, h_i, e_j, optimize='optimal')
    
    # plt.figure()
    # U = plt.contourf(X, Y ,u_h)
    # plt.colorbar(U)
    # plt.title('Discretized u, '+ 'p='+str(p))
    
    # texpsize= [26,28,30]
    # SMALL_SIZE  = texpsize[0]
    # MEDIUM_SIZE = texpsize[1]
    # BIGGER_SIZE = texpsize[2]
    # plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    # plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    # matplotlib.rcParams['lines.linewidth']  = 1.5
    # matplotlib.rcParams['figure.facecolor'] = 'white'
    # matplotlib.rcParams['axes.facecolor']   = 'white'
    # matplotlib.rcParams["legend.fancybox"]  = False
    
    # ## Graph
    # fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    # cpp = ax[0,0].contourf(X, Y ,u_h)
    # fig.savefig("vector_u_discr_deg.png", bbox_inches='tight')  
    # plt.colorbar(cpp)
    
    v_ij = np.zeros((p,p+1))
    for i in range(p):
        for j in range(p+1):
            v_ij[i,j] = scipy.integrate.quad(lambda x: v(x, nodes[j]), nodes[i], nodes[i+1])[0]
    
    v_h = np.einsum('ij,im,jn->mn', v_ij, e_i, h_j, optimize='optimal')
# #    print(v_h.shape)

#     plt.figure()
#     V = plt.contourf(X, Y ,v_h)
#     plt.colorbar(V)
#     plt.title('Discretized v, '+ 'p='+str(p))

    texpsize= [26,28,30]
    SMALL_SIZE  = texpsize[0]
    MEDIUM_SIZE = texpsize[1]
    BIGGER_SIZE = texpsize[2]
    plt.rc('font', size=MEDIUM_SIZE)                    ## controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
    matplotlib.rcParams['lines.linewidth']  = 1.5
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor']   = 'white'
    matplotlib.rcParams["legend.fancybox"]  = False
    
    ## Graph
    fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    cpp = ax[0,0].contourf(X, Y ,v_h)
    fig.savefig("vector_v_discr_deg.png", bbox_inches='tight')  
    plt.colorbar(cpp)

    # plt.figure()
    # plt.quiver(x, y, u_h.T, v_h.T)
    # plt.title('Discretized, '+ 'p='+str(p))
    # return u_h, v_h

vector(u,v,5)