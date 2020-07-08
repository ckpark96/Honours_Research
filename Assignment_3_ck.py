import numpy as np
import matplotlib
import bf_polynomials as bf
from matplotlib import pyplot as plt

# def f(x,alpha):
    # return (alpha**2)*((np.pi)**2)*np.sin(alpha*np.pi*x)
    
def f(x):
    return np.sin(np.pi*x)

def func_u(x):
    return 1 / np.pi * np.cos(np.pi*x)

def func_phi(x):
    return 1 / (np.pi**2) * np.sin(np.pi*x)

def poisson_homogen(func, p, alpha):
    
    x_int,w_int = bf.gauss_quad(p+5)
    nodes = bf.lobatto_quad(p)[0]   
    
    e_i = bf.edge_basis(nodes,x_int)
    e_j = bf.edge_basis(nodes,x_int)
    
    h_i = bf.lagrange_basis(nodes,x_int)
    h_j = bf.lagrange_basis(nodes,x_int)
    
    #source matrix
    f_i = np.zeros(p)
    for i in range(p):
        # f_i[i] = np.sum(w_int*func(x_int,alpha)*e_i[i,:])
        f_i[i] = np.sum(w_int*func(x_int)*e_i[i,:])
        
    
    #mass matrices
    Mu = np.einsum("ik, jk, k -> ij", h_i, h_j, w_int, optimize="optimal")
    Mphi = np.einsum("ik, jk, k -> ij", e_i, e_j, w_int, optimize="optimal")
    
    #building incidence matrix
    E = np.zeros((p,p))
    np.fill_diagonal(E,-1)
    np.fill_diagonal(E[:,1:], 1)
    extracol = np.zeros((p,1))
    extracol[-1,:] = 1
    E = np.hstack((E,extracol))
    
    
    #building the main matrix equation
    MphiE = Mphi.dot(E)
    emptysq = np.zeros((p,p))
    emptycol = np.zeros((p+1,1))
    
    K = np.hstack((np.vstack((Mu,MphiE)), np.vstack((MphiE.T,emptysq))))
    f_i = f_i.reshape((1,p))
    F = np.vstack((emptycol,-1*np.transpose(f_i)))
    
    #solution
    soln = np.linalg.solve(K,F)
    u = soln[:p+1].T
    phi = soln[p+1:].T
    
    #reconstruction
    x = np.linspace(-1,1,100)
    h_ix = bf.lagrange_basis(nodes,x)
    u_h = np.einsum('ij,jk->k', u, h_ix, optimize='optimal')
    e_ix = bf.edge_basis(nodes,x)
    phi_h = np.einsum('ij,jk->k', phi, e_ix, optimize='optimal')
    
    # plt.figure()
    # plt.title('u')
    # plt.plot(x,u_h,label='approx')
    # plt.plot(x,alpha*np.pi*np.cos(alpha*np.pi*x),label='exact')
    # plt.legend()
    
    # plt.figure()
    # plt.title('phi')
    # plt.plot(x,phi_h,label='approx')
    # plt.plot(x,np.sin(alpha*np.pi*x),label='exact')
    # plt.legend()
    # plt.show()
    
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
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel(r'$\phi$')
    ax[0,0].plot(x, func_phi(x), label="Exact")
    ax[0,0].plot(x, phi_h, label="Discretized")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    fig.savefig("1D_Poisson_Homogeneous_deg.png", bbox_inches='tight')                                    ## Insert save destination
    
    ## If you want to see the figure, else disable last two lines.
    fig.tight_layout()
    plt.legend()
    plt.show()
    
    return E, K
    
    
# poisson_homogen(f,6,1)

def poisson_nonhomogen(func, p):
    
    x_int,w_int = bf.gauss_quad(p+5)
    nodes = bf.lobatto_quad(p)[0]   
    
    e_i = bf.edge_basis(nodes,x_int)
    e_j = bf.edge_basis(nodes,x_int)
    
    h_i = bf.lagrange_basis(nodes,x_int)
    h_j = bf.lagrange_basis(nodes,x_int)
    
    #source matrix
    # f_i = np.zeros(p)
    # for i in range(p):
    #     # f_i[i] = np.sum(w_int*func(x_int,alpha)*e_i[i,:])
    #     f_i[i] = np.sum(w_int*func(x_int)*e_i[i,:])
    up_i = np.zeros(p+1)
    for i in range(p+1):
        up_i[i] = np.sum(w_int * func_u(x_int) * h_i[i,:])
        
    
    #mass matrices
    Mu = np.einsum("ik, jk, k -> ij", h_i, h_j, w_int, optimize="optimal")
    Mphi = np.einsum("ik, jk, k -> ij", e_i, e_j, w_int, optimize="optimal")
    
    #building incidence matrix
    E = np.zeros((p,p))
    np.fill_diagonal(E,-1)
    np.fill_diagonal(E[:,1:], 1)
    extracol = np.zeros((p,1))
    extracol[-1,:] = 1
    E = np.hstack((E,extracol))
    
    
    #building the main matrix equation
    MphiE = -1 * Mphi.dot(E)
    emptysq = np.zeros((p,p))
    emptycol = np.zeros((p,1))
    
    K = np.hstack((np.vstack((Mu,MphiE)), np.vstack((MphiE.T,emptysq))))
    # f_i = f_i.reshape((1,p))
    print(up_i.shape)
    print(emptycol.shape)
    up_i = up_i.reshape((p+1,1))
    F = np.vstack((up_i,emptycol))
    
    #solution
    soln = np.linalg.solve(K,F)
    uo = soln[:p+1].T
    phi = soln[p+1:].T
    
    #reconstruction
    x = np.linspace(-1,1,100)
    h_ix = bf.lagrange_basis(nodes,x)
    uo_h = np.einsum('ij,jk->k', uo, h_ix, optimize='optimal')
    x_for_p = np.linspace(-1,1,np.size(uo_h))
    u_h = uo_h + func(x_for_p)
    e_ix = bf.edge_basis(nodes,x)
    phi_h = np.einsum('ij,jk->k', phi, e_ix, optimize='optimal')
    
    # plt.figure()
    # plt.title('u')
    # plt.plot(x,u_h,label='approx')
    # plt.plot(x,alpha*np.pi*np.cos(alpha*np.pi*x),label='exact')
    # plt.legend()
    
    # plt.figure()
    # plt.title('phi')
    # plt.plot(x,phi_h,label='approx')
    # plt.plot(x,np.sin(alpha*np.pi*x),label='exact')
    # plt.legend()
    # plt.show()
    
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
    ax[0,0].set_title('Implementation B: Polynomial Degree 5')
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel(r'$\phi$')
    ax[0,0].plot(x, func_phi(x), label="Exact")
    ax[0,0].plot(x, phi_h, label="Discretized")
    ax[0,0].grid(True,which="major",color="#999999")
    ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    ax[0,0].minorticks_on()
    ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    fig.savefig("1D_Poisson_NonHomogeneous_deg.png", bbox_inches='tight')                                    ## Insert save destination
    
    ## If you want to see the figure, else disable last two lines.
    fig.tight_layout()
    plt.legend()
    plt.show()
    
    return E, K

poisson_nonhomogen(f,6)
