import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import bf_polynomials as bf

def lagrange_2d(func,p):
    x = np.linspace(-1,1,100)
    y = x #square
    xe,ye = np.meshgrid(x,y)
    f_exact = func(xe,ye)
    
    # # plt.figure()
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
    
    # # Graph
    # fig, ax = plt.subplots(1,1,squeeze=False,figsize=(12,9))
    # cp = ax[0,0].contour(xe,ye,f_exact)
    # cp = plt.contourf(xe,ye,f_exact)
    # # ax[0,0].set_ylabel(r"x")          ## String is treatable as latex code
    # # ax[0,0].set_xlabel(r"y")
    # # ax[0,0].grid(True,which="major",color="#999999")
    # # ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    # # ax[0,0].minorticks_on()
    # # ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    # # ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    # # ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    # fig.savefig("lagrange_2D_exact_deg.png", bbox_inches='tight')           
    # plt.colorbar(cp)
    
    
    xi = bf.lobatto_quad(p)[0]
    eta = xi #square
    
    h_i = bf.lagrange_basis(xi,x)
    h_j = bf.lagrange_basis(eta,y)
    
    nodes = np.meshgrid(xi,eta)
    f_ij = func(*nodes)
    
    f_h = np.einsum('ij,im,jn->mn', f_ij,h_i,h_j, optimize='optimal')
    
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
    cpp = ax[0,0].contourf(xe,ye,f_h)
    # ax[0,0].set_ylabel(r"x")          ## String is treatable as latex code
    # ax[0,0].set_xlabel(r"y")
    # ax[0,0].grid(True,which="major",color="#999999")
    # ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
    # ax[0,0].minorticks_on()
    # ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
    # ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
    # ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
    fig.savefig("lagrange_2D_discr_deg.png", bbox_inches='tight')  
    plt.colorbar(cpp)
    
    
    # fig.tight_layout()
    plt.show()
    return

def f2(x,y):
    return 2*np.pi*np.cos(np.pi*x)*np.cos(np.pi*y)

lagrange_2d(f2,6)