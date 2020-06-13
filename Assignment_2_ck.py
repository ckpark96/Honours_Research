import bf_polynomials as bf
import numpy as np
from scipy import integrate

#f = lambda a: np.sin(np.pi*a)

def f(x):
    return np.exp(3*x)*np.sin(np.pi*x)

def numerical_integration_1d(func , quad_degree , quad_type = 'gauss' , a=-1 , b =1):
    """
    This module do numerical integration of 'func ' over domain [a, b] using
    'quad_type ' quadrature at degree 'quad_degree '.
    """
#   s = (quad_degree + 1) / 2
    xi , wi = bf.gauss_quad(quad_degree)
#    print(len(xi))
    intx = 0
    for i in range(quad_degree): #mistake: -1
        intx += wi[i] * func(xi[i])
    # intx = np.sum(wi*func(0.5*(1-xi)*a+0.5*(1+xi)*b)) * (b-a)*0.5
    return intx 



print(numerical_integration_1d(f, 20, a=-1, b=1))

#
#
true = integrate.quad(f, a=-1 , b=1)
#print(true)
