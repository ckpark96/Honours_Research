# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 23:45:10 2020

@author: chang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

x = np.linspace(-1,1.0001,100)
P0 = np.ones((np.size(x)))
P1 = x
P2 = 3/2*x**2 - 1/2
P3 = 5/2*x**3 - 3/2*x
P4 = 35/8*x**4 - 30/8*x**2 + 3/8

# plt.figure
# plt.plot(x, P0, label=r'$P_0$')
# plt.plot(x, P1, label=r'$P_1$')
# plt.plot(x, P2, label=r'$P_2$')
# plt.plot(x, P3, label=r'$P_3$')
# plt.plot(x, P4, label=r'$P_4$')
# plt.legend()


texpsize= [26,28,30]
## Graphing Parameters
SMALL_SIZE  = texpsize[0]
MEDIUM_SIZE = texpsize[1]
BIGGER_SIZE = texpsize[2]

# plt.style.use('grayscale')
plt.rc('font', size=MEDIUM_SIZE, family='serif')    ## controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)                ## fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)                ## fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)               ## fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)               ## legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)             ## fontsize of the figure title
plt.rc('text', usetex=False)
matplotlib.rcParams['lines.linewidth']  = 1.5
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor']   = 'white'
matplotlib.rcParams["legend.fancybox"]  = False

## Graph
fig, ax = plt.subplots(1,1,squeeze=False,figsize=(16,9))
ax[0,0].plot(x, P0, label=r'$P_0$')
ax[0,0].plot(x, P1, label=r'$P_1$')
ax[0,0].plot(x, P2, label=r'$P_2$')
ax[0,0].plot(x, P3, label=r'$P_3$')
ax[0,0].plot(x, P4, label=r'$P_4$')
#ax[0,0].loglog(x, y, marker = "s", color='black', markerfacecolor='none', markeredgewidth=2, markersize=6, label="test")
# ax[0,0].set_ylabel(r"Deflection in $y^{\prime}$ direction $u\,\,[m]$")          ## String is treatable as latex code
# ax[0,0].set_xlabel(r"Position along Aileron $x\,\,[m]$")
#ax[0,0].set_xlim(0,x[-1])
ax[0,0].grid(True,which="major",color="#999999")
ax[0,0].grid(True,which="minor",color="#DDDDDD",ls="--")
ax[0,0].minorticks_on()
ax[0,0].tick_params(which='major', length=10, width=2, direction='inout')
ax[0,0].tick_params(which='minor', length=5, width=2, direction='in')
ax[0,0].legend(loc=0, framealpha=1.0).get_frame().set_edgecolor('k')
fig.savefig("yeet.png", bbox_inches='tight')                                    ## Insert save destination

## If you want to see the figure, else disable last two lines.
fig.tight_layout()


plt.show()