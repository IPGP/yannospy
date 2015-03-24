#!/usr/bin/env python
"""
plots classical dispersion diagram
"""
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from yannosclasses import YannosModeBinary

def main():
    modes = YannosModeBinary('data/PREMQL6.bin.mhz25.S')
    freqs = modes.modes['w']
    ls    = modes.modes['l']
    ns    = modes.modes['n']

    lgrid = np.linspace(0.,np.max(ls),2*modes.nmodes)
    ngrid = np.linspace(0.,np.max(ns),2*modes.nmodes)

    fgrid = griddata( (ls,ns),freqs, (lgrid[None,:],ngrid[:,None]), method='cubic' )

    cs = plt.contour( lgrid,ngrid,fgrid,20,colors='k' )
    cs = plt.contourf( lgrid,ngrid,fgrid,20,cmap=plt.cm.jet )
    plt.colorbar(cs)

    plt.scatter(ls,ns,marker='o',c='b',s=5)
    plt.title('mode dispersion diagram (nl-plane)')
    plt.show()

if __name__ == "__main__":
    main()
