#!/usr/bin/env python
"""
computes full wavefield snapshots
"""

from yannosclasses import YannosModeBinary

def main():
    fname_bin   = 'data/PREMQL6.bin.mhz25.S'
    modes = YannosModeBinary(fname_bin)

    moment_tensor = [1.,1.,1.,0.5,-0.5,0.]

    modes.plot_swradiation(40.,moment_tensor,0,100,show=False)
    modes.plot_wavefield(2000.,40.,40.,moment_tensor)

if __name__ == "__main__":
    main()
