#!/usr/bin/env python
"""
computes full wavefield snapshots
"""

import numpy as np
import matplotlib.pyplot as plt
from yannosclasses import YannosModeBinary, YannosModel

def main():
    fname_model = 'data/PREMQL6'
    fname_bin   = 'data/PREMQL6.bin.mhz25.S'
    model = YannosModel(fname_model)
    modes = YannosModeBinary(fname_bin)

    slat = 30.
    slon = 50.

if __name__ == "__main__":
    main()
