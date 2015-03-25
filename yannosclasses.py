#!/usr/bin/env python
"""
This module contains three classes that are associated with the output/input
files of the normal mode code yannos:

    1) YannosModeBinary:

    2) YannosModeAscii: this ascii file contains information about normal mode
    meta information (frequencies, group velocities, ...)

    3) YannosModel:
"""

#==== IMPORTS AND GLOBAL DEFINITIONS ====
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import ipdb

sys.path.append('/home/matthias/projects/gitprojects/SHTOOLS_dev/SHTOOLS')
from pyshtools import MakeGridDH

#==== FIGURE STYLE ====
pt2inch = 1.0/72.27
textwidth_pt = 504.0
columnwidth_pt = 240.0
textwidth_in = textwidth_pt*pt2inch
columnwidth_in = columnwidth_pt*pt2inch
style = {'font.size'       : 7,
         'axes.labelsize'  : 'medium',
         'axes.titlesize'  : 'medium',
         'xtick.labelsize' : 'medium',
         'ytick.labelsize' : 'medium',
         'legend.fontsize' : 'medium',
         'figure.dpi'      : 160,
         'savefig.dpi'     : 160,
         'font.family'     : 'sans',
         'font.serif'      : ['Computer Modern Roman'],
         'figure.figsize'  : (columnwidth_in, columnwidth_in/2)}
mpl.rcParams.update(style)

#===== MAIN FUNCTION (FOR DIRECT EXECUTION) =====
def main():
    fname_model = 'data/PREMQL6'
    fname_bin   = 'data/PREMQL6.bin.mhz25.S'
    model = YannosModel(fname_model)

    modes = YannosModeBinary(fname_bin)
    modes.info()
    #modes.test_normalization(rhos)
    mask = (modes.modes['n']==0) * ((modes.modes['l']==10) + (modes.modes['l']==40))
    modes.plot_modes(mask ,show=False)
    modes.plot_kernels(model,mask, show=True)

#========== CLASS DEFINITIONS ==============
#========== MODE BINARY FILE ===============
class YannosModeBinary(object):

    #==== constructor methods ====
    def __init__(self,fname):
        """
        This class reads yannos binary mode files (ucb format). 
        Works only for spherioidal modes so far
        """
        #--- record 1: meta information ---
        binfile = open(fname,'rb')
        rstart = np.fromfile(binfile,count=1,dtype=np.int32)
        self.nocean,self.nradii = np.fromfile(binfile,count=2,dtype=np.int32)
        self.radii = np.fromfile(binfile,count=222,dtype=np.float32)
        rend = np.fromfile(binfile,count=1,dtype=np.int32)
        assert rstart==rend,'record incomplete'
        assert len(self.radii) == self.nradii, 'incorrect number of layers'
        print('{:d} radial knots ({:d} ocean layers)'.format(self.nradii,self.nocean))
    
        #--- read mode records ---
        ModeRecord = np.dtype([('n' ,np.int,1),
                               ('l' ,np.int,1),
                               ('w' ,np.float,1),
                               ('q' ,np.float,1),
                               ('gv',np.float,1),
                               ('U' ,np.float32,self.nradii),
                               ('dU',np.float32,self.nradii),
                               ('V' ,np.float32,self.nradii),
                               ('dV',np.float32,self.nradii),
                               ('P' ,np.float32,self.nradii),
                               ('dP',np.float32,self.nradii)])

        print('reading mode records...')
        modes = []
        while True:
            rstart = np.fromfile(binfile,count=1,dtype=np.int32)
            if len(rstart)==0:
                break
            n,l    = np.fromfile(binfile,count=2,dtype=np.int32)
            w,q,gv = np.fromfile(binfile,count=3,dtype=np.float64)
            U      = np.fromfile(binfile,count=self.nradii,dtype=np.float64)
            dU     = np.fromfile(binfile,count=self.nradii,dtype=np.float64)
            if l != 0: #spheroidal mode
                V  = np.fromfile(binfile,count=self.nradii,dtype=np.float64)
                dV = np.fromfile(binfile,count=self.nradii,dtype=np.float64)
                P  = np.fromfile(binfile,count=self.nradii,dtype=np.float64)
                dP = np.fromfile(binfile,count=self.nradii,dtype=np.float64)
            else:      #radial mode
                V  = np.zeros_like(U)
                dV = np.zeros_like(U)
                P  = np.zeros_like(U)
                dP = np.zeros_like(U)
            rend = np.fromfile(binfile,count=1,dtype=np.int32)
            assert rstart==rend,'mode record incomplete, file corrupt?'
            modes.append( (n,l,w,q,gv,U,dU,V,dV,P,dP) )

        self.modes  = np.array(modes,dtype=ModeRecord)
        self.nmodes = len(self.modes)
        self.lmax   = np.max(self.modes['l'])
        self.nmax   = np.max(self.modes['n'])
        print('found {:d} modes'.format(self.nmodes))

    #==== CATALOGUE INFO METHODS ====
    def info(self):
        """prints some informations about the mode catalogue"""
        print('contains {:d} modes'.format(self.nmodes))
        minmode = self.modes[np.argmin(self.modes['w'])]
        print('lowest frequency mode:')
        info = minmode['n'],minmode['l'],1e3*minmode['w']/2./np.pi
        print('{:d} S {:d} f={:2.2f}mHz'.format(*info))
        maxmode = self.modes[np.argmax(self.modes['w'])]
        print('lowest frequency mode:')
        info = maxmode['n'],maxmode['l'],1e3*maxmode['w']/2./np.pi
        print('{:d} S {:d} f={:2.2f}mHz'.format(*info))

    #==== EIGENFUNCTIONS ====
    def plot_modes(self,mask,show=True):
        """
        plots normal mode eigenfunctions.
        mask: e.g. self.modes['n'] == 0
        """
        fig,ax = plt.subplots(1,1)
        radii = np.linspace(0.,6371.,self.nradii)
        for mode in self.modes[mask]:
            ax.plot(radii,mode['U'],label=r'$U_{{ {:d},{:d} }}$'.format(mode['n'],mode['l']))
            ax.plot(radii,mode['V'],label=r'$V_{{ {:d},{:d} }}$'.format(mode['n'],mode['l']))
        fig.tight_layout(pad=0.1)
        fig.subplots_adjust(right=0.8)
        ax.legend(bbox_to_anchor=(1.25,1.))
        if show: plt.show()

    def test_normalization(self,rhos):
        """test the normal mode normalization using a simple trapezoidal integration"""
        #integrate over radius
        rhon  = 5515.0
        wn    = 1.07519064529946291e-003
        norms  = []
        for imode,mode in enumerate(self.modes):
            integrand = rhos/rhon*self.radii**2*(mode['U']**2 + mode['V']**2)*(mode['w']/wn)**2
            norm = np.trapz(integrand,x=self.radii)
            norms.append( (imode,norm) )
        norms = np.array(norms)
        bins = np.logspace(-4,0,10)
        errors = np.abs(norms[:,1]-1.0)
        hist,bins = np.histogram(errors,bins=bins)
        print('distribution of norms (due to undersampling errors?):')
        for ibin,nmodes in enumerate(hist):
            mask   = np.logical_and(errors>bins[ibin],errors<bins[ibin+1])
            avfreq = np.mean(self.modes['w'][mask])*1e3/2./np.pi
            avl    = int(np.mean(self.modes['l'][mask]))
            avn    = int(np.mean(self.modes['n'][mask]))
            print('error {:2.5f}-{:2.5f} : {:d} modes,  avl: {:d}, avn: {:d}, avfreq: {:2.2f}mHz'.format(bins[ibin],bins[ibin+1],nmodes,avl,avn,avfreq))
        print('modes in last bin:')
        for imode in norms[errors>bins[-2],0]:
            mode = self.modes[imode]
            info = mode['n'],mode['l'],1e3*mode['w']/2./np.pi
            print('{:d} S {:d} f={:2.2f}mHz'.format(*info))

    #==== 1D SENSITIVITY KERNEL ROUTINES ====
    def get_kernels(self,model,mask):
        """computes vp/vs kernels according to Dahlen & Tromp Eq: 9.13 ff."""
        print('NOT YET PROPERLY BENCHMARKED!')

        #---- compute anisotropic love parameters and isotropic velocity ---
        rho  = model.data['rho']
        vpv  = model.data['vpv']
        vsv  = model.data['vsv']
        vph  = model.data['vph']
        vsh  = model.data['vsh']
        eta  = model.data['eta']

        A = vph**2*rho
        C = vpv**2*rho
        N = vsh**2*rho
        L = vsv**2*rho
        F = eta*(A-2*L)
        Kappa_iso = 1./9.*(4.*A + C + 4.*F - 4.*N)
        Mu_iso    = 1./15.*(A+C-2.*F+5.*N + 6.*L)

        vs_iso = np.sqrt(Mu_iso/rho)
        vp_iso = np.sqrt((Kappa_iso+4./3.*Mu_iso)/rho)
 
        #---- loop through all selected modes and compute and store vp/vs kernels ----
        nselect  = np.count_nonzero(mask)
        nkernels = 2 #isotropic vp and vs kernels
        kernels  = np.zeros( (nselect,nkernels,self.nradii) )

        r  = self.radii
        for imode,mode in enumerate(self.modes[mask]):
            l  = mode['l']
            k  = np.sqrt(l*(l+1))
            w  = mode['w']
            U  = mode['U']
            dU = mode['dU']
            V  = mode['V']
            dV = mode['dV']

            KKappa = (r*dU+2*U-k*V)**2
            KMu    = 1/3*(2*r*dU-2*U+k*V)**2\
                    +(r*dV-V+k*U)**2+(k**2-2)*V**2

            #careful there is an extra factor vp_iso and vs_iso because we show relative kernels!
            kernels[imode,0] = 2*rho*vp_iso**2/(2*w)*KKappa 
            kernels[imode,1] = 2*rho*vs_iso**2/(2*w)*(KMu-4/3*KKappa)

        return kernels

    def plot_kernels(self,model,mask,show=False):
        """plots isotropic vs and vp kernels"""
        kernels = self.get_kernels(model,mask)
        fig,axes = plt.subplots(1,2, figsize=(textwidth_in,textwidth_in/2.))
        axes[0].set_title('isotropic vp kernels')
        axes[1].set_title('isotropic vs kernels')
        ylim = np.max(np.abs(kernels))

        for kernel,mode in zip(kernels,self.modes[mask]):
            disc = np.array([670.,2880.])
            axes[0].plot(self.radii*6371.,kernel[0])
            axes[0].vlines(6371.-disc,0.,kernel[0].max())

            axes[1].plot(self.radii*6371.,kernel[1],label='{:d}S{:d}'.format(mode['n'],mode['l']))
            axes[1].vlines(6371.-disc,0.,kernel[1].max())

        fig.tight_layout(pad=0.2)
        fig.subplots_adjust(bottom=0.3)
        plt.setp(axes,ylim=(-ylim,ylim),xlim=(0,6371))
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5,0.25),bbox_transform=plt.gcf().transFigure,ncol=4)
        if show: plt.show()

    #==== SOURCE EXCITATION ROUTINES ====
    def get_coefficients(self, depth, moment_tensor, mask=None):
        """computes moment tensor excitation coefficients according to Dahlen & Tromp 10.35 ff"""
        #compute excitation wavefield
        Mrr = moment_tensor[0]
        Mtt = moment_tensor[1]
        Mpp = moment_tensor[2]
        Mrt = moment_tensor[3]
        Mrp = moment_tensor[4]
        Mtp = moment_tensor[5]

        r = 1.-depth/6371.
        lmax = self.lmax
        nmax = self.nmax
        coeffs = np.zeros( (nmax+1,lmax+1,2,3) )

        for mode in self.modes:
            l = mode['l']
            k = np.sqrt(l*(l+1))
            n = mode['n']

            #get eigenfunctions at source depth 
            #'map_coordinates' function should be faster!
            U  = np.interp(r,self.radii,mode['U'])
            dU = np.interp(r,self.radii,mode['dU'])
            V  = np.interp(r,self.radii,mode['V'])
            dV = np.interp(r,self.radii,mode['dV'])

            #compute A coefficients
            if k==0:
                coeffs[n,l,0,0] = Mrr*dU+(Mtt+Mpp)*(U-0.5*k*V)/r
            else:
                coeffs[n,l,0,0] = Mrr*dU+(Mtt+Mpp)*(U-0.5*k*V)/r
                coeffs[n,l,0,1] = Mrt*(dV-V/r+k*U/r)/k
                coeffs[n,l,0,2] = 0.5*(Mtt-Mpp)*V/(k*r)

                #compute B coefficients
                #coeffs[n,l,1,0] = 0.
                coeffs[n,l,1,1] = Mrp*(dV-V/r+k*U/r)/k
                coeffs[n,l,1,2] = Mtp*V/k

        return coeffs

    def get_layer(self, time, depth, Sdepth, moment_tensor, mask=None):
        """gets excitation coefficients and computes wavefield"""
        print('CAREFUL, BUGGY!!')
        excoeffs = self.get_coefficients(depth, moment_tensor, mask)
        coeffs = np.zeros( (2,self.lmax+1,self.lmax+1) )
        r = 1.-depth/6371.
        for mode in self.modes:
            U  = np.interp(r,self.radii,mode['U'])
            n = mode['n']
            l = mode['l']
            omega = mode['w']
            coeffs[:,l,:3] += U*excoeffs[n,l]*(1.-np.cos(omega*time))
        grid = MakeGridDH(coeffs)
        return grid

    def plot_wavefield(self,depth,Sdepth,moment_tensor,mask=None):
        """plots wavefield to screen"""
        grid = self.get_layer(depth,Sdepth,moment_tensor,mask)
        plt.figure()
        plt.imshow(grid)
        plt.show()

    def get_swradiation(self,Sdepth,moment_tensor,n,l):
        """computes surface wave radiation pattern according to Dahlen & Tromp 1998, eq. 11.34"""
        #select mode
        mask  = np.logical_and(self.modes['n'] == n, self.modes['l']==l)
        mode = self.modes[mask][0]
        r   = 1.-Sdepth/6371.

        #get eigenfunctions at source depth
        k = np.sqrt(l*(l+1))
        U  = np.interp(r,self.radii,mode['U'])
        dU = np.interp(r,self.radii,mode['dU'])
        V  = np.interp(r,self.radii,mode['V'])
        dV = np.interp(r,self.radii,mode['dV'])

        #rename moment tensor values:
        Mrr = moment_tensor[0]
        Mtt = moment_tensor[1]
        Mpp = moment_tensor[2]
        Mrt = moment_tensor[3]
        Mrp = moment_tensor[4]
        Mtp = moment_tensor[5]

        #make azimuth array:
        az = np.linspace(0.,2.*np.pi,1024)

        #compute radiation pattern:
        R = Mrr*dU + (Mpp+Mtt)/r*(U-0.5*k*V) * np.exp(1j*np.pi/4) \
          + (-1)**0 * (dV-V/r +k/r*U)*(Mrp* np.sin(az) +Mrt*np.cos(az))*np.exp(-1j*np.pi/4) \
          - k/r*V * (Mtp*np.sin(2.*az) + 0.5*(Mtt-Mpp)*np.cos(2.*az))*np.exp(1j*np.pi/4) 

        R  = np.abs(R)
        R /= np.max(R)

        return az,R

    def plot_swradiation(self,Sdepth,moment_tensor,n,l,show=True):
        """plots the surface wave radiation pattern for a given moment tensor according to Dahlen & Tromp, eq.11.34"""
        az,R = self.get_swradiation(Sdepth,moment_tensor,n,l)
        fig,ax = plt.subplots(1,1)
        ax.plot(R*np.sin(-az),R*np.cos(-az))
        ax.set_aspect("equal")
        plt.setp(ax,xlim=(-1.2,1.2),ylim=(-1.2,1.2))
        if show: plt.show()

#============ MODE ASCII CLASS ===============
class YannosModeAscii:
    """
    this class reads a 1D model ascii mode file from yannos and uses it to
    extract the Q, as well as the group velocity value of fundamental mode
    surface waves (possibly overtones).
    """
    def __init__(self, fn_yannos=None, fn_model=None):
        if fn_yannos is None:
            fn_yannos = '/home/matthias/projects/python/modules/yannosclasses/PREMQL6.asc.mhz25.S'
        self.fname = fn_yannos

        modefile = [line.strip().rsplit(None,7) for line in open(fn_yannos).readlines()[17:]]
        modefile = sorted(modefile,key=lambda line: float(line[2]))

        Ls = []
        Fs = []
        Qs = []
        Vs = []

        for mode in modefile:
            n = int(mode[0].split()[0])
            l = int(mode[0].split()[2])
            w = float(mode[2])*1e-3
            v = float(mode[4])
            Q = float(mode[5])
            if n == 0 and l>5:
                Ls.append(l)
                Fs.append(w)
                Qs.append(Q)
                Vs.append(v)

        self.Ls = np.array(Ls)
        self.Fs = np.array(Fs)
        self.Qs = np.array(Qs)
        self.Vs = np.array(Vs)

    def getQ(self,fmeas):
        return np.interp(fmeas,self.Fs,self.Qs)

    def getV(self,fmeas):
        return np.interp(fmeas,self.Fs,self.Vs)

    def plotQ(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('1D model Attenuation from file: %s'%self.fname.split('/')[-1])
        ax.set_xlabel('frequency in [Hz]')
        ax.set_ylabel('Attenuation factor Q [unitless]')
        ax.plot(self.Fs,self.Qs)

    def plotV(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('1D model group velocity curve from file: %s'%self.fname.split('/')[-1])
        ax.set_xlabel('frequency in [Hz]')
        ax.set_ylabel('group velocity in [km/s]')
        ax.plot(self.Fs,self.Vs)
      
    def arrival_times(self, statdist, npfreqs, narr = 10):
        """
        this function computes the surface wave arrival times for a given 1dmodel
        :param statdist: station distance in km
        :param npfreqs:  numpy array with frequencies (linear interpolation between modes)
        :param narr: number of arrivals
        :param fmin: minimum frequency calculated
        :param fmax: maximum frequency calculated

        """
        npvs = self.getV(npfreqs)
        nparrivals = np.zeros( (len(npfreqs),narr) )

        dt_arr = 2. * np.pi * 6371./npvs
        dt_sta = statdist/npvs
    
        for i in range(0,narr/2,1):
            minor =  dt_sta + dt_arr * float(i)
            major = -dt_sta + dt_arr * float(i+1)
            nparrivals[:,i*2] = minor
            nparrivals[:,i*2+1] = major
        
        return nparrivals

    def freqsToDegrees(self, freqs, unit='mhz'):
        if unit == 'mhz':
            freqs_Hz = freqs*1e-3
        else:
            freqs_Hz = freqs
        Ls = np.interp(freqs_Hz,self.Fs,self.Ls)
        return Ls

#============ MODEL CLASS ===============
class YannosModel(object):
    def __init__(self, fname):
        """
        initializes the class from a 1d model file
        :param fname: file name of the model file
        """
        file_object = open(fname,'r')
        self.name    = file_object.readline()
        self.flags   = file_object.readline().split()
        self.nlayers = file_object.readline().split()
        DataLayout = np.dtype([('rad',np.float32,1),
                               ('rho',np.float32,1),
                               ('vpv',np.float32,1),
                               ('vsv',np.float32,1),
                               ('Qk',np.float32,1),
                               ('Qmu',np.float32,1),
                               ('vph',np.float32,1),
                               ('vsh',np.float32,1),
                               ('eta',np.float32,1)])
        self.data   = np.loadtxt(file_object,dtype=DataLayout)
        self.units  = np.array(['m','kg/m^3','m/s','m/s','m/s','-','-','m/s','m/s','-'])

    #-------------------------------
    def get(self, variable):
        """
        :param variable: available quantities are 'rad','rho','vpv','vsv','Qk','Qmu','vph','vsh','eta'
        :returns: numpy array of the desired variable
        """
        return self.data[variable]

    #-------------------
    def set(self, variable, array):
        """
        :param variable: available quantities are 'density','vpv','vsv','Qk','Qmu','vph','vsh','eta'
        :param array:    the new array (should have correct dimensions)
        """
        self.data[variable] = array

    #--------------------
    def info(self):
        """prints model info"""
        print('model name: ',self.name)
        print('total number of layers: ',self.nlayers[0])

    #--------------------
    def plot(self, variable = 'vpv'):
        """
        plots one parameter
        :param variable: available quantities are 'density','vpv','vsv','Qk','Qmu','vph','vsh','eta'
        """
        plt.figure()
        plt.plot(self.data[:,0],self.data[variable],label=variable)
        plt.xlabel(self.labels[0] + ' in: ' + self.units[0])
        plt.ylabel(variable)
        plt.show()

    #------------------
    def plot_all(self):
        """
        plots all model data in one figure
        """
        fig = plt.figure()           
        fig.suptitle(self.name)
        ax1 = fig.add_subplot(211)   
        ax1twin = ax1.twinx()       
        ax2 = fig.add_subplot(212) 
        ax2twin = ax2.twinx()     
    
        radii = self.data['rad']  
    
        ax1.plot(radii,self.data['rho'],'--',label='rho')
        ax1.legend(loc=2)
        ax1.set_xlabel(r'radius in m')
        ax1.set_ylabel(r'density in $kg/m^3$')
    
        ax1twin.plot(radii,self.data['vpv'],label='vpv')
        ax1twin.plot(radii,self.data['vsv'],label='vsv')
        ax1twin.plot(radii,self.data['vph'],label='vph')
        ax1twin.plot(radii,self.data['vsh'],label='vsh')
        ax1twin.legend()
        ax1twin.set_ylabel(r'velocity in $m/s$')
    
        ax2.plot(radii,self.data['Qmu'],'--')
        ax2.legend(loc=2)
        ax2.set_xlabel(r'radius in m')
        ax2.set_ylabel(r'Attenuation factor Q')
    
        ax2twin.plot(radii,self.data['eta'])
        ax2twin.legend()
        ax2twin.set_ylabel(r'anisotropy $\eta$')
        plt.show()

#==== EXECUTE SCRIPT ====
if __name__ == "__main__":
    main()
