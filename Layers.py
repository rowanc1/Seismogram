import numpy as np
from Wavelets import WaveletGenerator


def syntheticSeismogram(v, rho, d, wavtyp='RICKER', wavf=[100], usingT=False, maxDepth=500, plotIt=False):
    """

    syntheticSeismogram generates and displays a synthetic seismogram for
    a simple 1-D layered model.

    Inputs:
        v      : velocity of each layer (m/s)
        rho    : density of each layer (kg/m^3)
        d      : depth to the top of each layer (m)
                    The last layer is assumed to be a half-space
        wavtyp : type of Wavelet
                    The wavelet options are:
                        Ricker: takes one frequency
                        Gaussian: still in progress
                        Ormsby: takes 4 frequencies
                        Klauder: takes 2 frequencies
        usingT :


    Lindsey Heagy
    lheagy@eos.ubc.ca
    Created:  November 30, 2013
    Modified: January 16, 2013

    v   = np.array([350, 1000, 2000])  # Velocity of each layer (m/s)
    rho = np.array([1700, 2000, 2500]) # Density of each layer (kg/m^3)
    d   = np.array([0, 100, 200])      # Position of top of each layer (m)
    """

    # Ensure that these are float numpy arrays
    v, rho, d , wavf = np.array(v, dtype=float),   np.array(rho, dtype=float), np.array(d, dtype=float), np.array(wavf,dtype=float)
    usingT           = np.array(usingT, dtype=bool)

    nlayer = len(v) # number of layers

    # Check that the number of layers match
    assert len(rho) == nlayer, 'Number of layer densities must match number of layer velocities'
    assert len(d)   == nlayer, 'Number of layer tops must match the number of layer velocities'

    # compute necessary parameters
    Z   = rho*v                       # acoustic impedance
    R   = np.diff(Z)/(Z[:-1] + Z[1:]) # reflection coefficients
    twttop  = 2*np.diff(d)/v[:-1]     # 2-way travel time within each layer
    twttop  = np.cumsum(twttop)       # 2-way travel time from surface to top of each layer

    # create model logs
    resolution = 400                                                      # How finely we discretize in depth
    dpth       = np.linspace(0,maxDepth,resolution) # create depth vector
    nd         = len(dpth)

    # Initialize logs
    rholog  = np.zeros(nd)  # density
    vlog    = np.zeros(nd)  # velocity
    zlog    = np.zeros(nd)  # acoustic impedance
    rseries = np.zeros(nd)  # reflectivity series
    t       = np.zeros(nd)  # time

    # Loop over layers to put information in logs
    for i in range(nlayer):
        di         = (dpth >= d[i]) # current depth indicies
        rholog[di] = rho[i]         # density
        vlog[di]   = v[i]           # velocity
        zlog[di]   = Z[i]           # acoustic impedance
        if i < nlayer-1:
            di  = np.logical_and(di, dpth < d[i+1])
            ir = np.arange(resolution)[di][-1:][0]
            if usingT:
                if i == 0:
                    rseries[ir] = R[i]
                else:
                    rseries[ir] = R[i]*np.prod(1-R[i-1]**2)
            else:
                rseries[ir] = R[i]
        if i > 0:
            t[di] = 2*(dpth[di] - d[i])/v[i] + twttop[i-1]
        else:
            t[di] = 2*dpth[di]/v[i]


    # make wavelet
    dtwav  = np.abs(np.min(np.diff(t)))/10.0
    twav   = np.arange(-2.0/np.min(wavf), 2.0/np.min(wavf), dtwav)

    # Get source wavelet
    wav = WaveletGenerator(wavtyp,wavf,twav)

    # create synthetic seismogram
    tref  = np.arange(0,np.max(t),dtwav) + np.min(twav)  # time discretization for reflectivity series
    tr    = t[np.abs(rseries) > 0]
    rseriesconv = np.zeros(len(tref))
    for i in range(len(tr)):
        index = np.abs(tref - tr[i]).argmin()
        rseriesconv[index] = R[i]

    seis  = np.convolve(wav,rseriesconv)
    tseis = np.min(twav)+dtwav*np.arange(len(seis))
    index = np.logical_and(tseis >= 0, tseis <= np.max(t))
    tseis = tseis[index]
    seis  = seis[index]


    if plotIt:

        import matplotlib.pyplot as plt
        plt.figure(1)

        # Plot Density
        plt.subplot(151)
        plt.plot(rholog,dpth,linewidth=2)
        plt.title('Density')
        # xlim([min(rholog) max(rholog)] + [-1 1]*0.1*[max(rholog)-min(rholog)])
        # ylim([min(dpth),max(dpth)])
        # set(gca,'Ydir','reverse')
        plt.grid()

        plt.subplot(152)
        plt.plot(vlog,dpth,linewidth=2)
        plt.title('Velocity')
        # xlim([min(vlog) max(vlog)] + [-1 1]*0.1*[max(vlog)-min(vlog)])
        # ylim([min(dpth),max(dpth)])
        # set(gca,'Ydir','reverse')
        plt.grid()

        plt.subplot(153)
        plt.plot(zlog,dpth,linewidth=2)
        plt.title('Acoustic Impedance')
        # xlim([min(zlog) max(zlog)] + [-1 1]*0.1*[max(zlog)-min(zlog)])
        # ylim([min(dpth),max(dpth)])
        # set(gca,'Ydir','reverse')
        plt.grid()

        plt.subplot(154)
        plt.hlines(dpth,np.zeros(nd),rseries,linewidth=2) #,'marker','none'
        plt.title('Reflectivity Series');
        # set(gca,'cameraupvector',[-1, 0, 0]);
        plt.grid()
        # set(gca,'ydir','reverse');

        plt.subplot(155)
        plt.plot(t,dpth,linewidth=2);
        plt.title('Depth-Time');
        # plt.xlim([np.min(t), np.max(t)] + [-1, 1]*0.1*[np.max(t)-np.min(t)]);
        # plt.ylim([np.min(dpth),np.max(dpth)]);
        # set(gca,'Ydir','reverse');
        plt.grid()
        ##
        plt.figure(2)
        # plt.subplot(141)
        # plt.plot(dpth,t,linewidth=2);
        # title('Time-Depth');
        # ylim([min(t), max(t)] + [-1 1]*0.1*[max(t)-min(t)]);
        # xlim([min(dpth),max(dpth)]);
        # set(gca,'Ydir','reverse');
        # plt.grid()

        plt.subplot(132)
        plt.hlines(tref,np.zeros(len(rseriesconv)),rseriesconv,linewidth=2) #,'marker','none'
        plt.title('Reflectivity Series')
        # set(gca,'cameraupvector',[-1, 0, 0])
        plt.grid()

        plt.subplot(131)
        plt.plot(wav,twav,linewidth=2)
        plt.title('Wavelet')
        plt.grid()
        # set(gca,'ydir','reverse')

        plt.subplot(133)
        plt.plot(seis,tseis,linewidth=2)
        plt.grid()
        # set(gca,'ydir','reverse')

        plt.show()


    return dpth, t, seis, tseis


if __name__ == '__main__':

    d      = [0, 50, 100]      # Position of top of each layer (m)
    v      = [350, 1000, 2000]  # Velocity of each layer (m/s)
    rho    = [1700, 2000, 2500] # Density of each layer (kg/m^3)

    syntheticSeismogram(v, rho, d, maxDepth=250, plotIt=True)
