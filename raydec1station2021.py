import numpy as np
from scipy.signal import detrend


def raydec1station(vert=None, north=None, east=None, time=None, fmin=None, fmax=None, fsteps=None, cycles=None, dfpar=None, nwind=None):
    # RAYDEC1STATION(VERT, NORTH, EAST, TIME, FMIN, FMAX, FSTEPS, CYCLES, DFPAR, NWIND)
    #    calculates the ellipticity of Rayleigh waves for the
    #    input data VERT, NORTH, EAST and TIME for a single station
    #    for FSTEPS frequencies (on a logarithmic scale) between
    #    FMIN and FMAX, using CYCLES periods for the stacked signal
    #    and DFPAR as the relative bandwidth for the filtering.
    #    The signal is cut into NWIN different time windows and
    #    RayDec is applied to each of them.

    #    VERT, NORTH, EAST and TIME have to be data matrices
    #    (N x 1 or 1 x N) of equal sizes

    #    suggested values: CYCLES = 10
    #                      DFPAR = 0.1
    #                      NWIND such that the single time windows are about 10 minutes long

    #    Code written by Manuel Hobiger,
    #    Laboratoire de Géophysique Interne et Tectonophysique (LGIT), Grenoble, France, 2008-10
    #    Last change: 18/06/2021

    v1 = vert
    n1 = north
    e1 = east
    t1 = time
#    if v1.shape[1] > v1.shape[0]:
#        v1 = transpose(v1)
#        n1 = transpose(n1)
#        e1 = transpose(e1)
#        t1 = transpose(t1)

    # setting up
    K0 = v1.shape[0]
    K = int(np.floor(K0 / nwind))
    tau = t1[2] - t1[1]
    DTmax = 30
    fnyq = 1 / (2 * tau)
    fstart = max(fmin, 1./DTmax)
    fend = min(fmax, fnyq)
    flist = np.zeros((fsteps, 1))
    constlog = (fend / fstart) ** (1 / (fsteps - 1))
    fl = fstart * constlog ** (np.cumsum(np.ones((fsteps, nwind))) - 1)
    el = np.zeros((fsteps, nwind))
    # loop over the time windows
    for ind1 in np.arange(1, nwind+1).reshape(-1):
        vert = detrend(v1[np.arange((ind1 - 1) * K + 1, ind1 * K+1)])
        north = detrend(n1[np.arange((ind1 - 1) * K + 1, ind1 * K+1)])
        east = detrend(e1[np.arange((ind1 - 1) * K + 1, ind1 * K+1)])
        time = t1[np.arange((ind1 - 1) * K + 1, ind1 * K+1)]
        horizontalamp = np.zeros((fsteps, 1))
        verticalamp = np.zeros((fsteps, 1))
        horizontallist = np.zeros((fsteps, 1))
        verticallist = np.zeros((fsteps, 1))
        Tmax = np.amax(time)
        print(fsteps, np.ceil(Tmax * fend))
        thetas = np.zeros((fsteps, int(np.ceil(Tmax * fend))))
        corr = np.zeros((fsteps, int(np.ceil(Tmax * fend))))
        ampl = np.zeros((fsteps, int(np.ceil(Tmax * fend))))
        dvmax = np.zeros((fsteps, 1))
        for findex in np.arange(1, fsteps+1, 1).reshape(-1):
            f = fl(findex, ind1)
            df = dfpar * f
            fmin = np.amax(fstart, f - df / 2)
            fmax = np.amin(fnyq, f + df / 2)
            flist[findex] = f
            DT = cycles / f
            wl = np.round(DT / tau)
            na, wn = cheb1ord(np.array([fmin + (fmax - fmin) / 10, fmax - (fmax - fmin) / 10]) /
                              fnyq, np.array([fmin - (fmax - fmin) / 10, fmax + (fmax - fmin) / 10]) / fnyq, 1, 5)
            ch1, ch2 = cheby1(na, 0.5, wn)
            taper1 = np.arange(
                0, 1+1 / np.round(time.shape[1-1] / 100), 1 / np.round(time.shape[1-1] / 100))
            taper2 = np.ones((1, time.shape[1-1] - taper1.shape[2-1] * 2))
            taper3 = fliplr(taper1)
            taper = transpose(np.array([taper1, taper2, taper3]))
            # filtering the signals
            norths = filter(ch1, ch2, np.multiply(taper, north))
            easts = filter(ch1, ch2, np.multiply(taper, east))
            verts = filter(ch1, ch2, np.multiply(taper, vert))
            derive = (np.sign(verts(np.arange(2, K+1))) -
                      np.sign(verts(np.arange(1, (K - 1)+1)))) / 2
            vertsum = np.zeros((wl, 1))
            horsum = np.zeros((wl, 1))
            dvindex = 0
            for index in np.arange(np.ceil(1 / (4 * f * tau)) + 1, len(derive) - wl+1, 1).reshape(-1):
                if derive(index) == 1:
                    dvindex = dvindex + 1
                    vsig = verts(np.arange(index, (index + wl - 1)+1))
                    esig = easts(np.arange(index - int(np.floor(1 / (4 * f * tau))),
                                 (index - int(np.floor(1 / (4 * f * tau))) + wl - 1)+1))
                    nsig = norths(np.arange(index - int(np.floor(1 / (4 * f * tau))),
                                  (index - int(np.floor(1 / (4 * f * tau))) + wl - 1)+1))
                    integral1 = sum(np.multiply(vsig, esig))
                    integral2 = sum(np.multiply(vsig, nsig))
                    theta = np.arctan(integral1 / integral2)
                    if integral2 < 0:
                        theta = theta + np.pi
                    theta = np.mod(theta + np.pi, 2 * np.pi)
                    hsig = np.sin(theta) * esig + np.cos(theta) * nsig
                    correlation = sum(np.multiply(
                        vsig, hsig)) / np.sqrt(sum(np.multiply(vsig, vsig)) * sum(np.multiply(hsig, hsig)))
                    if correlation >= - 1:
                        vertsum = vertsum + correlation ** 2 * vsig
                        horsum = horsum + correlation ** 2 * hsig
                    thetas[findex, dvindex] = theta
                    correlationlist[index] = correlation
                    thetalist[index] = theta
                    corr[findex, dvindex] = correlation
                    dvmax[findex] = dvindex
                    ampl[findex, dvindex] = sum(vsig ** 2 + hsig ** 2)
            klimit = np.round(DT / tau)
            verticalamp[findex] = np.sqrt(
                sum(vertsum(np.arange(1, klimit+1)) ** 2))
            horizontalamp[findex] = np.sqrt(
                sum(horsum(np.arange(1, klimit+1)) ** 2))
        ellist = horizontalamp / verticalamp
        fl[:, ind1] = flist
        el[:, ind1] = ellist

    V = fl
    W = el
    return V, W
