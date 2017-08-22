import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import scipy
import scipy.signal as sig
import os
import pickle as pkl
from scipy import io
from scipy import signal
from PyQt5 import QtGui, QtWidgets
from numpy import linalg as lin
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import h5py
from timeit import default_timer as timer
import platform
from scipy.optimize import curve_fit

def RecursiveLowPassFast(signal, coeff, samplerate):
    ml = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], signal)
    vl = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], np.square(signal - ml))
    sl = ml - coeff['S'] * np.sqrt(vl)
    Ni = len(signal)
    points = np.array(np.where(signal<=sl)[0])
    to_pop=np.array([])
    for i in range(1,len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop=np.append(to_pop, i)
    points = np.delete(points, to_pop)
    RoughEventLocations = []
    NumberOfEvents=0

    for i in points:
        if NumberOfEvents is not 0:
            if i >= RoughEventLocations[NumberOfEvents-1][0] and i <= RoughEventLocations[NumberOfEvents-1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] - coeff['E'] * np.sqrt(vl[i])
        Mm = ml[i]
        Vv = vl[i]
        duration = 0
        while signal[i + 1] < El and i < (Ni - 2) and duration < coeff['eventlengthLimit']*samplerate:
            duration += 1
            i += 1
        if duration >= coeff['eventlengthLimit']*samplerate or i > (Ni - 10):
            NumberOfEvents -= 1
        else:
            k = start
            while signal[k] < Mm and k > 1:
                k -= 1
            start = k - 1
            k2 = i + 1
            while signal[k2] > Mm:
                k2 -= 1
            endp = k2
            if start<0:
                start=0
            RoughEventLocations.append((start, endp, ml[start], vl[start]))

    return np.array(RoughEventLocations)

def RecursiveLowPassFastUp(signal, coeff, samplerate):
    ml = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], signal)
    vl = scipy.signal.lfilter([1 - coeff['a'], 0], [1, -coeff['a']], np.square(signal - ml))
    sl = ml + coeff['S'] * np.sqrt(vl)
    Ni = len(signal)
    points = np.array(np.where(signal>=sl)[0])
    to_pop=np.array([])
    for i in range(1,len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop=np.append(to_pop, i)
    points = np.delete(points, to_pop)

    points =np.delete(points, np.array(np.where(points == 0)[0]))

    RoughEventLocations = []
    NumberOfEvents=0
    for i in points:
        if NumberOfEvents is not 0:
            if i >= RoughEventLocations[NumberOfEvents-1][0] and i <= RoughEventLocations[NumberOfEvents-1][1]:
                continue
        NumberOfEvents += 1
        start = i
        El = ml[i] + coeff['E'] * np.sqrt(vl[i])
        Mm = ml[i]
        duration = 0
        while signal[i + 1] > El and i < (Ni - 2) and duration < coeff['eventlengthLimit']*samplerate:
            duration += 1
            i += 1
        if duration >= coeff['eventlengthLimit']*samplerate or i > (Ni - 10):
            NumberOfEvents -= 1
        else:
            k = start
            while signal[k] > Mm and k > 2:
                k -= 1
            start = k - 1
            k2 = i + 1
            while signal[k2] > Mm:
                k2 -= 1
            endp = k2
            RoughEventLocations.append((start, endp, ml[start], vl[start]))

    return np.array(RoughEventLocations)

def ImportAxopatchData(datafilename):
    x=np.fromfile(datafilename, np.dtype('>f4'))
    f=open(datafilename, 'rb')
    graphene=0
    for i in range(0, 10):
        a=str(f.readline())
        #print(a)
        if 'Acquisition' in a or 'Sample Rate' in a:
            samplerate=int(''.join(i for i in a if i.isdigit()))/1000
        if 'FEMTO preamp Bandwidth' in a:
            femtoLP=int(''.join(i for i in a if i.isdigit()))
        if 'I_Graphene' in a:
            graphene=1
            print('This File Has a Graphene Channel!')
    end = len(x)
    if graphene:
        #pore current
        i1 = x[250:end-3:4]
        #graphene current
        i2 = x[251:end-2:4]
        #pore voltage
        v1 = x[252:end-1:4]
        #graphene voltage
        v2 = x[253:end:4]
        print('The femto was set to : {} Hz, if this value was correctly entered in the LabView!'.format(str(femtoLP)))
        output={'FemtoLowPass': femtoLP, 'type': 'Axopatch', 'graphene': 1, 'samplerate': samplerate, 'i1': i1, 'v1': v1, 'i2': i2, 'v2': v2, 'filename': datafilename}
    else:
        i1 = np.array(x[250:end-1:2])
        v1 = np.array(x[251:end:2])
        output={'type': 'Axopatch', 'graphene': 0, 'samplerate': samplerate, 'i1': i1, 'v1': v1, 'filename': datafilename}
    return output

def ImportChimeraRaw(datafilename):
    matfile=io.loadmat(str(os.path.splitext(datafilename)[0]))
    #buffersize=matfile['DisplayBuffer']
    data = np.fromfile(datafilename, np.dtype('<u2'))
    samplerate = np.float64(matfile['ADCSAMPLERATE'])
    TIAgain = np.int32(matfile['SETUP_TIAgain'])
    preADCgain = np.float64(matfile['SETUP_preADCgain'])
    currentoffset = np.float64(matfile['SETUP_pAoffset'])
    ADCvref = np.float64(matfile['SETUP_ADCVREF'])
    ADCbits = np.int32(matfile['SETUP_ADCBITS'])

    closedloop_gain = TIAgain * preADCgain
    bitmask = (2 ** 16 - 1) - (2 ** (16 - ADCbits) - 1)
    data = -ADCvref + (2 * ADCvref) * (data & bitmask) / 2 ** 16
    data = (data / closedloop_gain + currentoffset)
    data.shape = [data.shape[1], ]
    output = {'matfilename': str(os.path.splitext(datafilename)[0]),'i1raw': data, 'v1': np.float64(matfile['SETUP_mVoffset']), 'samplerate': np.int64(samplerate), 'type': 'ChimeraRaw', 'filename': datafilename}
    return output

def ImportChimeraData(datafilename):
    matfile = io.loadmat(str(os.path.splitext(datafilename)[0]))
    samplerate = matfile['ADCSAMPLERATE']
    if samplerate<4e6:
        data = np.fromfile(datafilename, np.dtype('float64'))
        buffersize = matfile['DisplayBuffer']
        out = Reshape1DTo2D(data, buffersize)
        output = {'i1': out['i1'], 'v1': out['v1'], 'samplerate':float(samplerate), 'type': 'ChimeraNotRaw', 'filename': datafilename}
    else:
        output = ImportChimeraRaw(datafilename)
    return output

def OpenFile(filename = ''):
    if filename == '':
        datafilename = QtGui.QFileDialog.getOpenFileName()
        datafilename=datafilename[0]
        print(datafilename)
    else:
        datafilename=filename
    if datafilename[-3::] == 'dat':
        isdat = 1
        output = ImportAxopatchData(datafilename)
    else:
        isdat = 0
        output = ImportChimeraData(datafilename)
    return output

def RefinedEventDetection(out, AnalysisResults, signals, limit):
    for sig in signals:
        if sig is 'i1_Up':
            sig1 = 'i1'
        elif sig is 'i2_Up':
            sig1 = 'i2'
        else:
            sig1 = sig
        if len(AnalysisResults[sig]['RoughEventLocations']) is not 0:
            startpoints = np.uint64(AnalysisResults[sig]['RoughEventLocations'][:, 0])
            endpoints = np.uint64(AnalysisResults[sig]['RoughEventLocations'][:, 1])
            localBaseline = AnalysisResults[sig]['RoughEventLocations'][:, 2]
            localVariance = AnalysisResults[sig]['RoughEventLocations'][:, 3]

            CusumBaseline=500
            numberofevents = len(startpoints)
            AnalysisResults[sig]['StartPoints'] = startpoints
            AnalysisResults[sig]['EndPoints'] = endpoints
            AnalysisResults[sig]['LocalBaseline'] = localBaseline
            AnalysisResults[sig]['LocalVariance'] = localVariance
            AnalysisResults[sig]['NumberOfEvents'] = len(startpoints)

            #### Now we want to move the endpoints to be the last minimum for each ####
            #### event so we find all minimas for each event, and set endpoint to last ####

            deli = np.zeros(numberofevents)
            dwell = np.zeros(numberofevents)
            AllFits={}

            for i in range(numberofevents):
                length = endpoints[i] - startpoints[i]
                if length <= limit and length>3:
                    # Impulsion Fit to minimal value
                    deli[i] = localBaseline[i] - np.min(out[sig1][startpoints[i]+np.uint(1):endpoints[i]-np.uint(1)])
                    dwell[i] = (endpoints[i] - startpoints[i]) / out['samplerate']
                elif length > limit:
                    deli[i] = localBaseline[i] - np.mean(out[sig1][startpoints[i]+np.uint(5):endpoints[i]-np.uint(5)])
                    dwell[i] = (endpoints[i] - startpoints[i]) / out['samplerate']
                    # # Cusum Fit
                    # sigma = np.sqrt(localVariance[i])
                    # delta = 2e-9
                    # h = 1 * delta / sigma
                    # (mc, kd, krmv) = CUSUM(out[sig][startpoints[i]-CusumBaseline:endpoints[i]+CusumBaseline], delta, h)
                    # zeroPoint = startpoints[i]-CusumBaseline
                    # krmv = krmv+zeroPoint+1
                    # AllFits['Event' + str(i)] = {}
                    # AllFits['Event' + str(i)]['mc'] = mc
                    # AllFits['Event' + str(i)]['krmv'] = krmv
                else:
                    deli[i] = localBaseline[i] - np.min(out[sig1][startpoints[i]:endpoints[i]])
                    dwell[i] = (endpoints[i] - startpoints[i]) / out['samplerate']

            frac = deli / localBaseline
            dt = np.array(0)
            dt = np.append(dt, np.diff(startpoints) / out['samplerate'])
            numberofevents = len(dt)

            #AnalysisResults[sig]['CusumFits'] = AllFits
            AnalysisResults[sig]['FractionalCurrentDrop'] = frac
            AnalysisResults[sig]['DeltaI'] = deli
            AnalysisResults[sig]['DwellTime'] = dwell
            AnalysisResults[sig]['Frequency'] = dt
    return AnalysisResults

def CorrelateTheTwoChannels(AnalysisResults, DelayLimit):
    if len(AnalysisResults['i1']['RoughEventLocations']) is not 0:
        i1StartP = AnalysisResults['i1']['StartPoints'][:]
    else:
        i1StartP =[]
    if len(AnalysisResults['i2']['RoughEventLocations']) is not 0:
        i2StartP = AnalysisResults['i2']['StartPoints'][:]
    else:
        i2StartP = []

    # Common Events, # Take Longer
    CommonEventsi1Index = np.array([], dtype=np.uint64)
    CommonEventsi2Index = np.array([], dtype=np.uint64)

    for k in i1StartP:
        val = i2StartP[(i2StartP > k - DelayLimit) & (i2StartP < k + DelayLimit)]
        if len(val)==1:
            CommonEventsi2Index = np.append(CommonEventsi2Index, np.where(i2StartP == val)[0])
            CommonEventsi1Index = np.append(CommonEventsi1Index, np.where(i1StartP == k)[0])
        if len(val) > 1:
            diff=np.absolute(val-k)
            minIndex=np.where(diff == np.min(diff))
            CommonEventsi2Index = np.append(CommonEventsi2Index, np.where(i2StartP == val[minIndex])[0])
            CommonEventsi1Index = np.append(CommonEventsi1Index, np.where(i1StartP == k)[0])

    # Only i1
    Onlyi1Indexes = np.delete(range(len(i1StartP)), CommonEventsi1Index)
    # Only i2
    Onlyi2Indexes = np.delete(range(len(i2StartP)), CommonEventsi2Index)

    CommonIndexes={}
    CommonIndexes['i1']=CommonEventsi1Index
    CommonIndexes['i2']=CommonEventsi2Index
    OnlyIndexes={}
    OnlyIndexes['i1'] = Onlyi1Indexes
    OnlyIndexes['i2'] = Onlyi2Indexes
    return (CommonIndexes, OnlyIndexes)

def PlotEvent(t1, i1, t2 = [], i2 = [], fit1 = np.array([]), fit2 = np.array([]), channel = 'i1'):
    if len(t2)==0:
        fig1 = plt.figure(1, figsize=(20, 7))
        ax1 = fig1.add_subplot(111)
        ax1.plot(t1, i1*1e9, 'b')
        if len(fit1) is not 0:
            ax1.plot(t1, fit1*1e9, 'y')
        ax1.set_ylabel([channel, ' Current [nA]'])
        ax1.set_ylabel([channel, ' Time [ms]'])
        ax1.ticklabel_format(useOffset=False)
        ax1.ticklabel_format(useOffset=False)
        return fig1
    else:
        fig1 = plt.figure(1, figsize=(20, 7))
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212, sharex=ax1)
        ax1.plot(t1, i1*1e9, 'b')
        if len(fit1) is not 0:
            ax1.plot(t1, fit1*1e9, 'y')
        ax2.plot(t2, i2*1e9, 'r')
        if len(fit2) is not 0:
            ax2.plot(t2, fit2*1e9, 'y')
        ax1.set_ylabel('Ionic Current [nA]')
        #ax1.set_xticklabels([])
        ax2.set_ylabel('Transverse Current [nA]')
        ax2.set_xlabel('Time [ms]')
        ax2.ticklabel_format(useOffset=False)
        ax2.ticklabel_format(useOffset=False)
        ax1.ticklabel_format(useOffset=False)
        ax1.ticklabel_format(useOffset=False)
        return fig1

def SaveAllPlots(CommonIndexes, OnlyIndexes, AnalysisResults, directory, out, buffer, withFit = 1):
    if len(CommonIndexes['i1']) is not 0:
        # Plot All Common Events
        pp = PdfPages(directory + '_SavedEventsCommon.pdf')
        ind1 = np.uint64(CommonIndexes['i1'])
        ind2 = np.uint64(CommonIndexes['i2'])

        t = np.arange(0, len(out['i1']))
        t = t / out['samplerate'] * 1e3
        count=1
        for eventnumber in range(len(ind1)):
            parttoplot = np.arange(AnalysisResults['i1']['StartPoints'][ind1[eventnumber]] - buffer,
                                   AnalysisResults['i1']['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)
            parttoplot2 = np.arange(AnalysisResults['i2']['StartPoints'][ind2[eventnumber]] - buffer,
                                    AnalysisResults['i2']['EndPoints'][ind2[eventnumber]] + buffer, 1, dtype=np.uint64)

            fit1 = np.concatenate([np.ones(buffer) * AnalysisResults['i1']['LocalBaseline'][ind1[eventnumber]],
                                   np.ones(AnalysisResults['i1']['EndPoints'][ind1[eventnumber]] - AnalysisResults['i1']['StartPoints'][
                                       ind1[eventnumber]]) * (
                                       AnalysisResults['i1']['LocalBaseline'][ind1[eventnumber]] - AnalysisResults['i1']['DeltaI'][ind1[eventnumber]]),
                                   np.ones(buffer) * AnalysisResults['i1']['LocalBaseline'][ind1[eventnumber]]])

            fit2 = np.concatenate([np.ones(buffer) * AnalysisResults['i2']['LocalBaseline'][ind2[eventnumber]],
                                   np.ones(AnalysisResults['i2']['EndPoints'][ind2[eventnumber]] - AnalysisResults['i2']['StartPoints'][
                                       ind2[eventnumber]]) * (
                                       AnalysisResults['i2']['LocalBaseline'][ind2[eventnumber]] - AnalysisResults['i2']['DeltaI'][ind2[eventnumber]]),
                                   np.ones(buffer) * AnalysisResults['i2']['LocalBaseline'][ind2[eventnumber]]])
            if withFit:
                fig = PlotEvent(t[parttoplot], out['i1'][parttoplot], t[parttoplot2], out['i2'][parttoplot2],
                                   fit1=fit1, fit2=fit2)
            else:
                fig = PlotEvent(t[parttoplot], out['i1'][parttoplot], t[parttoplot2], out['i2'][parttoplot2])

            if not divmod(eventnumber+1,200):
                pp.close(fig)
                pp = PdfPages(directory + '_SavedEventsCommon_' + str(count) + '.pdf')
                count+=1
            pp.savefig(fig)
            print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
            print('Length i1: {}, Fit i1: {}'.format(len(out['i1'][parttoplot]), len(fit1)))
            print('Length i2: {}, Fit i2: {}'.format(len(out['i2'][parttoplot2]), len(fit2)))
            fig.clear()
            plt.close(fig)
        pp.close()

    if len(OnlyIndexes['i1']) is not 0:
        # Plot All i1
        pp = PdfPages(directory + '_SavedEventsOnlyi1.pdf')
        ind1 = np.uint64(OnlyIndexes['i1'])

        t = np.arange(0, len(out['i1']))
        t = t / out['samplerate'] * 1e3
        count=1
        for eventnumber in range(len(ind1)):
            parttoplot = np.arange(AnalysisResults['i1']['StartPoints'][ind1[eventnumber]] - buffer,
                                   AnalysisResults['i1']['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)

            fit1 = np.concatenate([np.ones(buffer) * AnalysisResults['i1']['LocalBaseline'][ind1[eventnumber]],
                                   np.ones(AnalysisResults['i1']['EndPoints'][ind1[eventnumber]] - AnalysisResults['i1']['StartPoints'][
                                       ind1[eventnumber]]) * (
                                       AnalysisResults['i1']['LocalBaseline'][ind1[eventnumber]] - AnalysisResults['i1']['DeltaI'][ind1[eventnumber]]),
                                   np.ones(buffer) * AnalysisResults['i1']['LocalBaseline'][ind1[eventnumber]]])

            fig = PlotEvent(t[parttoplot], out['i1'][parttoplot], t[parttoplot], out['i2'][parttoplot], fit1=fit1)
            if not divmod(eventnumber+1,200):
                pp.close(fig)
                pp = PdfPages(directory + '_SavedEventsCommon_' + str(count) + '.pdf')
                count+=1
            pp.savefig(fig)
            print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
            fig.clear()
            plt.close(fig)
        pp.close()

    if len(OnlyIndexes['i2']) is not 0:
        # Plot All i2
        pp = PdfPages(directory + '_SavedEventsOnlyi2.pdf')
        ind1 = np.uint64(OnlyIndexes['i2'])

        t = np.arange(0, len(out['i2']))
        t = t / out['samplerate'] * 1e3
        count=1
        for eventnumber in range(len(ind1)):
            parttoplot = np.arange(AnalysisResults['i2']['StartPoints'][ind1[eventnumber]] - buffer,
                                   AnalysisResults['i2']['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)

            fit1 = np.concatenate([np.ones(buffer) * AnalysisResults['i2']['LocalBaseline'][ind1[eventnumber]],
                                   np.ones(AnalysisResults['i2']['EndPoints'][ind1[eventnumber]] - AnalysisResults['i2']['StartPoints'][
                                       ind1[eventnumber]]) * (
                                       AnalysisResults['i2']['LocalBaseline'][ind1[eventnumber]] - AnalysisResults['i2']['DeltaI'][ind1[eventnumber]]),
                                   np.ones(buffer) * AnalysisResults['i2']['LocalBaseline'][ind1[eventnumber]]])

            fig = PlotEvent(t[parttoplot], out['i1'][parttoplot], t[parttoplot], out['i2'][parttoplot], fit2=fit1)
            if not divmod(eventnumber+1,200):
                pp.close(fig)
                pp = PdfPages(directory + '_SavedEventsCommon_' + str(count) + '.pdf')
                count+=1
            pp.savefig(fig)
            print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
            fig.clear()
            plt.close(fig)
        pp.close()

    # Derivative
    if len(CommonIndexes['i1']) is not 0:
        # Plot All i1
        pp = PdfPages(directory + '_i1vsderivi2.pdf')
        ind1 = np.uint64(CommonIndexes['i1'])
        ind2 = np.uint64(CommonIndexes['i2'])

        t = np.arange(0, len(out['i1']))
        t = t / out['samplerate'] * 1e3
        count=1
        for eventnumber in range(len(ind1)):
            parttoplot = np.arange(AnalysisResults['i1']['StartPoints'][ind1[eventnumber]] - buffer,
                                   AnalysisResults['i1']['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)
            parttoplot2 = np.arange(AnalysisResults['i2']['StartPoints'][ind2[eventnumber]] - buffer,
                                    AnalysisResults['i2']['EndPoints'][ind2[eventnumber]] + buffer, 1, dtype=np.uint64)

            fig = PlotEvent(t[parttoplot], out['i1'][parttoplot], t[parttoplot2][:-1],
                               np.diff(out['i2'][parttoplot2]))

            if not divmod(eventnumber+1,200):
                pp.close(fig)
                pp = PdfPages(directory + '_SavedEventsCommon_' + str(count) + '.pdf')
                count+=1
            pp.savefig(fig)
            print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
            fig.clear()
            plt.close(fig)
        pp.close()

def PlotRecursiveLPResults(RoughEventLocations, inp, directory, buffer, channel='i2'):
    pp = PdfPages(directory + '_' + channel + '_DetectedEventsFromLPFilter.pdf')
    a=1
    for i in RoughEventLocations['RoughEventLocations']:
        startp = np.uint64(i[0]-buffer*inp['samplerate'])
        endp = np.uint64(i[1]+buffer*inp['samplerate'])
        t = np.arange(startp, endp)
        t = t / inp['samplerate'] * 1e3
        fig = PlotEvent(t, inp[channel][startp:endp], channel=channel)
        pp.savefig(fig)
        print('{} out of {} saved!'.format(str(a), str(len(RoughEventLocations['RoughEventLocations']))))
        a+=1
        fig.clear()
        plt.close(fig)
    pp.close()

def SaveAllAxopatchEvents(AnalysisResults, directory, out, buffer, withFit = 1):
    # Plot All Common Events
    pp = PdfPages(directory + '_SavedEventsAxopatch.pdf')
    t = np.arange(0, len(out['i1']))
    t = t / out['samplerate'] * 1e3

    for eventnumber in range(AnalysisResults['i1']['NumberOfEvents']):
        parttoplot = np.arange(AnalysisResults['i1']['StartPoints'][eventnumber] - buffer,
                               AnalysisResults['i1']['EndPoints'][eventnumber] + buffer, 1, dtype=np.uint64)

        fit1 = np.concatenate([np.ones(buffer) * AnalysisResults['i1']['LocalBaseline'][eventnumber],
                               np.ones(AnalysisResults['i1']['EndPoints'][eventnumber] -
                                       AnalysisResults['i1']['StartPoints'][
                                           eventnumber]) * (
                                   AnalysisResults['i1']['LocalBaseline'][eventnumber] -
                                   AnalysisResults['i1']['DeltaI'][eventnumber]),
                               np.ones(buffer) * AnalysisResults['i1']['LocalBaseline'][eventnumber]])

        if withFit:
            fig = PlotEvent(t[parttoplot], out['i1'][parttoplot], fit1=fit1)
        else:
            fig = PlotEvent(t[parttoplot], out['i1'][parttoplot])

        pp.savefig(fig)
        print('{} out of {} saved!'.format(str(eventnumber), str(AnalysisResults['i1']['NumberOfEvents'])))
        #print('Length i1: {}, Fit i1: {}'.format(len(out['i1'][parttoplot]), len(fit1)))
        fig.clear()
        plt.close(fig)
    pp.close()