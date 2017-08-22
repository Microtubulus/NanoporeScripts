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

def Reshape1DTo2D(inputarray, buffersize):
    npieces = np.uint16(len(inputarray)/buffersize)
    voltages = np.array([], dtype=np.float64)
    currents = np.array([], dtype=np.float64)
    #print(npieces)

    for i in range(1, npieces+1):
        if i % 2 == 1:
            currents = np.append(currents, inputarray[(i-1)*buffersize:i*buffersize-1], axis=0)
            #print('Length Currents: {}'.format(len(currents)))
        else:
            voltages = np.append(voltages, inputarray[(i-1)*buffersize:i*buffersize-1], axis=0)
            #print('Length Voltages: {}'.format(len(voltages)))

    v1 = np.ones((len(voltages)), dtype=np.float64)
    i1 = np.ones((len(currents)), dtype=np.float64)
    v1[:]=voltages
    i1[:]=currents

    out = {'v1': v1, 'i1': i1}
    print('Currents:' + str(v1.shape))
    print('Voltages:' + str(i1.shape))
    return out

def CalculatePoreSize(G, L, s):
    return (G+np.sqrt(G*(G+16*L*s/np.pi)))/(2*s)

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

def zoom_factory(ax, base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def PlotData(output):
    if output['type'] == 'Axopatch':
        time=np.float32(np.arange(0, len(output['i1']))/output['samplerate'])
        #plot channel 1
        ch1_current = pg.PlotWidget(title="Current vs time Channel 1")
        ch1_current.plot(time, output['i1'])
        ch1_current.setLabel('left', text='Current', units='A')
        ch1_current.setLabel('bottom', text='Time', units='s')

        ch1_voltage = pg.PlotWidget(title="Voltage vs time Channel 1")
        ch1_voltage.plot(time, output['v1'])
        ch1_voltage.setLabel('left', text='Voltage', units='V')
        ch1_voltage.setLabel('bottom', text='Time', units='s')
        #ch1_voltage.setYLink(ch1_current)
        ch1_voltage.setXLink(ch1_current)
        if output['graphene']:
            # plot channel 1
            ch2_current = pg.PlotWidget(title="Current vs time Channel 2")
            ch2_current.plot(time, output['i2'])
            ch2_current.setLabel('left', text='Current', units='A')
            ch2_current.setLabel('bottom', text='Time', units='s')

            ch2_voltage = pg.PlotWidget(title="Voltage vs time Channel 2")
            ch2_voltage.plot(time, output['v2'])
            ch2_voltage.setLabel('left', text='Voltage', units='V')
            ch2_voltage.setLabel('bottom', text='Time', units='s')
            #ch2_voltage.setYLink(ch2_current)
            ch2_voltage.setXLink(ch2_current)

            fig_handles={'Ch1_Voltage': ch1_voltage, 'Ch2_Voltage': ch2_voltage, 'Ch2_Current': ch2_current, 'Ch1_Current': ch1_current}
            return fig_handles
        else:
            fig_handles = {'Ch1_Voltage': ch1_voltage, 'Ch1_Current': ch1_current, 'Ch2_Voltage': 0, 'Ch2_Current': 0}
            return fig_handles

    if output['type'] == 'ChimeraRaw':
        time = np.float32(np.arange(0, len(output['current']))/output['samplerate'])
        figure = plt.figure('Chimera Raw Current @ {} mV'.format(output['voltage']*1e3))
        plt.plot(time, output['current']*1e9)
        plt.ylabel('Current [nA]')
        plt.xlabel('Time [s]')
        figure.show()
        fig_handles = {'Fig1': figure, 'Fig2': 0, 'Zoom1': 0, 'Zoom2': 0}
        return fig_handles

    if output['type'] == 'ChimeraNotRaw':
        time = np.float32(np.arange(0, len(output['current']))/output['samplerate'])
        figure2 = plt.figure('Chimera Not Raw (Display Save Mode)')
        ax3 = plt.subplot(211)
        ax3.plot(time, output['current'] * 1e9)
        plt.ylabel('Current [nA]')
        ax4 = plt.subplot(212, sharex=ax3)
        ax4.plot(time, output['voltage'] * 1e3)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        f2 = zoom_factory(ax3, 1.5)
        figure2.show()
        fig_handles = {'Fig1': 0, 'Fig2': figure2, 'Zoom1': 0, 'Zoom2': f2}
        return fig_handles

def CutDataIntoVoltageSegments(output):
    sweepedChannel = ''
    if output['type'] == 'ChimeraNotRaw' or (output['type'] == 'Axopatch' and not output['graphene']):
        ChangePoints = np.where(np.diff(output['v1']))[0]
        sweepedChannel = 'v1'
        if len(ChangePoints) is 0:
            print('No voltage sweeps in this file')
            return (0,0)
    elif (output['type'] == 'Axopatch' and output['graphene']):
        ChangePoints = np.where(np.diff(output['v1']))[0]
        sweepedChannel = 'v1'
        if len(ChangePoints) is 0:
            ChangePoints = np.where(np.diff(output['v2']))[0]
            if len(ChangePoints) is 0:
                print('No voltage sweeps in this file')
                return (0,0)
            else:
                sweepedChannel = 'v2'
    print('Cutting into Segments...\n{} change points detected in channel {}...'.format(len(ChangePoints), sweepedChannel))
    return (ChangePoints, sweepedChannel)

def MakeIVData(output, approach = 'mean', delay = 0.7):
    (ChangePoints, sweepedChannel) = CutDataIntoVoltageSegments(output)
    if ChangePoints is 0:
        return 0

    if output['graphene']:
        currents = ['i1', 'i2']
    else:
        currents = ['i1']

    Values = output[sweepedChannel][ChangePoints]
    Values = np.append(Values, output[sweepedChannel][::-1][0])
    delayinpoints = np.int64(delay * output['samplerate'])
    #   Store All Data
    All={}
    for current in currents:
        Item = {}
        l = len(Values)
        Item['Voltage'] = np.zeros(l)
        Item['StartPoint'] = np.zeros(l,dtype=np.uint64)
        Item['EndPoint'] = np.zeros(l, dtype=np.uint64)
        Item['Mean'] = np.zeros(l)
        Item['STD'] = np.zeros(l)
        Item['SweepedChannel'] = sweepedChannel
        Item['YorkFitValues'] = {}
        Item['ExponentialFitValues'] = np.zeros((3, l))
        Item['ExponentialFitSTD'] = np.zeros((3, l))
        # First item
        Item['Voltage'][0] = Values[0]
        trace = output[current][0 + delayinpoints:ChangePoints[0]]
        trace = trace[1-np.isnan(trace)]
        Item['StartPoint'][0] = 0 + delayinpoints
        Item['EndPoint'][0] = ChangePoints[0]
        Item['Mean'][0] = np.mean(trace)
        Item['STD'][0] = np.std(trace)
        (popt, pcov) = MakeExponentialFit(np.arange(len(trace))/output['samplerate'], trace)
        if popt[0]:
            Item['ExponentialFitValues'][:, 0] = popt
            Item['ExponentialFitSTD'][:, 0] = np.sqrt(np.diag(pcov))
        else:
            print('Exponential Fit on for ' + current + ' failed at V=' + str(Item['Voltage'][0]))
            Item['ExponentialFitValues'][:, 0] = np.mean(trace)
            Item['ExponentialFitSTD'][:, 0] = np.std(trace)

        for i in range(1, len(Values) - 1):
            trace = output[current][ChangePoints[i - 1]+delayinpoints:ChangePoints[i]]
            Item['Voltage'][i] = Values[i]
            Item['StartPoint'][i] = ChangePoints[i - 1]+delayinpoints
            Item['EndPoint'][i] = ChangePoints[i]
            Item['Mean'][i] = np.mean(trace)
            Item['STD'][i] = np.std(trace)
            (popt, pcov) = MakeExponentialFit(np.arange(len(trace)) / output['samplerate'], trace)
            if popt[0]:
                Item['ExponentialFitValues'][:, i] = popt
                Item['ExponentialFitSTD'][:, i] = np.sqrt(np.diag(pcov))
            else:
                print('Exponential Fit on for ' + current + ' failed at V=' + str(Item['Voltage'][i]))
                Item['ExponentialFitValues'][:, 0] = np.mean(trace)
                Item['ExponentialFitSTD'][:, 0] = np.std(trace)

        # Last
        if 1:
            trace = output[current][ChangePoints[len(ChangePoints) - 1] + delayinpoints : len(output[current]) - 1]
            Item['Voltage'][-1:] = Values[len(Values) - 1]
            Item['StartPoint'][-1:] = ChangePoints[len(ChangePoints) - 1]+delayinpoints
            Item['EndPoint'][-1:] = len(output[current]) - 1
            Item['Mean'][-1:] = np.mean(trace)
            Item['STD'][-1:] = np.std(trace)
            (popt, pcov) = MakeExponentialFit(np.arange(len(trace)) / output['samplerate'], trace)
            if popt[0]:
                Item['ExponentialFitValues'][:, -1] = popt
                Item['ExponentialFitSTD'][:, -1] = np.sqrt(np.diag(pcov))
            else:
                print('Exponential Fit on for ' + current + ' failed at V=' + str(Item['Voltage'][-1:]))
                Item['ExponentialFitValues'][:, 0] = np.mean(trace)
                Item['ExponentialFitSTD'][:, 0] = np.std(trace)

        sigma_v = 1e-12 * np.ones(len(Item['Voltage']))
        (a, b, sigma_a, sigma_b, b_save) = YorkFit(Item['Voltage'], Item['Mean'], sigma_v, Item['STD'])
        x_fit = np.linspace(min(Item['Voltage']), max(Item['Voltage']), 1000)
        y_fit = scipy.polyval([b, a], x_fit)
        Item['YorkFitValues'] = {'x_fit': x_fit, 'y_fit': y_fit, 'Yintercept': a, 'Slope': b, 'Sigma_Yintercept': sigma_a,
                         'Sigma_Slope': sigma_b, 'Parameter': b_save}
        (a, b, sigma_a, sigma_b, b_save) = YorkFit(Item['Voltage'], Item['ExponentialFitValues'][2,:], sigma_v, Item['ExponentialFitSTD'][2,:])
        y_fit = scipy.polyval([b, a], x_fit)
        Item['YorkFitValuesExponential'] = {'x_fit': x_fit, 'y_fit': y_fit, 'Yintercept': a, 'Slope': b, 'Sigma_Yintercept': sigma_a,
                         'Sigma_Slope': sigma_b, 'Parameter': b_save}
        All[current] = Item
        All['Currents'] = currents
    return All

def PlotIV(output, AllData, current = 'i1', unit=1e9, axis = '', WithFit = 1):
    axis.errorbar(AllData[current]['Voltage'], AllData[current]['Mean']*unit, yerr=AllData[current]['STD']*unit, fmt='o', label=str(os.path.split(output['filename'])[1])[:-4])
    axis.set_ylabel('Current ' + current + ' [nA]')
    axis.set_xlabel('Voltage ' + AllData[current]['SweepedChannel'] +' [V]')
    if WithFit:
        axis.set_title('IV Plot with Fit: G={:.2f}nS'.format(AllData[current]['YorkFitValues']['Slope']*unit))
        axis.plot(AllData[current]['YorkFitValues']['x_fit'], AllData[current]['YorkFitValues']['y_fit']*unit, 'r--', label='Linear Fit')
    else:
        axis.set_title('IV Plot')
    return axis

def ExpFunc(x, a, b, c):
    return a * np.exp(-b * x) + c

def MakeExponentialFit(xdata,ydata):
    try:
        popt, pcov = curve_fit(ExpFunc, xdata, ydata)
        return (popt, pcov)
    except RuntimeError:
        popt = (0,0,0)
        pcov = 0
        return (popt, pcov)

def FitIV(IVData, plot=1, x='i1', y='v1', iv=0):
    sigma_v = 1e-12*np.ones(len(IVData['Voltage']))
    (a, b, sigma_a, sigma_b, b_save) = YorkFit(IVData['Voltage'], IVData['Mean'], sigma_v, IVData['STD'])
    x_fit = np.linspace(min(IVData['Voltage']), max(IVData['Voltage']), 1000)
    y_fit = scipy.polyval([b,a], x_fit)
    if plot:
        spacing = np.sort(IVData['Voltage'])
        #iv = pg.PlotWidget(title='Current-Voltage Plot', background=None)
        err = pg.ErrorBarItem(x=IVData['Voltage'], y=IVData['Mean'], top=IVData['STD'],
                              bottom=IVData['STD'], pen='b', beam=((spacing[1]-spacing[0]))/2)
        iv.addItem(err)
        iv.plot(IVData['Voltage'], IVData['Mean'], symbol='o', pen=None)
        iv.setLabel('left', text=x + ', Current', units='A')
        iv.setLabel('bottom', text=y + ', Voltage', units='V')
        iv.plot(x_fit, y_fit, pen='r')
        textval = pg.siFormat(1/b, precision=5, suffix='Ohm', space=True, error=None, minVal=1e-25, allowUnicode=True)
        textit = pg.TextItem(text=textval, color=(0, 0, 0))
        textit.setPos(min(IVData['Voltage']), max(IVData['Mean']))
        iv.addItem(textit)

    else:
        iv=0
    YorkFitValues={'x_fit': x_fit, 'y_fit': y_fit, 'Yintercept':a, 'Slope':b, 'Sigma_Yintercept':sigma_a, 'Sigma_Slope':sigma_b, 'Parameter':b_save}
    return (YorkFitValues, iv)

def PlotExtractedPart(output, AllData, current = 'i1', unit=1e9, axis = '', axis2 = ''):
    time = np.arange(0, len(output[current])) / output['samplerate']
    axis.plot(time, output[current] * unit, 'b', label=str(os.path.split(output['filename'])[1])[:-4])
    axis.set_ylabel('Current ' + current + ' [nA]')
    axis.set_title('Time Trace')
    for i in range(0, len(AllData[current]['StartPoint'])):
        axis.plot(time[AllData[current]['StartPoint'][i]:AllData[current]['EndPoint'][i]],
                 output[current][AllData[current]['StartPoint'][i]:AllData[current]['EndPoint'][i]] * unit, 'r')
    axis2.plot(time, output[AllData[current]['SweepedChannel']], 'b', label=str(os.path.split(output['filename'])[1])[:-4])
    axis2.set_ylabel('Voltage ' + AllData[current]['SweepedChannel'] + ' [V]')
    axis2.set_xlabel('Time')
    return (axis, axis2)

def ExportIVData(self):
    f = h5py.File(self.matfilename + '_IVData.hdf5', "w")
    ivdata = f.create_group("IVData")
    for k, l in self.IVData.items():
        ivdata.create_dataset(k, data=l)
    fitdata = f.create_group("FitData")
    for k, l in self.FitValues.items():
        fitdata.create_dataset(k, data=l)
    Data={}
    Data['IVFit']=self.FitValues
    Data['IVData']=self.IVData
    scipy.io.savemat(self.matfilename + '_IVData.mat', Data, appendmat=True)

def MakePSD(input, samplerate, fig):
    f, Pxx_den = scipy.signal.periodogram(input, samplerate)
    #f, Pxx_den = scipy.signal.welch(input, samplerate, nperseg=10*256, scaling='spectrum')
    fig.setLabel('left', 'PSD', units='pA^2/Hz')
    fig.setLabel('bottom', 'Frequency', units='Hz')
    fig.setLogMode(x=True, y=True)
    fig.plot(f, Pxx_den*1e24, pen='k')
    return (f,Pxx_den)

def YorkFit(X, Y, sigma_X, sigma_Y, r=0):
    N_itermax=10 #maximum number of interations
    tol=1e-15 #relative tolerance to stop at
    N = len(X)
    temp = np.matrix([X, np.ones(N)])
    #make initial guess at b using linear squares

    tmp = np.matrix(Y)*lin.pinv(temp)
    b_lse = np.array(tmp)[0][0]
    #a_lse=tmp(2);
    b = b_lse #initial guess
    omega_X = np.true_divide(1,np.power(sigma_X,2))
    omega_Y = np.true_divide(1, np.power(sigma_Y,2))
    alpha=np.sqrt(omega_X*omega_Y)
    b_save = np.zeros(N_itermax+1) #vector to save b iterations in
    b_save[0]=b

    for i in np.arange(N_itermax):
        W=omega_X*omega_Y/(omega_X+b*b*omega_Y-2*b*r*alpha)

        X_bar=np.sum(W*X)/np.sum(W)
        Y_bar=np.sum(W*Y)/np.sum(W)

        U=X-X_bar
        V=Y-Y_bar

        beta=W*(U/omega_Y+b*V/omega_X-(b*U+V)*r/alpha)

        b=sum(W*beta*V)/sum(W*beta*U)
        b_save[i+1]=b
        if np.abs((b_save[i+1]-b_save[i])/b_save[i+1]) < tol:
            break

    a=Y_bar-b*X_bar
    x=X_bar+beta
    y=Y_bar+b*beta
    x_bar=sum(W*x)/sum(W)
    y_bar=sum(W*y)/sum(W)
    u=x-x_bar
    #%v=y-y_bar
    sigma_b=np.sqrt(1/sum(W*u*u))
    sigma_a=np.sqrt(1./sum(W)+x_bar*x_bar*sigma_b*sigma_b)
    return (a, b, sigma_a, sigma_b, b_save)

def SaveFigureList(folder, list):
    filename=os.path.splitext(os.path.basename(folder))[0]
    dirname=os.path.dirname(folder)
    for i in list:
        if list[i]:
            list[i].savefig(dirname+os.sep+filename+'_'+i+'.png', format='png')
    return 0

def DoublePlot(self):
    p1 = self.p1.plotItem
    p1_v = self.voltagepl.plotItem
    p1.getAxis('left').setLabel(text='Ionic Current', color='#0000FF', units='A')
    p1_v.getAxis('left').setLabel(text='Ionic Voltage', color='#0000FF', units='V')
    ## create a new ViewBox, link the right axis to its coordinate system
    p1.showAxis('right')
    p1_v.showAxis('right')
    p1.scene().addItem(self.transverseAxis)
    p1_v.scene().addItem(self.transverseAxisVoltage)
    p1.getAxis('right').linkToView(self.transverseAxis)
    p1_v.getAxis('right').linkToView(self.transverseAxisVoltage)
    self.transverseAxis.setXLink(p1)
    self.transverseAxisVoltage.setXLink(p1_v)
    self.transverseAxis.show()
    self.transverseAxisVoltage.show()
    p1.getAxis('right').setLabel(text='Transverse Current', color='#FF0000', units='A')
    p1_v.getAxis('right').setLabel(text='Transverse Voltage', color='#FF0000', units='V')

    def updateViews():
        ## view has resized; update auxiliary views to match
        self.transverseAxis.setGeometry(p1.vb.sceneBoundingRect())
        self.transverseAxisVoltage.linkedViewChanged(p1_v.vb, self.transverseAxisVoltage.XAxis)
        self.transverseAxisVoltage.setGeometry(p1_v.vb.sceneBoundingRect())
        self.transverseAxis.linkedViewChanged(p1.vb, self.transverseAxis.XAxis)

    updateViews()
    p1.vb.sigResized.connect(updateViews)
    p1_v.vb.sigResized.connect(updateViews)
    p1.plot(self.t, self.out['i1'], pen='b')
    p1_v.plot(self.t, self.out['v1'], pen='b')
    self.transverseAxis.addItem(pg.PlotCurveItem(self.t, self.out['i2'], pen='r'))
    self.transverseAxisVoltage.addItem(pg.PlotCurveItem(self.t, self.out['v2'], pen='r'))

    #Set Ranges
    self.transverseAxis.setYRange(np.min(self.out['i2'])-10*np.std(self.out['i2']), np.max(self.out['i2'])+1*np.std(self.out['i2']))
    self.p1.setYRange(np.min(self.out['i1'])-1*np.std(self.out['i1']), np.max(self.out['i1'])+ 10*np.std(self.out['i1']))
    self.p1.enableAutoRange(axis='x')
    self.transverseAxisVoltage.setYRange(np.min(self.out['v2']) - 10/100 * np.mean(self.out['v2']),
                              np.max(self.out['v2']) + 1/100 * np.mean(self.out['v2']))
    self.voltagepl.setYRange(np.min(self.out['v1']) - 1/100 * np.mean(self.out['v1']),
                  np.max(self.out['v1']) + 10/100 * np.mean(self.out['v1']))

def MatplotLibIV(self):
    #IV
    if not hasattr(self, 'IVData'):
        return
    x_fit = np.linspace(min(self.IVData['Voltage']), max(self.IVData['Voltage']), 1000)
    y_fit = scipy.polyval([self.FitValues['Slope'], self.FitValues['Yintercept']], x_fit)
    x_si_params=pg.siScale(max(x_fit))
    y_si_params=pg.siScale(max(y_fit))
    plt.figure(1)
    plt.clf()
    plt.errorbar(self.IVData['Voltage']*x_si_params[0], self.IVData['Mean']*y_si_params[0], fmt='ob', yerr=self.IVData['STD']*y_si_params[0], linestyle='None')
    plt.hold(True)
    textval = pg.siFormat(1 / self.FitValues['Slope'], precision=5, suffix='Ohm', space=True, error=None, minVal=1e-25, allowUnicode=True)
    plt.annotate(textval, [min(self.IVData['Voltage']*x_si_params[0]), max(self.IVData['Mean']*y_si_params[0])])
    plt.plot(x_fit*x_si_params[0], y_fit*y_si_params[0], '-r')
    plt.xlabel('Voltage Channel {} [{}V]'.format(self.xaxisIV+1, x_si_params[1]))
    plt.ylabel('Current Channel {} [{}A]'.format(self.yaxisIV+1, y_si_params[1]))
    filename=os.path.splitext(os.path.basename(self.datafilename))[0]
    dirname=os.path.dirname(self.datafilename)
    plt.savefig(dirname+os.sep+filename+'_IV_' + str(self.xaxisIV) + str(self.yaxisIV) + '.eps')
    plt.savefig(dirname+os.sep+filename+'_IV_' + str(self.xaxisIV) + str(self.yaxisIV) + '.png')
    #plt.show()

def SaveDerivatives(self):
    PartToConsider=np.array(self.eventplot.viewRange()[0])
    partinsamples = np.int64(np.round(self.out['samplerate'] * PartToConsider))
    t = self.t[partinsamples[0]:partinsamples[1]]
    i1part = self.out['i1'][partinsamples[0]:partinsamples[1]]
    i2part = self.out['i2'][partinsamples[0]:partinsamples[1]]

    plt.figure(1, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.plot(t, i1part, 'b')
    plt.title('i1 vs. i2')
    plt.ylabel('Ionic Current [A]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.plot(t, i2part, 'r')
    plt.xlabel('time (s)')
    plt.ylabel('Transverse Current [A]')
    self.pp.savefig()

    plt.figure(2, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.plot(t, i1part, 'b')
    plt.title('i1 vs. its derivative')
    plt.ylabel('Ionic Current [A]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.plot(t[:-1], np.diff(i1part), 'y')
    plt.xlabel('time (s)')
    plt.ylabel('d(Ionic Current [A])/dt')
    self.pp.savefig()

    plt.figure(3, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.plot(t, i2part, 'r')
    plt.title('i2 vs. its derivative')
    plt.ylabel('Transverse Current [A]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.plot(t[:-1], np.diff(i2part), 'y')
    plt.xlabel('time (s)')
    plt.ylabel('d(Transverse Current [A])/dt')
    self.pp.savefig()

def MatplotLibCurrentSignal(self):
    if not hasattr(self, 'out'):
        return
    fig, ax1 = plt.subplots(figsize=(20, 7))
    if self.ui.plotBoth.isChecked():
        ax1.plot(self.t, self.out['i1'], 'b-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_xlabel('time [s]')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Ionic Current [A]', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        ax2 = ax1.twinx()
        ax2.plot(self.t, self.out['i2'], 'r-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_ylim(self.p1.viewRange()[1])
        ax2.set_ylabel('Transverse Current [A]', color='r')
        ax2.set_ylim(self.transverseAxis.viewRange()[1])
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
    elif self.ui.ndChannel.isChecked():
        ax1.plot(self.t, self.out['i2'], 'r-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_ylim(self.p1.viewRange()[1])
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('Transverse Current [A]')
        for tl in ax1.get_yticklabels():
            tl.set_color('r')
    else:
        ax1.plot(self.t, self.out['i1'], 'b-')
        ax1.set_xlim(self.p1.viewRange()[0])
        ax1.set_ylim(self.p1.viewRange()[1])
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('Ionic Current [A]')

    #self.pp.savefig()

    #SaveDerivatives(self)
    #plt.savefig(dirname+os.sep+filename+'_IV_' + str(self.xaxisIV) + str(self.yaxisIV) + '.eps')
    plt.savefig(self.matfilename + str(self.p1.viewRange()[0][0]) + '_Figure.eps')
    #plt.show()

def PlotSingle(self):
    self.p1.clear()
    self.transverseAxis.clear()
    self.p1.plotItem.hideAxis('right')
    self.voltagepl.plotItem.hideAxis('right')
    self.transverseAxisVoltage.clear()
    self.p3.clear()
    self.voltagepl.clear()

    if self.ui.ndChannel.isChecked():
        temp_i = self.out['i2']
        temp_v = self.out['v2']
        self.p1.setLabel('left', text='Transverse Current', units='A')
        self.voltagepl.setLabel('left', text='Transverse Voltage', units='V')
    else:
        temp_i = np.array(self.out['i1'])
        temp_v = self.out['v1']
        self.p1.setLabel('left', text='Ionic Current', units='A')
        self.voltagepl.setLabel('left', text='Ionic Voltage', units='V')

    self.p1.setLabel('bottom', text='Time', units='s')
    print('self.t:' + str(self.t.shape) + ', temp_i:'+str(temp_i.shape))
    self.voltagepl.setLabel('bottom', text='Time', units='s')
    self.p1.plot(self.t, temp_i, pen='b')
    aphy, aphx = np.histogram(temp_i, bins=np.int64(np.round(len(temp_i) / 1000)))
    aphx = aphx

    aphhist = pg.PlotCurveItem(aphx, aphy, stepMode=True, fillLevel=0, brush='b')
    self.p3.addItem(aphhist)

    if self.out['type'] == 'ChimeraRaw':
        self.voltagepl.addLine(y=self.out['v1'], pen='b')
    else:
        self.voltagepl.plot(self.t, temp_v, pen='b')

    self.psdplot.clear()
    print('Samplerate: ' + str(self.out['samplerate']))
    MakePSD(temp_i, self.out['samplerate'], self.psdplot)
    siSamplerate = pg.siScale(self.out['samplerate'])
    siSTD = pg.siScale(np.std(temp_i))

    self.ui.SampleRateLabel.setText(
        'Samplerate: ' + str(self.out['samplerate'] * siSamplerate[0]) + siSamplerate[1] + 'Hz')
    self.ui.STDLabel.setText('STD: ' + str(siSTD[0] * np.std(temp_i)) + siSTD[1] + 'A')

    self.voltagepl.enableAutoRange(axis='y')
    self.p1.enableAutoRange(axis='y')

def SaveToHDF5(self):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "w")
    general = f.create_group("General")
    general.create_dataset('FileName', data=self.out['filename'])
    general.create_dataset('Samplerate', data=self.out['samplerate'])
    general.create_dataset('Machine', data=self.out['type'])
    general.create_dataset('TransverseRecorded', data=self.out['graphene'])
    segmentation_LP = f.create_group("LowPassSegmentation")
    for k,l in self.AnalysisResults.items():
        set1 = segmentation_LP.create_group(k)
        lpset1 = set1.create_group('LowPassSettings')
        for o, p in self.coefficients[k].items():
             lpset1.create_dataset(o, data=p)
        for m, l in self.AnalysisResults[k].items():
             set1.create_dataset(m, data=l)
    # #
    # segmentation_LP.create_dataset('RoughEventLocations', data=self.RoughEventLocations)
    # segmentation_LP.create_dataset('StartPoints', data=self.startpoints)
    # segmentation_LP.create_dataset('EndPoints', data=endpoints)
    # segmentation_LP.create_dataset('LocalBaseline', data=self.localBaseline)
    # segmentation_LP.create_dataset('LocalVariance', data=self.localVariance)
    # segmentation_LP.create_dataset('DeltaI', data=self.deli)
    # segmentation_LP.create_dataset('DwellTime', data=self.dwell)
    # segmentation_LP.create_dataset('FractionalCurrentDrop', data=self.frac)
    # segmentation_LP.create_dataset('Frequency', data=self.dt)
    # segmentation_LP.create_dataset('NumberOfEvents', data=self.numberofevents)

def RecursiveLowPass(signal, coeff):
    Ni = len(signal)
    RoughEventLocations = []
    ml = np.zeros((Ni, 1))
    vl = np.zeros((Ni, 1))
    ml[0] = np.mean(signal)
    vl[0] = np.var(signal)
    i = 0
    NumberOfEvents = 0
    while i < (Ni - 2):
        i += 1
        # local mean low pass filtering
        ml[i] = coeff['a'] * ml[i - 1] + (1 - coeff['a']) * signal[i]
        vl[i] = coeff['a'] * vl[i - 1] + (1 - coeff['a']) * (signal[i] - ml[i])**2
        Sl = ml[i] - coeff['S'] * np.sqrt(vl[i])
        if signal[i + 1] <= Sl:
            NumberOfEvents += 1
            start = 1 + i
            El = ml[i] - coeff['E'] * np.sqrt(vl[i])
            Mm = ml[i]
            Vv = vl[i]
            duration = 0
            while signal[i + 1] < El and i < (Ni - 2) and duration < coeff['eventlengthLimit']:
                duration += 1
                i += 1
            if duration >= coeff['eventlengthLimit'] or i > (Ni - 10):
                NumberOfEvents -= 1
            else:
                k = start
                while signal[k] < Mm and k > 1:
                    k -= 1
                start = k - 1
                k2 = i + 1
                while signal[k2] > Mm:
                    k2 -= 1
                endp = k2 + 1

                RoughEventLocations.append((start,endp,Mm, Vv))
                ml[i] = Mm
                vl[i] = Vv
    return np.array(RoughEventLocations)

def RecursiveLowPassFast(signal, coeff):
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
        while signal[i + 1] < El and i < (Ni - 2) and duration < coeff['eventlengthLimit']:
            duration += 1
            i += 1
        if duration >= coeff['eventlengthLimit'] or i > (Ni - 10):
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

def RecursiveLowPassFastUp(signal, coeff):
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
        while signal[i + 1] > El and i < (Ni - 2) and duration < coeff['eventlengthLimit']:
            duration += 1
            i += 1
        if duration >= coeff['eventlengthLimit'] or i > (Ni - 10):
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

def AddInfoAfterRecursive(self):
    startpoints = np.uint64(self.AnalysisResults[self.sig]['RoughEventLocations'][:, 0])
    endpoints = np.uint64(self.AnalysisResults[self.sig]['RoughEventLocations'][:, 1])
    localBaseline = self.AnalysisResults[self.sig]['RoughEventLocations'][:, 2]
    localVariance = self.AnalysisResults[self.sig]['RoughEventLocations'][:, 3]

    CusumBaseline=500
    numberofevents = len(startpoints)
    self.AnalysisResults[self.sig]['StartPoints'] = startpoints
    self.AnalysisResults[self.sig]['EndPoints'] = endpoints
    self.AnalysisResults[self.sig]['LocalBaseline'] = localBaseline
    self.AnalysisResults[self.sig]['LocalVariance'] = localVariance
    self.AnalysisResults[self.sig]['NumberOfEvents'] = len(startpoints)

    #### Now we want to move the endpoints to be the last minimum for each ####
    #### event so we find all minimas for each event, and set endpoint to last ####

    deli = np.zeros(numberofevents)
    dwell = np.zeros(numberofevents)
    limit=500e-6*self.out['samplerate']
    AllFits={}

    for i in range(numberofevents):
        length = endpoints[i] - startpoints[i]
        if length <= limit and length>3:
            # Impulsion Fit to minimal value
            deli[i] = localBaseline[i] - np.min(self.out[self.sig][startpoints[i]+1:endpoints[i]-1])
            dwell[i] = (endpoints[i] - startpoints[i]) / self.out['samplerate']
        elif length > limit:
            deli[i] = localBaseline[i] - np.mean(self.out[self.sig][startpoints[i]+5:endpoints[i]-5])
            dwell[i] = (endpoints[i] - startpoints[i]) / self.out['samplerate']
            # # Cusum Fit
            # sigma = np.sqrt(localVariance[i])
            # delta = 2e-9
            # h = 1 * delta / sigma
            # (mc, kd, krmv) = CUSUM(self.out[self.sig][startpoints[i]-CusumBaseline:endpoints[i]+CusumBaseline], delta, h)
            # zeroPoint = startpoints[i]-CusumBaseline
            # krmv = krmv+zeroPoint+1
            # AllFits['Event' + str(i)] = {}
            # AllFits['Event' + str(i)]['mc'] = mc
            # AllFits['Event' + str(i)]['krmv'] = krmv
        else:
            deli[i] = localBaseline[i] - np.min(self.out[self.sig][startpoints[i]:endpoints[i]])
            dwell[i] = (endpoints[i] - startpoints[i]) / self.out['samplerate']

    frac = deli / localBaseline
    dt = np.array(0)
    dt = np.append(dt, np.diff(startpoints) / self.out['samplerate'])
    numberofevents = len(dt)

    #self.AnalysisResults[self.sig]['CusumFits'] = AllFits
    AnalysisResults[self.sig]['FractionalCurrentDrop'] = frac
    AnalysisResults[self.sig]['DeltaI'] = deli
    AnalysisResults[self.sig]['DwellTime'] = dwell
    self.AnalysisResults[self.sig]['Frequency'] = dt

def SavingAndPlottingAfterRecursive(self):

    startpoints=self.AnalysisResults[self.sig]['StartPoints']
    endpoints=self.AnalysisResults[self.sig]['EndPoints']
    numberofevents=self.AnalysisResults[self.sig]['NumberOfEvents']
    deli=self.AnalysisResults[self.sig]['DeltaI']
    dwell=self.AnalysisResults[self.sig]['DwellTime']
    frac=self.AnalysisResults[self.sig]['FractionalCurrentDrop']
    dt=self.AnalysisResults[self.sig]['Frequency']
    localBaseline=self.AnalysisResults[self.sig]['LocalBaseline']

    if not self.ui.actionDon_t_Plot_if_slow.isChecked():
        self.p1.clear()
        # Event detection plot, Main Window
        self.p1.plot(self.t, self.out[self.sig], pen='b')
        self.p1.plot(self.t[startpoints],  self.out[self.sig][startpoints], pen=None, symbol='o', symbolBrush='g', symbolSize=10)
        self.p1.plot(self.t[endpoints],  self.out[self.sig][endpoints], pen=None, symbol='o', symbolBrush='r', symbolSize=10)
        #self.p1.plot(self.t[startpoints-10], localBaseline, pen=None, symbol='x', symbolBrush='y', symbolSize=10)

    try:
        self.p2.data = self.p2.data[np.where(np.array(self.sdf.fn) != self.matfilename)]
    except:
        IndexError
    self.sdf = self.sdf[self.sdf.fn != self.matfilename]

    fn = pd.Series([self.matfilename, ] * numberofevents)
    color = pd.Series([self.cb.color(), ] * numberofevents)

    self.sdf = self.sdf.append(pd.DataFrame({'fn': fn, 'color': color, 'deli': deli,
                                             'frac': frac, 'dwell': dwell,
                                             'dt': dt, 'startpoints': startpoints,
                                             'endpoints': endpoints, 'baseline': localBaseline}), ignore_index=True)

    self.p2.addPoints(x=np.log10(dwell), y=frac,
                      symbol='o', brush=(self.cb.color()), pen=None, size=10)

    self.w1.addItem(self.p2)
    self.w1.setLogMode(x=True, y=False)
    self.p1.autoRange()
    self.w1.autoRange()
    self.ui.scatterplot.update()
    self.w1.setRange(yRange=[0, 1])

    colors = self.sdf.color
    for i, x in enumerate(colors):
        fracy, fracx = np.histogram(self.sdf.frac[self.sdf.color == x],
                                    bins=np.linspace(0, 1, int(self.ui.fracbins.text())))
        deliy, delix = np.histogram(self.sdf.deli[self.sdf.color == x],
                                    bins=np.linspace(float(self.ui.delirange0.text()) * 10 ** -9,
                                                     float(self.ui.delirange1.text()) * 10 ** -9,
                                                     int(self.ui.delibins.text())))
        bins_dwell=np.linspace(float(self.ui.dwellrange0.text()), float(self.ui.dwellrange1.text()), int(self.ui.dwellbins.text()))
        dwelly, dwellx = np.histogram(np.log10(self.sdf.dwell[self.sdf.color == x]),
                                      bins=bins_dwell,range=(bins_dwell.min(),bins_dwell.max()))
        dty, dtx = np.histogram(self.sdf.dt[self.sdf.color == x],
                                bins=np.linspace(float(self.ui.dtrange0.text()), float(self.ui.dtrange1.text()),
                                                 int(self.ui.dtbins.text())))

        #            hist = pg.PlotCurveItem(fracy, fracx , stepMode = True, fillLevel=0, brush = x, pen = 'k')
        #            self.w2.addItem(hist)

        hist = pg.BarGraphItem(height=fracy, x0=fracx[:-1], x1=fracx[1:], brush=x)
        self.w2.addItem(hist)

        #            hist = pg.PlotCurveItem(delix, deliy , stepMode = True, fillLevel=0, brush = x, pen = 'k')
        #            self.w3.addItem(hist)

        hist = pg.BarGraphItem(height=deliy, x0=delix[:-1], x1=delix[1:], brush=x)
        self.w3.addItem(hist)
        #            self.w3.autoRange()
        self.w3.setRange(
            xRange=[float(self.ui.delirange0.text()) * 10 ** -9, float(self.ui.delirange1.text()) * 10 ** -9])

        #            hist = pg.PlotCurveItem(dwellx, dwelly , stepMode = True, fillLevel=0, brush = x, pen = 'k')
        #            self.w4.addItem(hist)

        hist = pg.BarGraphItem(height=dwelly, x0=dwellx[:-1], x1=dwellx[1:], brush=x)
        self.w4.addItem(hist)

        #            hist = pg.PlotCurveItem(dtx, dty , stepMode = True, fillLevel=0, brush = x, pen = 'k')
        #            self.w5.addItem(hist)

        hist = pg.BarGraphItem(height=dty, x0=dtx[:-1], x1=dtx[1:], brush=x)
        self.w5.addItem(hist)

def save(self):
    np.savetxt(self.matfilename + 'DB.txt', np.column_stack((self.deli, self.frac, self.dwell, self.dt)),
               delimiter='\t')

def PlotEventSingle(self, clicked=[]):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")
    sig='i1'

    startpoints=self.AnalysisResults[sig]['StartPoints']
    endpoints=self.AnalysisResults[sig]['EndPoints']
    localBaseline=self.AnalysisResults[sig]['LocalBaseline']

    # Reset plot
    self.p3.setLabel('bottom', text='Time', units='s')
    self.p3.setLabel('left', text='Current', units='A')
    self.p3.clear()
    eventnumber = np.int(self.ui.eventnumberentry.text())
    eventbuffer = np.int(self.ui.eventbufferentry.value())

    # plot event trace
    self.p3.plot(self.t[startpoints[eventnumber] - eventbuffer:endpoints[eventnumber] + eventbuffer],
                 self.out[sig][startpoints[eventnumber] - eventbuffer:endpoints[eventnumber] + eventbuffer],
                 pen='b')

    # plot event fit
    self.p3.plot(self.t[startpoints[eventnumber] - eventbuffer:endpoints[eventnumber] + eventbuffer], np.concatenate((
        np.repeat(np.array([localBaseline[eventnumber]]), eventbuffer),
        np.repeat(np.array([localBaseline[eventnumber] - self.AnalysisResults['i1']['DeltaI'][eventnumber
        ]]), endpoints[eventnumber] - startpoints[eventnumber]),
        np.repeat(np.array([localBaseline[eventnumber]]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))

    self.p3.autoRange()

def PlotEventDouble(self, clicked=[]):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")

    if self.ui.actionPlot_i1_detected_only.isChecked():
        indexes = f['LowPassSegmentation/i1/OnlyIndex']
        i = f['LowPassSegmentation/i1/']
        sig = 'i1'
        sig2 = 'i2'
        leftlabel = "Ionic Current"
        rightlabel = "Transverse Current"
    if self.ui.actionPlot_i2_detected_only.isChecked():
        indexes = f['LowPassSegmentation/i2/OnlyIndex']
        i = f['LowPassSegmentation/i2/']
        sig = 'i2'
        sig2 = 'i1'
        rightlabel = "Ionic Current"
        leftlabel = "Transverse Current"
    self.p3.clear()
    self.transverseAxisEvent.clear()

    p1 = self.p3.plotItem
    p1.getAxis('left').setLabel(text=leftlabel, color='#0000FF', units='A')
    ## create a new ViewBox, link the right axis to its coordinate system
    p1.showAxis('right')
    p1.scene().addItem(self.transverseAxisEvent)
    p1.getAxis('right').linkToView(self.transverseAxisEvent)
    self.transverseAxisEvent.setXLink(p1)
    self.transverseAxisEvent.show()
    p1.getAxis('right').setLabel(text=rightlabel, color='#FF0000', units='A')

    def updateViews():
        ## view has resized; update auxiliary views to match
        self.transverseAxisEvent.setGeometry(p1.vb.sceneBoundingRect())
        self.transverseAxisEvent.linkedViewChanged(p1.vb, self.transverseAxisEvent.XAxis)

    updateViews()
    p1.vb.sigResized.connect(updateViews)

    # Correct for user error if non-extistent number is entered
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    maxEvents=self.NumberOfEvents

    eventnumber = np.int(self.ui.eventnumberentry.text())
    if eventnumber >= maxEvents:
        eventnumber=0
        self.ui.eventnumberentry.setText(str(eventnumber))
    elif eventnumber < 0:
        eventnumber=maxEvents
        self.ui.eventnumberentry.setText(str(eventnumber))

    # plot event trace
    parttoplot=np.arange(i['StartPoints'][indexes[eventnumber]] - eventbuffer, i['EndPoints'][indexes[eventnumber]] + eventbuffer,1, dtype=np.uint64)

    p1.plot(self.t[parttoplot], self.out[sig][parttoplot], pen='b')

    # plot event fit
    p1.plot(self.t[parttoplot],
                 np.concatenate((
                     np.repeat(np.array([i['LocalBaseline'][indexes[eventnumber]]]), eventbuffer),
                     np.repeat(np.array([i['LocalBaseline'][indexes[eventnumber]] - i['DeltaI'][indexes[eventnumber]
                     ]]), i['EndPoints'][indexes[eventnumber]] - i['StartPoints'][indexes[eventnumber]]),
                     np.repeat(np.array([i['LocalBaseline'][indexes[eventnumber]]]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))

    # plot 2nd Channel
    if self.Derivative == 'i2':
        self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot][:-1], np.diff(self.out[sig][parttoplot]), pen='r'))
        p1.getAxis('right').setLabel(text='Derivative of i2', color='#FF0000', units='A')
        print('In if...')
        #plt.plot(t[:-1], np.diff(i1part), 'y')
    else:
        self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot], self.out[sig2][parttoplot], pen='r'))

    min1 = np.min(self.out[sig][parttoplot])
    max1 = np.max(self.out[sig][parttoplot])
    self.p3.setYRange(min1-(max1-min1), max1)
    self.p3.enableAutoRange(axis='x')
    min2 = np.min(self.out[sig2][parttoplot])
    max2 = np.max(self.out[sig2][parttoplot])
    self.transverseAxisEvent.setYRange(min2, max2+(max2-min2))

    # Mark event start and end points
    p1.plot([self.t[i['StartPoints'][indexes[eventnumber]]], self.t[i['StartPoints'][indexes[eventnumber]]]],
                 [self.out[sig][i['StartPoints'][indexes[eventnumber]]], self.out[sig][i['StartPoints'][indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='g', symbolSize=12)
    p1.plot([self.t[i['EndPoints'][indexes[eventnumber]]], self.t[i['EndPoints'][indexes[eventnumber]]]],
                 [self.out[sig][i['EndPoints'][indexes[eventnumber]]], self.out[sig][i['EndPoints'][indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='r', symbolSize=12)

    dtime=pg.siFormat(i['DwellTime'][indexes[eventnumber]], precision=5, suffix='s', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dI=pg.siFormat(i['DwellTime'][indexes[eventnumber]], precision=5, suffix='A', space=True, error=None, minVal=1e-25, allowUnicode=True)
    self.ui.eventinfolabel.setText(leftlabel + ': Dwell Time=' + dtime + ', Deli=' + dI)

def PlotEventDoubleFit(self, clicked=[]):
    f = h5py.File(self.matfilename + '_OriginalDB.hdf5', "r")
    i1_indexes=f['LowPassSegmentation/i1/CommonIndex']
    i2_indexes=f['LowPassSegmentation/i2/CommonIndex']
    i1=f['LowPassSegmentation/i1/']
    i2=f['LowPassSegmentation/i2/']

    self.p3.clear()
    self.transverseAxisEvent.clear()

    leftlabel="Ionic Current"
    rightlabel="Transverse Current"

    p1 = self.p3.plotItem
    p1.getAxis('left').setLabel(text=leftlabel, color='#0000FF', units='A')
    ## create a new ViewBox, link the right axis to its coordinate system
    p1.showAxis('right')
    p1.scene().addItem(self.transverseAxisEvent)
    p1.getAxis('right').linkToView(self.transverseAxisEvent)
    self.transverseAxisEvent.setXLink(p1)
    self.transverseAxisEvent.show()
    p1.getAxis('right').setLabel(text=rightlabel, color='#FF0000', units='A')

    def updateViews():
        ## view has resized; update auxiliary views to match
        self.transverseAxisEvent.setGeometry(p1.vb.sceneBoundingRect())
        self.transverseAxisEvent.linkedViewChanged(p1.vb, self.transverseAxisEvent.XAxis)

    updateViews()
    p1.vb.sigResized.connect(updateViews)

    # Correct for user error if non-extistent number is entered
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    maxEvents=len(i1_indexes)

    eventnumber = np.int(self.ui.eventnumberentry.text())
    if eventnumber >= maxEvents:
        eventnumber=0
        self.ui.eventnumberentry.setText(str(eventnumber))
    elif eventnumber < 0:
        eventnumber=maxEvents
        self.ui.eventnumberentry.setText(str(eventnumber))

    # plot event trace
    parttoplot=np.arange(i1['StartPoints'][i1_indexes[eventnumber]] - eventbuffer, i1['EndPoints'][i1_indexes[eventnumber]] + eventbuffer,1, dtype=np.uint64)
    parttoplot2=np.arange(i2['StartPoints'][i2_indexes[eventnumber]] - eventbuffer, i2['EndPoints'][i2_indexes[eventnumber]] + eventbuffer,1, dtype=np.uint64)

    p1.plot(self.t[parttoplot], self.out['i1'][parttoplot], pen='b')

    # plot event fit
    p1.plot(self.t[parttoplot],
                 np.concatenate((
                     np.repeat(np.array([i1['LocalBaseline'][i1_indexes[eventnumber]]]), eventbuffer),
                     np.repeat(np.array([i1['LocalBaseline'][i1_indexes[eventnumber]] - i1['DeltaI'][i1_indexes[eventnumber]
                     ]]), i1['EndPoints'][i1_indexes[eventnumber]] - i1['StartPoints'][i1_indexes[eventnumber]]),
                     np.repeat(np.array([i1['LocalBaseline'][i1_indexes[eventnumber]]]), eventbuffer)), 0),
                 pen=pg.mkPen(color=(173, 27, 183), width=3))

    # plot 2nd Channel
    self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot2], self.out['i2'][parttoplot2], pen='r'))
    self.transverseAxisEvent.addItem(pg.PlotCurveItem(self.t[parttoplot2], np.concatenate((np.repeat(
        np.array([i2['LocalBaseline'][i2_indexes[eventnumber]]]), eventbuffer), np.repeat(
        np.array([i2['LocalBaseline'][i2_indexes[eventnumber]] - i2['DeltaI'][i2_indexes[eventnumber]]]),
        i2['EndPoints'][i2_indexes[eventnumber]] - i2['StartPoints'][i2_indexes[eventnumber]]), np.repeat(
        np.array([i2['LocalBaseline'][i2_indexes[eventnumber]]]), eventbuffer)), 0), pen=pg.mkPen(color=(173, 27, 183), width=3)))

    min1 = np.min(self.out['i1'][parttoplot])
    max1 = np.max(self.out['i1'][parttoplot])
    self.p3.setYRange(min1-(max1-min1), max1)
    self.p3.enableAutoRange(axis='x')
    min2 = np.min(self.out['i2'][parttoplot2])
    max2 = np.max(self.out['i2'][parttoplot2])
    self.transverseAxisEvent.setYRange(min2, max2+(max2-min2))

    # Mark event start and end points
    p1.plot([self.t[i1['StartPoints'][i1_indexes[eventnumber]]], self.t[i1['StartPoints'][i1_indexes[eventnumber]]]],
                 [self.out['i1'][i1['StartPoints'][i1_indexes[eventnumber]]], self.out['i1'][i1['StartPoints'][i1_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='g', symbolSize=12)
    p1.plot([self.t[i1['EndPoints'][i1_indexes[eventnumber]]], self.t[i1['EndPoints'][i1_indexes[eventnumber]]]],
                 [self.out['i1'][i1['EndPoints'][i1_indexes[eventnumber]]], self.out['i1'][i1['EndPoints'][i1_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='r', symbolSize=12)

    self.transverseAxisEvent.addItem(pg.PlotCurveItem([self.t[i2['StartPoints'][i2_indexes[eventnumber]]], self.t[i2['StartPoints'][i2_indexes[eventnumber]]]],
                 [self.out['i2'][i2['StartPoints'][i2_indexes[eventnumber]]], self.out['i2'][i1['StartPoints'][i2_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='g', symbolSize=12))
    self.transverseAxisEvent.addItem(pg.PlotCurveItem([self.t[i2['EndPoints'][i2_indexes[eventnumber]]], self.t[i2['EndPoints'][i2_indexes[eventnumber]]]],
                 [self.out['i2'][i1['EndPoints'][i2_indexes[eventnumber]]], self.out['i2'][i1['EndPoints'][i2_indexes[eventnumber]]]],
                 pen=None,
                 symbol='o', symbolBrush='r', symbolSize=12))

    dtime=pg.siFormat(i1['DwellTime'][i1_indexes[eventnumber]], precision=5, suffix='s', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dI=pg.siFormat(i1['DwellTime'][i1_indexes[eventnumber]], precision=5, suffix='A', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dtime2=pg.siFormat(i2['DwellTime'][i2_indexes[eventnumber]], precision=5, suffix='s', space=True, error=None, minVal=1e-25, allowUnicode=True)
    dI2=pg.siFormat(i2['DwellTime'][i2_indexes[eventnumber]], precision=5, suffix='A', space=True, error=None, minVal=1e-25, allowUnicode=True)

    self.ui.eventinfolabel.setText('Ionic Dwell Time=' + dtime + ',   Ionic Deli=' + dI + ', ' 'Trans Dwell Time=' + dtime2 + ',   Trans Deli=' + dI2)

def SaveEventPlotMatplot(self):
    eventbuffer = np.int(self.ui.eventbufferentry.value())
    eventnumber = np.int(self.ui.eventnumberentry.text())

    parttoplot = np.arange(i['StartPoints'][indexes[eventnumber]][eventnumber] - eventbuffer, i['EndPoints'][indexes[eventnumber]][eventnumber] + eventbuffer, 1,
                           dtype=np.uint64)

    t=np.arange(0,len(parttoplot))
    t=t/self.out['samplerate']*1e3

    fig1=plt.figure(1, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.cla
    plt.plot(t, self.out['i1'][parttoplot]*1e9, 'b')
    plt.ylabel('Ionic Current [nA]')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(2, 1, 2)
    plt.cla
    plt.plot(t, self.out['i2'][parttoplot]*1e9, 'r')
    plt.ylabel('Transverse Current [nA]')
    plt.xlabel('time (ms)')
    self.pp.savefig()
    fig1.clear()

def CombineTheTwoChannels(file):
    f = h5py.File(file, 'a')
    i1 = f['LowPassSegmentation/i1/']
    i2 = f['LowPassSegmentation/i2/']
    i1StartP = i1['StartPoints'][:]
    i2StartP = i2['StartPoints'][:]

    # Common Events
    # Take Longer
    CommonEventsi1Index = np.array([], dtype=np.uint64)
    CommonEventsi2Index = np.array([], dtype=np.uint64)
    DelayLimit = 10e-3*f['General/Samplerate/'].value

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


    # for i in range(len(i1StartP)):
    #     for j in range(len(i2StartP)):
    #         if np.absolute(i1StartP[i] - i2StartP[j]) < DelayLimit:
    #             CommonEventsi1Index = np.append(CommonEventsi1Index, i)
    #             CommonEventsi2Index = np.append(CommonEventsi2Index, j)

    # Only i1
    Onlyi1Indexes = np.delete(range(len(i1StartP)), CommonEventsi1Index)
    # Only i2
    Onlyi2Indexes = np.delete(range(len(i2StartP)), CommonEventsi2Index)

    e = "CommonIndex" in i1
    if e:
        del i1['CommonIndex']
        i1.create_dataset('CommonIndex', data=CommonEventsi1Index)
        del i2['CommonIndex']
        i2.create_dataset('CommonIndex', data=CommonEventsi2Index)
        del i1['OnlyIndex']
        i1.create_dataset('OnlyIndex', data=Onlyi1Indexes)
        del i2['OnlyIndex']
        i2.create_dataset('OnlyIndex', data=Onlyi2Indexes)
    else:
        i1.create_dataset('CommonIndex', data=CommonEventsi1Index)
        i2.create_dataset('CommonIndex', data=CommonEventsi2Index)
        i1.create_dataset('OnlyIndex', data=Onlyi1Indexes)
        i2.create_dataset('OnlyIndex', data=Onlyi2Indexes)

    CommonIndexes={}
    CommonIndexes['i1']=CommonEventsi1Index
    CommonIndexes['i2']=CommonEventsi2Index
    OnlyIndexes={}
    OnlyIndexes['i1'] = Onlyi1Indexes
    OnlyIndexes['i2'] = Onlyi2Indexes
    return (CommonIndexes, OnlyIndexes)

def PlotEvent(t1, t2, i1, i2, fit1 = np.array([]), fit2 = np.array([])):
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

def EditInfoText(self):
    text2='ionic: {} events, trans: {} events\n'.format(str(self.AnalysisResults['i1']['RoughEventLocations'].shape[0]), str(self.AnalysisResults['i2']['RoughEventLocations'].shape[0]))
    text1='The file contains:\n{} Common Events\n{} Ionic Only Events\n{} Transverse Only Events'.format(len(self.CommonIndexes['i1']), len(self.OnlyIndexes['i1']), len(self.OnlyIndexes['i2']))
    self.ui.InfoTexts.setText(text2+text1)

def creation_date(path_to_file):
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_mtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

def CUSUM(input, delta, h):
    Nd = 0
    kd = len(input)
    krmv = len(input)
    k0 = 0
    k = 0
    l = len(input)
    m = np.zeros(l)
    m[k0] = input[k0]
    v = np.zeros(l)
    sp = np.zeros(l)
    Sp = np.zeros(l)
    gp = np.zeros(l)
    sn = np.zeros(l)
    Sn = np.zeros(l)
    gn = np.zeros(l)

    while k < l:
        m[k] = np.mean(input[k0:k])
        v[k] = np.var(input[k0:k])

        sp[k] = delta / v[k] * (input[k] - m[k] - delta / 2)
        sn[k] = -delta / v[k] * (input[k] - m[k] + delta / 2)

        Sp[k] = Sp[k - 1] + sp[k]
        Sn[k] = Sn[k - 1] + sn[k]

        gp[k] = np.max(gp[k - 1] + sp[k], 0)
        gn[k] = np.max(gn[k - 1] + sn[k], 0)

        if gp[k] > h or gn[k] > h:
            kd[Nd] = k
            kmin = int(np.where(Sn == np.min(Sn[k0:k]))[0])
            krmv[Nd] = kmin + k0 - 1
            if gp(k) > h:
                kmin = int(np.where(Sn == np.min(Sn[k0:k]))[0])
                krmv[Nd] = kmin + k0 - 1
            k0 = k
            m[k0] = input[k0]
            v[k0] = 0
            sp[k0] = 0
            Sp[k0] = 0
            gp[k0] = 0
            sn[k0] = 0
            Sn[k0] = 0
            gn[k0] = 0
            Nd = Nd + 1
        k += 1

    if Nd == 0:
        mc = np.mean(input) * np.ones(k)
    elif Nd == 1:
        mc = [m[krmv[0]] * np.ones(krmv[0]), m[k] * np.ones(k - krmv[0])]
    else:
        mc = m[krmv[0]] * np.ones(krmv[0])
        for ii in range(1, Nd):
            mc = [mc, m[krmv[ii]] * np.ones(krmv[ii] - krmv[ii - 1])]
        mc = [mc, m(k) * np.ones(k - krmv[Nd])]
    return (mc, kd, krmv)

def CombineEventDatabases(filename, DBfiles):
    f = h5py.File(filename, "w")

    general = f.create_group("RawDB")
    general.create_dataset('FileName', data=self.out['filename'])
    general.create_dataset('Samplerate', data=self.out['samplerate'])
    general.create_dataset('Machine', data=self.out['type'])
    general.create_dataset('TransverseRecorded', data=self.out['graphene'])
    segmentation_LP = f.create_group("LowPassSegmentation")
    for k, l in self.AnalysisResults.items():
        set1 = segmentation_LP.create_group(k)
        lpset1 = set1.create_group('LowPassSettings')
        for o, p in self.coefficients[k].items():
            lpset1.create_dataset(o, data=p)
        for m, l in self.AnalysisResults[k].items():
            set1.create_dataset(m, data=l)