import numpy as np
import scipy
import scipy.signal as sig
import UsefulFunctions as uf
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
Tk().withdraw()
os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
expname = 'Gradient'

filename='/Users/migraf/Desktop/Roche meetings/30B_1MKCl_AxoIV_FemtoOff_1.dat'

output = uf.OpenFile(filename)
directory = (str(os.path.split(filename)[0]) + os.sep + expname + '_SavedImages')
AllData = uf.MakeIVData(output, delay=0.642)

# Plot all the Fits
time=np.arange(len(output['i1']))/output['samplerate']
fig1, ax = plt.subplots(1)
ax.plot(time,output['i1'])
ax2 = ax.twinx()
ax2.plot(time,output['v1'],'y')
ch='i1'
#Loop through the parts
for idx, val in enumerate(AllData[ch]['StartPoint']):
    timepart=np.arange(AllData[ch]['EndPoint'][idx]-val)/output['samplerate']
    ax.plot(val/output['samplerate']+timepart, uf.ExpFunc(timepart, AllData[ch]['ExponentialFitValues'][0][idx], AllData[ch]['ExponentialFitValues'][1][idx], AllData[ch]['ExponentialFitValues'][2][idx]),'r')
fig1.show()
plt.show()




#figIV = plt.figure(1)
#ax1IV = figIV.add_subplot(111)
#ax1IV = uf.PlotIV(output, AllData, current='i2', unit=1e9, axis=ax1IV, WithFit=0)
#figIV.tight_layout()

#figIV.show()
#plt.show()

# Save Figures
#figIV.savefig(directory + os.sep + str(os.path.split(filename)[1]) + 'IV_i1.png', dpi=150)
#figIV.savefig(directory + os.sep + str(os.path.split(filename)[1]) + 'IV_i1.eps')
