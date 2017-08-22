#Input Stuff
import AnalysisParameters as pm
import numpy as np
import scipy
import scipy.signal as sig
import Functions as f
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib.ticker import EngFormatter
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
import pyqtgraph as pg

fontP.set_size('small')
pm.init()
root = Tk()
root.withdraw()
os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')
SaveAsPDF=0
foldername='PSD'

# Load File, empty string prompts a pop.up window for file selection. Else a file-path can be given
root.update()
filenames = askopenfilenames()
root.destroy()

# Make Dir to save images
directory = (str(os.path.split(filenames[0])[0]) + os.sep + foldername)
if not os.path.exists(directory):
    os.makedirs(directory)
if SaveAsPDF:
    pp = PdfPages(directory + os.sep + '_ALLPSD.pdf')

for filename in filenames:
    print(filename)
    #filename = '/Users/migraf/Desktop/TestTrace/07B_10mMKClBoth_1kBDN_BothChannels12.dat'
    inp = f.OpenFile(filename)
    folder = str(os.path.split(filename)[0]) + os.sep +'PSD'
    file = os.sep + str(os.path.split(filename)[1][:-4])

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig1, ax = plt.subplots(1)
    fr, Pxx_den = scipy.signal.periodogram(inp['i1'], inp['samplerate'])
    #f, Pxx_den = scipy.signal.welch(input, samplerate, nperseg=10*256, scaling='spectrum')
    ax.set_ylabel('PSD [pA^2/Hz]')
    ax.set_xlabel('Frequency')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax.plot(fr, Pxx_den*1e24, 'b')
    ax.grid(1)
    ax.autoscale()

    if inp['graphene']:
        fr, Pxx_den = scipy.signal.periodogram(inp['i2'], inp['samplerate'])
        ax.plot(fr, Pxx_den * 1e24, 'r')
        ax.legend(['Ion', 'Transverse'])
        textstr = 'STD ionic: {}\nSTD trans: {}'.format(pg.siFormat(np.std(inp['i1'])), pg.siFormat(np.std(inp['i2'])))
    else:
        textstr = 'STD ionic: {}'.format(pg.siFormat(np.std(inp['i1'])))

    ax.text(0.75, 0.1, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

    if SaveAsPDF:
        pp.savefig(fig1)
    else:
        fig1.savefig(directory + os.sep + str(os.path.split(filename)[1][:-4]) + '_PSD.png')

    fig1.clear()
    ax.clear()
    plt.close(fig1)

if SaveAsPDF:
    pp.close()
