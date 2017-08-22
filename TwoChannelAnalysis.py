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
from tkinter.filedialog import askopenfilename
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
pm.init()
root = Tk()
root.withdraw()
os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

# Load File, empty string prompts a pop.up window for file selection. Else a file-path can be given
root.update()
filenames = askopenfilenames()
root.destroy()

for filename in filenames:
    print(filename)
    #filename = '/Users/migraf/Desktop/TestTrace/07B_10mMKClBoth_1kBDN_BothChannels12.dat'
    inp = f.OpenFile(filename)
    folder = str(os.path.split(filename)[0]) + os.sep +'Event Detections'
    file = os.sep + str(os.path.split(filename)[1][:-4])

    if not os.path.exists(folder):
        os.makedirs(folder)
    directory = folder + file

    #Low Pass Event Detection
    AnalysisResults = {}

    inp['graphene']=0

    if inp['graphene']:
        chan = ['i1', 'i2'] # For looping over the channels
    else:
        chan = ['i1']

    for sig in chan:
        AnalysisResults[sig] = {}
        AnalysisResults[sig]['RoughEventLocations'] = f.RecursiveLowPassFast(inp[sig], pm.coefficients[sig], inp['samplerate'])
        if pm.UpwardsOn: # Upwards detection can be turned on or off
            AnalysisResults[sig + '_Up'] = {}
            AnalysisResults[sig + '_Up']['RoughEventLocations'] = f.RecursiveLowPassFastUp(inp[sig], pm.coefficients[sig], inp['samplerate'])

    #f.PlotRecursiveLPResults(AnalysisResults['i2'], inp, directory, 1e-3, channel='i2')
    #f.PlotRecursiveLPResults(AnalysisResults['i2_Up'], inp, directory+'UP_', 1e-3, channel='i2')

    # Refine the Rough Event Detection done by the LP filter and Add event infos
    AnalysisResults = f.RefinedEventDetection(inp, AnalysisResults, signals=chan, limit=pm.MinimalFittingLimit*inp['samplerate'])

    #Correlate the two channels
    if inp['graphene']:
        (CommonIndexes, OnlyIndexes) = f.CorrelateTheTwoChannels(AnalysisResults, 10e-3*inp['samplerate'])
        print('\n\nAnalysis Done...\nThere are {} common events\n{} Events on i1 only\n{} Events on i2 only'.format(
            len(CommonIndexes['i1']), len(OnlyIndexes['i1']), len(OnlyIndexes['i2'])))
        #Plot The Events
        f.SaveAllPlots(CommonIndexes, OnlyIndexes, AnalysisResults, directory, inp, pm.PlotBuffer)
    else:
        #Plot The Events
        f.SaveAllAxopatchEvents(AnalysisResults, directory, inp, pm.PlotBuffer)