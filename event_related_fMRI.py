import os
from matplotlib.mlab import csv2rec
import matplotlib.pyplot as plt
import nitime
import nitime.timeseries as ts
import nitime.analysis as nta
import nitime.viz as viz

TR = 2.
len_et = 15

data_path=os.path.join(nitime.__path__[0],'data')
data = csv2rec(os.path.join(data_path, 'event_related_fmri.csv'))

t1=ts.TimeSeries(data.bold,sampling_interval=TR)
print(t1.shape)
t2=ts.TimeSeries(data.events,sampling_interval=TR)
print(t2.shape)
E=nta.EventRelatedAnalyzer(t1, t2, len_et)


fig01=viz.plot_tseries(E.eta,ylabel='Bold (change)')
fig02=viz.plot_tseries(E.FIR,ylabel='Bold (change)')
fig03=viz.plot_tseries(E.xcorr_eta,ylabel='Bold (change)')
plt.show()
