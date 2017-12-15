import os
import numpy as np
import matplotlib.pyplot as plt

import nitime.timeseries as ts
import nitime.analysis as nta
import nitime.viz as viz

try:
    from nibabel import load
except ImportError:
    raise ImportError ('You need nibabel (http:/nipy.org/nibabel/) in order to run this example')

data_path=test_dir_path=os.path.join('D:/FaceData/','func_img')
func_img=os.path.join(data_path,'wrarun1.nii')
func_img=load(func_img)
from nltools.mask import create_sphere
mask_img = create_sphere([-58,-41,4],radius=3)
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(mask_img=mask_img, standardize=True,
                     memory_level=1)
func_img = masker.fit_transform(func_img)


behavioral = np.recfromcsv('D:/FaceData/label/run1.txt', delimiter='')
condition = behavioral['name']
onset = behavioral['onset']
import pandas as pd
events = pd.DataFrame({'onset': onset, 'trial_type': condition})

# from nilearn.image import index_img
# func_img=index_img(func_img, onset)
# data=func_img.get_data()
#------------------------------------------------------------------------
# Create the mask image
#------------------------------------------------------------------------
#------------------------------------------------------------------------
TR = 1.
len_et = 494
n_jobs = -1

t1 = ts.TimeSeries(func_img, sampling_interval=TR)

t2=ts.TimeSeries(onset, sampling_interval=TR)

E=nta.EventRelatedAnalyzer(t1, t2, len_et, n_jobs)

# fig01=viz.plot_tseries(E.eta, ylabel='BOLD(% signal'
#                                    'change)', yerror=E.ets)

fig02=viz.plot_tseries(E.FIR, ylabel='BOLD (% signal change)')

fig03=viz.plot_tseries(E.xcorr_eta, ylabel='BOLD (% signal change)')

plt.show()






