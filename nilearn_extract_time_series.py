# Load the fMRI imaging

from nilearn.image import load_img

fmri_data=load_img('D:/FaceData/func_img/wrarun1.nii')

# Gain the mask

from nilearn import masking

mask=masking.compute_background_mask(fmri_data)

# Download the atlas from the internet

from nilearn import datasets

dataset=datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

atlas_filename=dataset.maps
labels=dataset.labels

# Apply atlas to my data

from nilearn.image import resample_to_img

atlas=resample_to_img(atlas_filename,mask,interpolation='nearest')

# Gain the timeseries

from nilearn.input_data import NiftiLabelsMasker

masker=NiftiLabelsMasker(labels_img=atlas,standardize=True,
                         memory='nilearn_cache',verbose=5)

time_series = masker.fit_transform(fmri_data)

import matplotlib.pyplot as plt

plt.plot(time_series)
plt.show()