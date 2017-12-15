from nilearn import masking
mask = masking.compute_epi_mask(r'D:\Program Files\MATLAB\R2015b\bin\data\fMRI4D_motor.nii')
print(mask.get_data().shape)
from nilearn.masking import apply_mask
masked_data = apply_mask(r'D:\Program Files\MATLAB\R2015b\bin\data\fMRI4D_motor.nii', mask)
print(masked_data.shape)
# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels

# And now plot a few of these
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.plot(masked_data[:230, 98:100])
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.xlim(0, 150)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
# plt.show()

from nilearn.input_data import NiftiLabelsMasker
# Download atlas from internet
from nilearn import datasets
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels
from nilearn.image import resample_to_img
Atlas = resample_to_img(atlas_filename, mask, interpolation='nearest')

masker = NiftiLabelsMasker(labels_img=Atlas, standardize=True, memory='nilearn_cache', verbose=5)

time_series = masker.fit_transform\
    (r'D:\Program Files\MATLAB\R2015b\bin\data\fMRI4D_motor.nii')

from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Plot the correlation matrix
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)

# Add labels and adjust margins
x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)

plt.show()
