# Load fmri image
# Note: functions in learn can accept parameters as: image object or fmri filepath
from nilearn.image import load_img
fMRIData = load_img(r'C:\Users\wq\nilearn_data\haxby2001\subj2\bold.nii')

# Gain mask
from nilearn import masking
mask = masking.compute_background_mask(fMRIData)

# Download atlas from internet
from nilearn import datasets
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

# Apply atlas to my data
from nilearn.image import resample_to_img
Atlas = resample_to_img(atlas_filename, mask, interpolation='nearest')

# Gain the TimeSeries
from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=Atlas, standardize=True,
                           memory='nilearn_cache', verbose=5)
time_series = masker.fit_transform(fMRIData)

# Extracting times series to build a functional connectome
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