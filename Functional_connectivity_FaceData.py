'''
extracting TimeSeries from "probabilistic atlas"
'''
# Load fmri image
# Note: functions in learn can accept parameters as: image object or fmri filepath
from nilearn.image import load_img
fMRIData = load_img(r'D:/FaceData/func_img/wrarun1.nii')

# Gain mask
from nilearn import masking
mask = masking.compute_background_mask(fMRIData)

# Download atlas from internet
# Retrieve the atlas and the data
from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

# Apply atlas to my data
from nilearn.image import resample_to_img
Atlas = resample_to_img(atlas_filename, mask, interpolation='continuous')

# Gain the TimeSeries
from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=Atlas, standardize=True,
                         memory='nilearn_cache', verbose=5)

time_series = masker.fit_transform(fMRIData)

############################################################################
# Build and display a correlation matrix
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Display the correlation matrix
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)
plt.colorbar()
# And display the labels
x_ticks = plt.xticks(range(len(labels)), labels, rotation=90)
y_ticks = plt.yticks(range(len(labels)), labels)

############################################################################
# And now display the corresponding graph
from nilearn import plotting
coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="70%", colorbar=True)

plotting.show()
