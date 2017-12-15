# Load the fMRI imaging

import nibabel as nib
import numpy as np

func_img=nib.load('D:/FaceData/func_img/wrarun1.nii')

# Gain the mask
from nltools.mask import create_sphere
mask_img= create_sphere([-29, -71, -4], radius=5)
# from nilearn.masking import compute_epi_mask
# mask_img=compute_epi_mask(func_icamcg)

# Transform dimensional form 4D to 2D
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(mask_img=mask_img, standardize=True,
                     memory_level=1)
time_series = masker.fit_transform(func_img)


import matplotlib.pyplot as plt

plt.plot(time_series)

plt.show()