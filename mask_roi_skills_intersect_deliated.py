import nibabel as nib
import numpy as np

func_img = nib.load('D:/FaceData/func_img/wrarun1.nii')
func_data = func_img.get_data()

import nibabel as nib

t_map = nib.load('D:/FaceData/mask/spmT_0001.nii')

anat_img = nib.load('D:/FaceData/anat_img/anat.nii')

from nilearn.image import threshold_img

threshold_value_img = threshold_img(t_map,threshold=4.5)

from nilearn import plotting

plotting.plot_stat_map(threshold_value_img,draw_cross=False, title='threshold image with intensity'

                       'value',colorbar=False)
from scipy import ndimage

dil_mask = ndimage.binary_dilation(threshold_value_img)

from nilearn.plotting import plot_roi,show
plot_roi(anat_img,dil_mask,cut_coords=cut_coords,
         annotate=False,title='Dilated mask')


from nltools.mask import create_sphere

mask_second = create_sphere([30,-72,-6],radius=5)

mask_combined = np.logical_and(threshold_value_img,mask_second)


from nilearn.plotting import plot_roi,show

plot_roi(anat_img, mask_combined,title='Intersect mask',display_mode='ortho',
         cmap='hot')

show()
