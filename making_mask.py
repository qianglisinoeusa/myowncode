import nibabel as nib
anat_img = nib.load('D:/FaceData/wmT1.nii')

from nltools.mask import create_sphere
mask = create_sphere([42,-58,-17],radius = 30)
mask_data = anat_img.apply_mask(mask)
