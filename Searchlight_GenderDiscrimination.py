import nibabel as nib
import glob
from nilearn.image import new_img_like

for wrarun in list(glob.glob('D:/FaceData/func_img/wrarun*.nii')):
    func_img=nib.load(wrarun)

from nilearn.image import index_img

func_img = index_img(func_img, slice(10, 490, 8))

anat_img=nib.load('D:/FaceData/anat_img/anat.nii')

# Load behavioral data
import numpy as np

for run in list(glob.glob('D:/FaceData/label/run*.txt')):
    behavioral=np.recfromcsv(run,delimiter='')
y=behavioral['name']
session=behavioral['label']
condition_mask= np.logical_or(y == b'Male', y == b'Female')

from nilearn.image import index_img

condition_mask=np.logical_or(y == b'Male', y==b'Female')
fmri_img=index_img(func_img,condition_mask)

y,session = y[condition_mask], session[condition_mask]

from nltools.mask import create_sphere
mask_img=create_sphere([40,-62,-20],radius=7)

# searchlight computation

n_jobs=1

from sklearn.cross_validation import KFold
cv=KFold(y.size,n_folds=4)

import nilearn.decoding
searchlight=nilearn.decoding.SearchLight(
    mask_img,
    radius=5.6, n_jobs=n_jobs,
    verbose=1, cv=cv)

searchlight.fit(fmri_img,y)

from nilearn.input_data import NiftiMasker

nifti_masker = NiftiMasker(mask_img=mask_img, sessions=session,
                           standardize=True,memory='nilearn_cache',
                           memory_level=1)
fmri_masked=nifti_masker.fit_transform(fmri_img)


from nilearn import  image
mean_img = image.mean_img(fmri_img)

from nilearn.plotting import plot_stat_map, plot_img, show

searchlight_img = new_img_like(mean_img,searchlight.scores_)

plot_img(searchlight_img, bg_img=mean_img,title='Searchlight',
         display_mode='z',cut_coords=True,cmap='hot',threshold=.2,
         black_bg=True)





show()


