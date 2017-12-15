from nilearn import datasets
haxby_dataset = datasets.fetch_haxby( )
fmri_filename = haxby_dataset.func[0]
print(fmri_filename)
type(fmri_filename)
print('First subject function Nifti image(4D) are at : %s' %
      fmri_function)
mask_filename = haxby_dataset.mask_vt[0]
from nilearn import plotting
plotting.plot_roi(mask_filename,bg_img=haxby_dataset.anat[0],cmap='Paired')
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename,standardize=True)
fmri_masked = masker.fit_transform(fmri_filename)
print(fmri_masked)
plotting.show()
import numpy as np
behavioral = np.recfromcsv(haxby_dataset.session_target[0],delimiter="")
print(behavioral)
conditions = behavioral['labels']
print(conditions)
condition_mask = np.logical_or(conditions == b'face',conditions == b'cat')
print(condition_mask)
fmri_masked = fmri_masked[condition_mask]
print(fmri_masked.shape)
conditions = conditions[condition_mask]
print(conditions.shape)
cdfrom sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)

svc.fit(fmri_masked,conditions)
prediction = svc.predict(fmri_masked)
print(prediction)

print((prediction == conditions).sum()/ float(len(conditions)))
plotting.show()
