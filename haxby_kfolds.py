from nilearn import datasets
haxby_dataset = datasets.fetch_haxby( )
fmri_filename = haxby_dataset.func[0]
print('First subject function Nifti image(4D) are at : %s' %
      fmri_filename)
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
condition_mask = np.logical_or(conditions == b'face',conditions == b'house')
fmri_masked = fmri_masked[condition_mask]
print(fmri_masked.shape)
conditions = conditions[condition_mask]
print(conditions.shape)

from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)
from sklearn.cross_validation import KFold
cv = KFold(n = len(fmri_masked),n_folds=5)
for train,test in cv:
    svc.fit(fmri_masked[train],conditions[train])
    prediction = svc.predict(fmri_masked[test])
    coef_ = svc.coef_
    print((prediction == conditions[test]).sum() / float(len(conditions[test])))
    print(coef_)
print(coef_.shape)
coef_img = masker.inverse_transform(coef_)
print(coef_img)
coef_img.to_filename('haxby_svc_weights.nii.gz')
from nilearn.plotting import plot_stat_map,show
plot_stat_map(coef_img,bg_img=haxby_dataset.anat[0],title="SVM weights",display_mode = "xz")
show()


