from nilearn import datasets
from nilearn import plotting
# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
# 'func' is a list of filename:one for each subject
fmri_filename = haxby_dataset.func[0]
# print basic information on the internet
print('First subject function nifti images (4D) are at : %s ' %
      fmri_filename)   # 4D data
mask_filename = haxby_dataset.mask_vt[0]
# Let's visualize it, using the subject's anatomical as a
# background
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0],
                cmap ='Paired')
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
# we give the mask a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_filename)
print(fmri_masked.shape)
import numpy as np
# load behavioral information
behavioral = np.recfromcsv(haxby_dataset.session_target[0], delimiter="")
# retrieve the experimental conditions, that we are going to use as prediction targets in the decoding
conditions = behavioral['labels']
print(conditions)
conditions_mask = np.logical_or(conditions == b'face', conditions == b'cat')
# we apply this mask in the same direction to restrict the
# classification to the face vs cat discrimination
fmri_masked = fmri_masked[conditions_mask]
print(fmri_masked.shape)
# we apply the same mask to the targets
conditions = conditions[conditions_mask]
print(conditions.shape)
# as a decoder, we use a Support Vector Classification ,with a linear kernel
# we first create it
from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)
# The svc object is an object that can be fit(on trained)on data with labels,and then predict labels on data without
# we first fit it on the data
svc.fit(fmri_masked, conditions)
# then we can predict the labels from the data
prediction = svc.predict(fmri_masked)
print(prediction)
# let's measure the error rate
print((prediction == conditions).sum() / float(len(conditions)))

plotting.show()






