import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt

for i in list(glob.glob('D:/FaceData/func_img/wrarun*.nii')):
    func_img=nib.load(i)

from nilearn.image import index_img

func_img=index_img(func_img,slice(10,490,8))

anat_img=nib.load('D:/FaceData/anat_img/anat.nii')

# Create the mask image

# from nltools.mask import create_sphere
# mask_img= create_sphere([-29,-71,-4],radius=5)
from nilearn.masking import compute_epi_mask
mask_img=compute_epi_mask(func_img)

# from nilearn import datasets

# dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# atlas_filename = dataset.maps

from nilearn.plotting import plot_stat_map,plot_roi,show
# plot_roi(atlas_filename,title='Harvard-Oxford-Atlas')
# Transform dimensional form 4D to 2D
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(mask_img=mask_img, standardize=True,
                     memory_level=1)
fmri_masked = masker.fit_transform(func_img)

# Load behavioral data
for j in list(glob.glob('D:/FaceData/label/run*.txt')):
    behavioral=np.recfromcsv(j,delimiter='')

conditions = behavioral['name']

condition_mask= np.logical_or(conditions == b'Male', conditions == b'Female')

# Restrict the analysis to male and female

fmri_masked = fmri_masked[condition_mask]

# Apply the same mask to the target(labels)
conditions= conditions[condition_mask]

# Notes:neuroimaging demension is very important, actually above the code, we
# should very carefually the demension of the neuroimaging. of course, the label
# of neuroimaging also is very import.

# machine learning with SVM
X,y= fmri_masked,conditions
from sklearn.svm import SVC
svc = SVC(kernel='linear')

# from sklearn.feature_selection import SelectPercentile, f_classif
# feature_selection = SelectPercentile(f_classif, percentile=7)

# from sklearn.pipeline import Pipeline
# anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])


from sklearn.cross_validation import KFold
cv=KFold(n=len(X),n_folds=12)
for train,test in cv:
    svc.fit(fmri_masked[train],conditions[train])
    prediction=svc.predict(fmri_masked[test])
    print ((prediction==conditions[test]).sum() /float(len(conditions[test])))


coef_=svc.coef_

# coef=feature_selection.inverse_transform(coef_)
weight_img = masker.inverse_transform(coef_)

# Plotting the learning curves
plot_stat_map(weight_img,anat_img,display_mode='ortho',draw_cross=False,
              title='Harvard-Oxford_Atlas')

title = "Harvard-Oxford_Atlas Learning Curves (SVM, Linear kernel)"
n_jobs=1
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    svc, X, y, cv=cv, n_jobs=n_jobs,train_sizes=np.array([5,7,11,15,19,23,27,29,33]))
plt.figure()
plt.title(title)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")

plt.show()

show()