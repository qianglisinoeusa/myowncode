print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(10,490,8)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    return plt

import nibabel as nib
import glob
for wrarun in list(glob.glob('D:/FaceData/func_img/wrarun*.nii')):
    func_img=nib.load(wrarun)

from nilearn.image import index_img

func_img=index_img(func_img,slice(10,490,8))

from nltools.mask import create_sphere
mask_img=create_sphere([40,-62,-20],radius=5)

# Transform dimensional form 4D to 2D
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(mask_img=mask_img, standardize=True,
                     memory_level=1)
X = masker.fit_transform(func_img)

import numpy as np

for run in list(glob.glob('D:/FaceData/label/run*.txt')):
    behavioral=np.recfromcsv(run,delimiter='')

y = behavioral['name']

condition_mask= np.logical_or(y == b'Male', y == b'Female')

# Restrict the analysis to male and female

fmri_masked = X[condition_mask]

# Apply the same mask to the target(labels)
y= y[condition_mask]

from sklearn.cross_validation import KFold
cv=KFold(n=len(X),n_folds=7)

title='"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)'
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

plt.show()

