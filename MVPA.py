# An (Almost) Complete Analysis in Python
# you don't need to know a lot of python, to do use PyMVPA
# what you need from python that is probably not already installed on your Linux box:
import os,sys,glob
import numpy as N
import pylab as P
from mvpa.suite import*
#1. load fMRI data
subdir = os.path.join(expdir,'Analysis/SVM/PyMVPA',subnum,'atCL_Norm')
sublabeldir = os.path.join(expdir,'Analysis/SVM/PyMVPA/Labels',subnum)
subsavedir = os.path.join(expdir,'Analysis/SVM/PyMVPA',subnum,'CV_maps')
attr_file = "%s/FvM_attr_%s.txt" %(sublabeldir,subnum)
wb_file = "%s/wb_%s.nii.gz" %(subdir,subnum)
attr = SampleAttributes(attr_file)
dataset = NiftiDataset((wb_file),
                       labels = attr.labels,
                       chunks = attr.chunks,
                       mask = os.path.join(roidir,'wb_15.nii.gz'))
#2. detrend data
detrend(dataset,perchunk = True,model = 'constant')
zscore = (dataset,perchunks =True,targetdtype ='float32'
# define Classifier
clf = LinearCSVMC()
cv =CrossValidatedTransferError(
    TransferError(clf),
    NFoldSplitter(),
enable_states = ['confusion'],
harvest_attribs = ['transerror.clf.getSensitivityAnalyzer(force_training = False)()'])
#3. to rnn searchlight anslysis for all voxels in your mask:
# set up searchlight with radius and measure configured above
radius_size =12
sl = Searchlight(cv,radius = radius_size)
# run searchlight on datset
sl_map = sl(dataset)
#concert
sl_test = N.array(sl_map)
sl_cv = 100*(1-sl_test)     #change from 0 to 1 to 100
sl_cv_list = list(sl_cv)
# save data
dataset.map2Nifti(sl_cv_list).save(os.path.join(subsavedir,'FvM_CV_12_wb.nii.gz'))
# run Classification and Collext Sensitivities for each ROI
mean_error = cv(dataset_roi)
print mean_error
mean_cv = 1-mean_error
mean_cv = 100*mean_cv
sens = cv.harvested.values()[0]
sens_ave = N.average(sens.axis = 0)
roi = roi_name.split("_")[0]
roi_save = "%s_%s_sensmap.nii.gz"   %(test_name,roi)
