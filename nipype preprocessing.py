import os # system functions
import nipype.interfaces.fsl as fsl # fsl
import nipype.interfaces.afni as afni # afni
import nipype.pipeline.engine as pe # the workflow and node wrappers
import nipype.interfaces.io as nio # Input/Output
import nipype.interfaces.utility as util # utility

#Preliminaries We need to initialize all variables and paths used in the pipeline.
# Name of subjects folder

subjectsfolder = '/gz_Subjects/'
# Location of experiment directory

experiment_dir = '/Volumes/homes/Shafquat/'
# MNI Location

MNI_2mm = experiment_dir + 'Template/MYtemplate_2mm.nii'

MNI_3mm = experiment_dir + 'Template/MYtemplate_3mm.nii'
# location of data folder
# Count all the subfolders within a given directory

subs = next(os.walk(experiment_dir+subjectsfolder))[1]

subject_list = [] # Initialize an empty list to store subjects

session_list = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5'] # list of session identifiers

# Set a last run based on the list of runs

last_run = session_list[-1] # Make sure to change the hardcoded last_run value within the picklast function below
# list of subject identifiers

for subject in subs:

subject_list.append(subject)
output_dir = 'OUTPUT_serial' # name of output folder

working_dir = 'workingdir_firstSteps_serial' # name of working directory
number_of_slices = 40 # number of slices in volume

TR = 2.0 # time repetition of volume

smoothing_size = 6 # size of FWHM in mm

number_volumes_trim = 8 # size of FWHM in mm