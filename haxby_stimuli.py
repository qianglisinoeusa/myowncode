import matplotlib.pyplot as plt
from nilearn import datasets
from scipy.misc import imread
haxby_dataset = datasets.fetch_haxby(subjects=[],fetch_stimuli=True)
stimulus_information = haxby_dataset.stimuli
for stim_type in sorted(stimulus_information.keys()):
    if stim_type ==b'controls':
        # skip control image, there too many
        continue
        file_name = stimulus_information[stim_type]
        plt.figure()
        for i in range(48):
            plot.subplot(6,8,i+1)
            try:
                plt.imshow(imread(file_name[i]),cmap = plt.cm.gray)
            except:
                   # just go to next one if the file is not present
              pass
        plt.axis("off")
    plt.suptitle(stim_type)
plt.show()



