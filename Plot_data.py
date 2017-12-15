import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('bmh')

img=nib.load('D:/FaceData/func_img/wrarun1.nii')
data=img.get_data()

import scipy.signal as sps

tsnr=np.mean(data,-1) /np.std(data,-1)

def plot_tenr(x=data.shape[0]/2,y=data.shape[1]/2,z=data.shape[2]/2):
    fig,axes = plt.subplots(2,2)
    ax=axes[0,0]
    ax.axis('off')
    ax.matshow(tsnr[:,:,z],cmap= mpl.cm.hot)
    ax=axes[0,1]
    ax.axis('off')
    ax.matshow(np.rot90(tsnr[:,y,:]),cmap=mpl.cm.hot)
    ax=axes[1,0]
    ax.axis('off')
    ax.matshow(np.rot90(tsnr[x,:,:]),cmap=mpl.cm.hot)
    ax=axes[1,1]
    ax.plot(data[x,y,z])
    ax.set_xlabel('Time')
    ax.set_ylabel('fMRI signal (a.u.)')
    fig.set_size_inches(10,10)
    return fig
import ipywidgets as wdg
import IPython.display as display

pb_widget=wdg.interactive(plot_tenr,
                          x=wdg.IntSlider(min=1,max=data.shape[0],value=data.shape[0] //2),
                          y=wdg.IntSlider(min=1, max=data.shape[1], value=data.shape[1] //2),
                          z=wdg.IntSlider(min=1, max=data.shape[2], value=data.shape[2] // 2)
                          )
display.display(pb_widget)
plt.show()