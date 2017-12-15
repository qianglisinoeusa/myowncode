# Retrive the dataset
from nilearn.image import load_img

func_img=load_img('D:/FaceData/func_img/wrarun1.nii')

# Coordinates of face selective areas
ffa_coords=[(40,-62,-20)]
labels=['ffa']

from nilearn import input_data

masker=input_data.NiftiSpheresMasker(ffa_coords,radius=5,detrend=True,standardize=True,
                              low_pass=0.1,high_pass=0.01,t_r=1,
                              memory='nilearn_cache',memory_level=1,verbose=2)
time_series=masker.fit_transform(func_img)

import matplotlib.pyplot as plt
plt.plot(time_series)
plt.title('Time Series')
plt.xlabel('Scan Number')
plt.ylabel('Normalized Signal')

plt.legend()

plt.tight_layout()
plt.show()


