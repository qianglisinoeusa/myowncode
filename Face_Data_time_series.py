# Load the data
#-----------------------------------------------------------------
import nibabel as nib
img=nib.load('D:/FaceData/func_img/wrarun1.nii')

data=img.get_data()

# The time series of center voxel

center_voxel_time_series=data[43,-78,-13,:]
print(center_voxel_time_series)

import numpy as np
# Select the last dimension
mean_tseries=np.mean(data,axis=-1)
std_tseries=np.std(data,axis=-1)
print(mean_tseries.shape)

#-------------------------------------------------------------------
# Plot the data
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1)
ax.plot(data[43,-78,-13,:])

mpl.style.use('bmh')

fig,ax =plt.subplots(1)

ax.plot(data[43,-78,-13,:])
ax.set_xlabel('Time (TR)')
ax.set_ylabel('MRI signal (a.u.)')
ax.set_title('Time-series from voxel[43,-78,-13]')
fig.set_size_inches([12,6])

#----------------------------------------------------------------
# Impressions about the data
fig, ax = plt.subplots(1)
ax.plot(data[43, -78, -13, :])
ax.plot(data[-41, -80, -12, :])
ax.plot(data[42, -52, -20, :])
ax.plot(data[-40, -54, -20, :])
ax.plot(data[43, -24, -25, :])
ax.plot(data[-42, -26, -23, :])
ax.plot(data[55, -59, 7, :])
ax.plot(data[-57, -62, 9, :])
ax.plot(data[54, -38, 4, :])
ax.plot(data[-58, -41, 4, :])
ax.plot(data[55, -7, -15, :])
ax.plot(data[-58, -6, -16, :])
ax.set_xlabel('Time (TR)')
ax.set_ylabel('MRI signal (a.u.)')
ax.set_title('Time-series from a few voxels')
fig.set_size_inches([12, 6])

#---------------------------------------------------------------------
# different subplots for each time-series
fig, ax = plt.subplots(3,2)   # now ax is array
ax[0, 0].plot(data[43, -78, -13, :])
ax[0, 1].plot(data[-41, -80, -12, :])
ax[1, 0].plot(data[42, -52, -20, :])
ax[1, 1].plot(data[-40, -54, -20, :])
ax[2, 0].plot(data[43, -24, -25, :])
ax[2, 1].plot(data[-42, -26, -23, :])

ax[2, 0].set_xlabel('Time (TR)')
ax[2, 1].set_xlabel('Time (TR)')
ax[0, 0].set_ylabel('MRI signal (a.u.)')
ax[1, 0].set_ylabel('MRI signal (a.u.)')
ax[2, 0].set_ylabel('MRI signal (a.u.)')
# Note that we now set the title through the fig object!
fig.suptitle('Time-series from a few voxels')
fig.set_size_inches([16, 10])
# ---------------------------------------------------------------------------
plt.show()

#-----------------------------------------------------------------------------
# Plot an image
fig, ax = plt.subplots(1,2)
ax[0].matshow(np.mean(data[:,:,-13],-1),cmap=mpl.cm.hot)
ax[0].axis('off')
ax[1].matshow(np.std(data[:,:,-13],-1),cmap=mpl.cm.hot)
ax[1].axis('off')
fig.set_size_inches([12,6])
fig.savefig('mean_and_std.png')

# -----------------------------------------------------------------------------
# There are many other kinds of figures you could create:
fig,ax=plt.subplots(2,2)
ax[0,0].hist(np.ravel(data))
ax[0,0].set_xlabel("fMRI signal")
ax[0,0].set_ylabel("# voxels")
# bars are 0.8 wide:
ax[0,1].bar([0.6,1.6,2.6,3.6], [np.mean(data[:,:,-13]),np.mean(data[:,:,-12]),
                                np.mean(data[:,:,-20]),np.mean(data[:,:,-25])])
ax[0,1].set_ylabel("Average signal in the slice")
ax[0,1].set_xticks([1,2,3,4])
ax[0,1].set_xticklabels(["1","2","3","4"])
ax[0,1].set_xlabel("Slice #")

# Compare subsequent time-points
ax[1,0].scatter(data[:,:,-13,0],data[:,:,-13,1])
ax[1,0].set_xlabel("fMRI signal (time-point 0)")
ax[1,0].set_ylabel("fMRI signal (time-point 1)")

# '.T' denotes a transposition
ax[1,1].boxplot(data[32,32].T)
fig.set_size_inches([12,12])
ax[1,1].set_xlabel("Position")
ax[1,1].set_ylabel("fMRI signal")

plt.show()