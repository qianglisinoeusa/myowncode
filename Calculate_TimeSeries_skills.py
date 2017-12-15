import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

img=nib.load('D:/FaceData/func_img/wrarun1.nii')
data=img.get_data()
mpl.style.use('bmh')

fig, ax =plt.subplots(1)
ax.plot(data[32,32,15])
ax.set_xlabel('Time(TR)')
ax.set_ylabel('fMRIM signal(a.u.)')

import scipy
import scipy.fftpack as fft
TR=2.4
sampling_rate=1/TR
Nyquist_freq=sampling_rate/2
freq_band=np.linspace(0,Nyquist_freq,data.shape[-1]/2+1)

print(freq_band)
f_data1=fft.fft(data)
print(f_data1.shape)

plt.plot(np.abs(f_data1[32,32,15]))

f_data = f_data1[...,:data.shape[-1]/2+1]
p_data=np.abs(f_data)
freqs=np.linspace(0,Nyquist_freq,data.shape[-1]/2+1)

fig, ax=plt.subplots(1)
ax.plot(freqs,p_data[32,32,15])

fig, ax=plt.subplot(1)
ax.plot(freqs[1:],p_data[32,32,15][1:])
ax.set_xlabel('Frequency(Hz)')
ax.set_ylabel('Power')

import scipy.signal as sps
detrend_data=sps.detrend(data)

detrend_data_power=np.abs(fft.fft(detrend_data))[32,32,15][:data.shape[-1]/2+1]
print(detrend_data_power.shape)

fig, ax=plt.subplots(1,2)
ax[0].plot(detrend_data[32,32,15])
ax[0].set_xlabel('Time(TR)')
ax[0].set_ylabel('Detrend fMRI signal (a.u.)')
ax[1].plot(freqs, detrend_data_power)
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Power')
fig.set_size_inches([12,6])

#-------------------------------------------------------
filter_order=16
n_coefficients=filter_order+1
ub=0.15
lb=0.00001
ub_frac=ub/Nyquist_freq
lb_frac=lb/Nyquist_freq
b_ub=sps.firwin(n_coefficients,ub_frac,window='hamming')
filtered_data=sps.filtfilt(b_ub,[1],data)

filtered_data_power=np.abs(fft.fft(filtered_data))[...,:data.shape[-1]/2+1]
fig,ax=plt.subplots(1,2)
ax[0].plot(data[32,32,15])
ax[0].plot(filtered_data[32,32,15])
ax[0].set_xlabel('Time(TR)')
ax.set_ylabel('Detrend fMRI signal(a.u)')
ax[1].plot(freqs[1:],filtered_data_power[32,32,15][1:])
ax[1].set_xlabel('Frequency(Hz)')
ax[1].set_ylabel('Power')
fig.set_size_inches([12,6])

b_lb=-1*sps.firwin(n_coefficients,lb_frac,window='hamming')
b_lb[(n_coefficients+1)/2]=b_lb[(n_coefficients+1)/2]+1
filtered_data_=sps.filtfilt(-1*b_lb,[1],filtered_data)

filtered_data_power=np.abs(fft.fft(filtered_data))[...,:data.shape[-1]/2+1]
fig,ax=plt.subplots(1,2)
ax[0].plot(data[32,32,15]-np.mean(data[32,32,15]))
ax[0].plot(filtered_data[32,32,15])
ax[0].set_xlabel('Time(TR)')
ax[0].set_ylabel('Detrended fMRI signal(a.u.)')
ax[1].plot(freqs[1:],filtered_data_power[32,32,15][1:])
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_yalbel('Power')
fig.set_size_inches([12,6])

plt.show()
