import nibabel as nib
t_map = nib.load('D:/FaceData/mask/rffa.nii')
anat_img = nib.load('D:/FaceData/anat_img/anat.nii')
from nilearn.image import threshold_img


threshold_value_img = threshold_img(t_map,threshold=3.5)


from nilearn import plotting


plotting.plot_stat_map(threshold_value_img,draw_cross=False, title='threshold image with intensity'

                       'value',colorbar=True)


# from nilearn.regions import connected_regions

#regions_value_img,index = connected_regions(threshold_value_img,
                                            # min_region_size=1000)
#print(regions_value_img.shape)


#plotting.plot_roi(regions_value_img,anat_img,title='FFA_ROI',draw_cross=False)



plotting.show()
