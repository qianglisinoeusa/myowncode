import nibabel as nib
t_map = nib.load('D:/FaceData/mask/spmT_0001.nii')
anat_img = nib.load('D:/FaceData/anat_img/anat.nii')
from nilearn.image import threshold_img

# threshold_percentile_img = threshold_img(t_map, threshold='97%')
threshold_value_img = threshold_img(t_map,threshold=4.5)

# threshold_value_img.to_filename("D:/FaceData/threshold_value_img.nii.gz")
from nilearn import plotting
# plotting.plot_stat_map(threshold_percentile_img,display_mode='x',
#                        cut_coords=5, title='threshold image with string'
#                       'percentile',colorbar=False)
plotting.plot_stat_map(threshold_value_img,draw_cross=False,
                       cut_coords=[30,-72,-6],title='threshold image with intensity'
                                          'value',colorbar=False)
from nilearn.regions import connected_regions
# regions_percentile_img,index = connected_regions(threshold_percentile_img,
#                                                 min_region_size=1500)
regions_value_img,index = connected_regions(threshold_value_img,
                                            min_region_size=1400)
print(regions_value_img.shape)
# regions_value_img.to_filename("D:/FaceData/roi.nii.gz")
# title = ("ROIs using percentile threshold."
#        "\n Each ROI in same color is an extracted region")
# plotting.plot_prob_atlas(regions_percentile_img,anat_img=t_map,
#                         view_type='contours',display_mode='x',
#                         cut_coords=5)
# title = ("ROIs using image intensity threshold."
#        "\n Each ROI in same color is an extracted region")
plotting.plot_prob_atlas(regions_value_img,bg_img=t_map,
                         view_type='contours',
                         cut_coords=[30,-72,-6])
plotting.plot_roi(regions_value_img,anat_img,title='RFFA_ROI',cut_coords=[30,-72,-6],
                  draw_cross=False)
plotting.show()
