from nilearn.image import load_img
func_img=load_img('D:/FaceData/func_img/wrarun1.nii')

T_map=load_img('D:/FaceData/mask/spmT_0001.nii')

# Got the cortical mesh
from nilearn import datasets
fsaverage=datasets.fetch_surf_fsaverage5()

#Sample the 3D data around each node of the mesh

from nilearn import plotting

plotting.plot_glass_brain(T_map,display_mode='r',plot_abs='False',
                          title='glass brain',threshold=2.)
plotting.plot_stat_map(T_map,display_mode='x',
                       threshold=1.,cut_coords=range(0,51,10),title='slices')
plotting.show()
