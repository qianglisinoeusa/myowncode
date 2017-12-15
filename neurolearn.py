from pynv import Client

api=Client(access_token='uctb6zhxK6BUY4JgPZZzGZK22tk2zxbX8U5QLNSl')

collection=api.create_collection('Gender')

image_file_path='D:/FaceData/func_img/wrarun1.nii'

image=api.add_image(
    collection['9850'],
    image_file_path,
    name='Gender',
    modality='fMRI_BOLD',
    map_type='Other'
)

