import os
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

from properties.config import BASE_DIR

class Resources():
    def __init__(self):
        # load pictures

        models_dir = os.path.join(BASE_DIR, 'models')
        shaders_dir = os.path.join(BASE_DIR, 'shaders')
        textures_dir = os.path.join(BASE_DIR, 'models', 'textures')

        img_path = os.path.join(BASE_DIR, 'assets', 'img')
        catalog_path = os.path.join(img_path, 'catalog')
        
        self.textures_dir = textures_dir
        self.shaders_dir = shaders_dir

        self.silhouette = os.path.join(img_path, 'silhouette.png')
        self.shirt_pcl = os.path.join(catalog_path, 'shirt_pcl.png')       
        self.shirt_white = os.path.join(catalog_path, 'shirt_white.png')
        self.shirt_spidey = os.path.join(catalog_path, 'shirt_spidey.png')

        self.twoD_shirt_path = os.path.join(
            models_dir,
            'Black_T_Shirt_PNG_Clip_Art-3107.png'
        )
        

        self.shirt_obj_paths = {
            "new_shirt" : os.path.join(
                models_dir,
                'obj',
                'new_shirt.obj'
            ),
            "new_shirt_frontal" : os.path.join(
                models_dir,
                'obj',
                'shirt_frontal.obj'
            ),
            "playeraM_v1" : os.path.join(
                models_dir,
                'obj',
                'playeraM_v1.0.obj'
            ),
        }
        
        self.shirt_obj_path = self.shirt_obj_paths["playeraM_v1"]
        

