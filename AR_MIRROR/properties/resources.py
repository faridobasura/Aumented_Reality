import os
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

from properties.config import BASE_DIR

class Resources():
    def __init__(self):
        # load pictures
        img_path = os.path.join(BASE_DIR, 'assets', 'img')
        catalog_path = os.path.join(img_path, 'catalog')

        self.shirt_white = os.path.join(catalog_path, 'shirt_white.png')
        self.shirt_spidey = os.path.join(catalog_path, 'shirt_spidey.png')
        self.shirt_pcl = os.path.join(catalog_path, 'shirt_pcl.png')       

