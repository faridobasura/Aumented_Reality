import os
import json
from properties.resources import Resources
from properties.settings import Settings

#from webapp.apps.payments.models import SalesShift
#from webapp.apps.users.models import User

class Properties():
    def __init__(self):
        self.version = '1.01'
        self.resources = Resources()
        self.settings = Settings()
        #self.settings_file = os.path.join(APP_DATA_DIR,'settings.json')
        self.CAMERA_ID = 0
        self.WINDOW_WIDTH = 768
        self.WINDOW_HEIGHT = 1024
properties = Properties()
