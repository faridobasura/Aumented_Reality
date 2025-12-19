import os
import json
from properties.resources import Resources
#from webapp.apps.payments.models import SalesShift
#from webapp.apps.users.models import User

class Properties():
    def __init__(self):
        self.version = '1.01'
        self.resources = Resources()
        #self.settings_file = os.path.join(APP_DATA_DIR,'settings.json')

properties = Properties()
