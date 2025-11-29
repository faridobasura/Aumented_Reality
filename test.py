# test_opencv_texture.py
from utils.objectLoader import ObjectLoader
import os

obj_path = os.path.expanduser('~/AR_python/Aumented_Reality/t_shirt_model/t_shirt.obj')

print("🧪 TEST OPENCV TEXTURE")

loader = ObjectLoader()
if loader.load_obj(obj_path):
    print(f"✅ Textura cargada: {loader.texture_loaded}")
    if loader.texture_cv is not None:
        print(f"✅ Textura OpenCV cargada: {loader.texture_cv.shape}")
    else:
        print("❌ Textura OpenCV no cargada")