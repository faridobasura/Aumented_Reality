# 3rd party imports
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Settings(BaseModel):
    Fullscreen:bool = True

    