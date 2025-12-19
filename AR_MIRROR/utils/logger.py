import os
import logging
import logging.handlers
from pytz import timezone
from datetime import datetime

from properties.config import APP_DATA_DIR

class CustomFormatter(logging.Formatter):
    white = "\033[97m"
    grey = "\033[90m"
    yellow = "\033[33m"
    red = "\033[91m"
    bold_red = "\033[1;31m"
    reset = "\033[0m"
    format = "%(asctime)s (%(name)s)[%(levelname)s]: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

########     Prepare logging    ########
tz = timezone('America/Mexico_City') # UTC, Asia/Shanghai, Europe/Berlin
def timetz(*args):
    return datetime.now(tz).timetuple()

# Create and configure logger
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG, INFO, WARNING
logger.setLevel(logging.DEBUG)


handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

# dump log to file as default
# create app data dir if not exists
if not os.path.exists(APP_DATA_DIR):
    os.makedirs(APP_DATA_DIR)

# limit log file size
handler = logging.handlers.RotatingFileHandler(os.path.join(APP_DATA_DIR,'ar_mirror.log'), maxBytes=1024*1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s (%(name)s)[%(levelname)s]: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def set_log_level(level):
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)