import os
import logging

LOG_DIRPATH  = os.getcwd()+os.path.sep + "LogData"
LOGFILE_PARAMETERS = "logger.txt"

class Logger:
    def __init__(self): 
        if not os.path.isdir(LOG_DIRPATH):
            os.makedirs(LOG_DIRPATH)
        self.filePath = LOG_DIRPATH+os.path.sep+LOGFILE_PARAMETERS

        # config logging 
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(self.filePath, mode='a')
        fileHandler.setLevel(logging.INFO)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)

        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)


# global variable in scope of the package 
logger = Logger()


       