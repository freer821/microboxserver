from global_def import parameters
from log_book import logger
from zipfile import ZipFile
from helper import *
import shutil

FILE_PREFIX = "backup"
FILE_EXT = ".zip"


class BackUp:
    def __init__(self):
        self.original_imagePath = ""
        self.logFilePath = logger.filePath
        self.configFilePath = parameters.filePath

    def set_filePath_originalImage(self, filePath):
        self.original_imagePath = filePath

    def backup(self):
        msgHeader = BackUp.__name__ + '.backup: '

        temp_dirPath = parameters.common.backup_dirPath + os.path.sep + FILE_PREFIX + "_" + dateTime2str()
        os.makedirs(temp_dirPath)

        file_name = temp_dirPath + FILE_EXT
        msg = msgHeader + "backup found in " + file_name
        logger.logger.info(msg)

        source_name = self.original_imagePath
        if os.path.isfile(source_name):
            target_name = os.path.basename(source_name)
            target_name = temp_dirPath + os.path.sep + target_name
            shutil.copyfile(source_name, target_name)
        else:
            msg = msgHeader + "file not found: " + source_name
            logger.logger.warning(msg)

        source_name = self.configFilePath
        if os.path.isfile(source_name):
            target_name = os.path.basename(source_name)
            target_name = temp_dirPath + os.path.sep + target_name
            shutil.copyfile(source_name, target_name)
        else:
            msg = msgHeader + "file not found: " + source_name
            logger.logger.warning(msg)

        source_name = self.logFilePath
        if os.path.isfile(source_name):
            target_name = os.path.basename(source_name)
            target_name = temp_dirPath + os.path.sep + target_name
            shutil.copyfile(source_name, target_name)
        else:
            msg = msgHeader + "file not found: " + source_name
            logger.logger.warning(msg)

        file_paths = get_all_file_paths(temp_dirPath)

        zip_fileName = temp_dirPath + FILE_EXT
        with ZipFile(zip_fileName, 'w') as zip:
            for file in file_paths:
                zip.write(file)

        shutil.rmtree(temp_dirPath)


# global variable 
backuper = BackUp()
