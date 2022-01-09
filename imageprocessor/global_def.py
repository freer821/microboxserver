import configparser
import os

from numpy import uint
from log_book import logger

INIDATA_DIRPATH = os.getcwd() + os.path.sep + "InitData"
INIFILE_PARAMETERS = "paramrers.ini"

SECTION_SCANNER = "scanner"
ITEM_SCANNER_RESOLUTION_DPI = "resolution_dpi"

SECTION_COMMON = "common"
ITEM_COMMON_ORIGINAL_IMAGE_DIR = "original_image_dir"
ITEM_COMMON_PROCESSED_IMAGE_DIR = "processed_image_dir"
ITEM_COMMON_BACKUP_DIR = "backup_dir"

SECTION_CANNY = "Canny"
ITEM_CANNY_THRESHOLD1 = "intensity_gradient_upper_bound"
ITEM_CANNY_THRESHOLD2 = "intensity_gradient_lower_bound"

SECTION_BILATERAL = "BilateralFilter"
ITEM_BILATERAL_KSIZE = "kernel_size"
ITEM_BILATERAL_SIGMA_COLOR = "sigma_color"
ITEM_BILATERAL_SIGMA_SPACE = "sigma_space"

SECTION_GAUSSIANBLUR = "GaussianBlur"
ITEM_GAUSSIANBLUR_KSIZE_X = "kernel_x"
ITEM_GAUSSIANBLUR_KSIZE_Y = "kernel_y"
ITEM_GAUSSIANBLUR_SIGMA_X = "standard_deviation_x"
ITEM_GAUSSIANBLUR_SIGMA_Y = "standard_deviation_y"

SECTION_CONTOUR = "Contour"
ITEM_CONTOUR_NUMBER_EDGE_BOUND_UPPER = "number_edge_upper_bound"
ITEM_CONTOUR_PERCENTAGE_CURVE_LENGHT = "percentage_curve_length"
ITEM_CONTOUR_PERCENTAGE_IMAGE_WIDTH = "percentage_image_width"
ITEM_CONTOUR_PERCENTAGE_IMAGE_HEIGHT = "percentage_image_height"

SECTION_GRAYSCALE = "GrayScale"
ITEM_GRAYSCALE_VALUE_BOUND_UPPER = "value_bound_upper"
ITEM_GRAYSCALE_DISTANCE_WIDTH = "distance_width"

SECTION_TIGHTNESS = "Tightness"
ITEM_TIGHTNESS_RADIUS = "radius"

SECTION_VERTEX = "vertex"
ITEM_VERTEX_DISTANCE_HORIZONTAL = "distance_horizontal"
ITEM_VERTEX_DISTANCE_VERTICAL = "distance_vertical"


class Parameters:
    def __init__(self):
        pass

    def get_attributes_number(self):
        return len(self.__dict__.keys())


class ScannerParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.resolution_dpi = 400


class CommonParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.original_image_dirPath = os.getcwd() + os.path.sep + "Images_original"
        self.processed_image_dirPath = os.getcwd() + os.path.sep + "Images_processed"
        self.backup_dirPath = os.getcwd() + os.path.sep + "Backup"


class CannyParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.intensity_gradient_upper_bound = 190
        self.intensity_gradient_lower_bound = 190


class GaussianBlurParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.kernel_x = 5
        self.kernel_y = 5
        self.standard_deviation_x = 1.0
        self.standard_deviation_y = 1.0


class BilateralParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.kernel_size = 9
        self.sigma_color = 75
        self.sigma_space = 75


class ContourParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.number_edge_upper_bound = 2
        self.percentage_curve_length = 0.02
        self.percentage_image_width = 0.02
        self.percentage_image_height = 0.02


class GrayScaleParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.value_bound_upper = 10
        self.distance_width = 5


class TightnessParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.radius = 10


class VertexParameters(Parameters):
    def __init__(self):
        super(Parameters, self).__init__()
        self.distance_horizontal = 10
        self.distance_vertical = 0


class ParameterSettings:
    def __init__(self):
        msgHead = ParameterSettings.__name__ + '.__init__: '
        self.Canny = CannyParameters()
        self.GaussianBlur = GaussianBlurParameters()
        self.Contour = ContourParameters()
        self.GrayScale = GrayScaleParameters()
        self.tightness = TightnessParameters()
        self.vertex = VertexParameters()
        self.common = CommonParameters()
        self.scanner = ScannerParameters()
        self.Bilateral = BilateralParameters()

        if not os.path.isdir(INIDATA_DIRPATH):
            msg = msgHead + 'create dir ' + INIDATA_DIRPATH
            logger.logger.info(msg)
            try:
                os.makedirs(INIDATA_DIRPATH)
            except OSError as err:
                msg = msgHead + 'OS error: {0}'.format(err)
                logger.logger.critical(msg)

        self.filePath = INIDATA_DIRPATH + os.path.sep + INIFILE_PARAMETERS

        self.load()

    def load(self):
        msg_head = ParameterSettings.__name__ + '.load: '
        if os.path.isfile(self.filePath):
            msg = msg_head + 'load parameters from ' + self.filePath
            logger.logger.info(msg)
            config = configparser.ConfigParser()
            config.read(self.filePath)
            for sec in config.sections():
                loaded_attr_num = 0
                if sec == SECTION_COMMON:
                    for item in config[sec]:
                        if item == ITEM_COMMON_ORIGINAL_IMAGE_DIR:
                            try:
                                self.common.image_ouput_dirPath = str(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_COMMON_PROCESSED_IMAGE_DIR:
                            try:
                                self.common.image_ouput_dirPath = str(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_COMMON_BACKUP_DIR:
                            try:
                                self.common.backup_dirPath = str(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue

                    num_atrr = self.common.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_SCANNER:
                    for item in config[sec]:
                        if item == ITEM_SCANNER_RESOLUTION_DPI:
                            try:
                                self.scanner.resolution_dpi = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.scanner.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_CANNY:
                    for item in config[sec]:
                        if item == ITEM_CANNY_THRESHOLD1:
                            try:
                                self.Canny.intensity_gradient_upper_bound = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_CANNY_THRESHOLD2:
                            try:
                                self.Canny.intensity_gradient_lower_bound = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.Canny.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_GAUSSIANBLUR:
                    for item in config[sec]:
                        if item == ITEM_GAUSSIANBLUR_KSIZE_X:
                            try:
                                self.GaussianBlur.kernel_x = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_GAUSSIANBLUR_KSIZE_Y:
                            try:
                                self.GaussianBlur.kernel_y = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_GAUSSIANBLUR_SIGMA_X:
                            try:
                                self.GaussianBlur.standard_deviation_x = float(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_GAUSSIANBLUR_SIGMA_Y:
                            try:
                                self.GaussianBlur.standard_deviation_y = float(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.GaussianBlur.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_BILATERAL:
                    for item in config[sec]:
                        if item == ITEM_BILATERAL_KSIZE:
                            try:
                                self.Bilateral.kernel_size = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue

                        if item == ITEM_BILATERAL_SIGMA_COLOR:
                            try:
                                self.Bilateral.sigma_color = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue

                        if item == ITEM_BILATERAL_SIGMA_SPACE:
                            try:
                                self.Bilateral.sigma_space = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue

                    num_atrr = self.Bilateral.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_CONTOUR:
                    for item in config[sec]:
                        if item == ITEM_CONTOUR_NUMBER_EDGE_BOUND_UPPER:
                            try:
                                self.Contour.number_edge_upper_bound = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_CONTOUR_PERCENTAGE_CURVE_LENGHT:
                            try:
                                self.Contour.percentage_curve_length = float(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_CONTOUR_PERCENTAGE_IMAGE_HEIGHT:
                            try:
                                self.Contour.percentage_image_height = float(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_CONTOUR_PERCENTAGE_IMAGE_WIDTH:
                            try:
                                self.Contour.percentage_image_width = float(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.Contour.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_GRAYSCALE:
                    for item in config[sec]:
                        if item == ITEM_GRAYSCALE_DISTANCE_WIDTH:
                            try:
                                self.GrayScale.distance_width == int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_GRAYSCALE_VALUE_BOUND_UPPER:
                            try:
                                self.GrayScale.value_bound_upper == int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.GrayScale.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_TIGHTNESS:
                    for item in config[sec]:
                        if item == ITEM_TIGHTNESS_RADIUS:
                            try:
                                self.tightness.radius = float(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.tightness.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

                if sec == SECTION_VERTEX:
                    for item in config[sec]:
                        if item == ITEM_VERTEX_DISTANCE_HORIZONTAL:
                            try:
                                self.vertex.distance_horizontal = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                        if item == ITEM_VERTEX_DISTANCE_VERTICAL:
                            try:
                                self.vertex.distance_vertical = int(config[sec][item])
                                loaded_attr_num = loaded_attr_num + 1
                            except ValueError:
                                msg = msg_head + 'ValueError, [' + sec + ']-' + item + ' uses default value'
                                logger.logger.error(msg)
                            continue
                    num_atrr = self.vertex.get_attributes_number()
                    if num_atrr > loaded_attr_num:
                        msg = msg_head + "not all parameters found in [" + sec + "], parameters use default values that not found"
                        logger.logger.warning(msg)
                    continue

        else:
            msg = msg_head + self.filePath + ' not found, using default parameters'
            logger.logger.info(msg)

            # save parameters: this step is nessary for the case that new parameters defined but they are not found in ini-file
        self.save()

    def save(self):
        msg = ParameterSettings.__name__ + '.save: save parameters in ' + self.filePath
        logger.logger.info(msg)

        if os.path.isfile(self.filePath):
            os.remove(self.filePath)

        config = configparser.RawConfigParser()

        config.add_section(SECTION_COMMON)
        config.set(SECTION_COMMON, ITEM_COMMON_ORIGINAL_IMAGE_DIR, str(self.common.original_image_dirPath))
        config.set(SECTION_COMMON, ITEM_COMMON_PROCESSED_IMAGE_DIR, str(self.common.processed_image_dirPath))
        config.set(SECTION_COMMON, ITEM_COMMON_BACKUP_DIR, str(self.common.backup_dirPath))

        config.add_section(SECTION_SCANNER)
        config.set(SECTION_SCANNER, ITEM_SCANNER_RESOLUTION_DPI, str(self.scanner.resolution_dpi))

        config.add_section(SECTION_CANNY)
        config.set(SECTION_CANNY, ITEM_CANNY_THRESHOLD1, str(self.Canny.intensity_gradient_upper_bound))
        config.set(SECTION_CANNY, ITEM_CANNY_THRESHOLD2, str(self.Canny.intensity_gradient_lower_bound))

        config.add_section(SECTION_GAUSSIANBLUR)
        config.set(SECTION_GAUSSIANBLUR, ITEM_GAUSSIANBLUR_KSIZE_X, str(self.GaussianBlur.kernel_x))
        config.set(SECTION_GAUSSIANBLUR, ITEM_GAUSSIANBLUR_KSIZE_Y, str(self.GaussianBlur.kernel_y))
        config.set(SECTION_GAUSSIANBLUR, ITEM_GAUSSIANBLUR_SIGMA_X, str(self.GaussianBlur.standard_deviation_x))
        config.set(SECTION_GAUSSIANBLUR, ITEM_GAUSSIANBLUR_SIGMA_Y, str(self.GaussianBlur.standard_deviation_y))

        config.add_section(SECTION_BILATERAL)
        config.set(SECTION_BILATERAL, ITEM_BILATERAL_KSIZE, str(self.Bilateral.kernel_size))
        config.set(SECTION_BILATERAL, ITEM_BILATERAL_SIGMA_COLOR, str(self.Bilateral.sigma_color))
        config.set(SECTION_BILATERAL, ITEM_BILATERAL_SIGMA_SPACE, str(self.Bilateral.sigma_space))

        config.add_section(SECTION_CONTOUR)
        config.set(SECTION_CONTOUR, ITEM_CONTOUR_NUMBER_EDGE_BOUND_UPPER, str(self.Contour.number_edge_upper_bound))
        config.set(SECTION_CONTOUR, ITEM_CONTOUR_PERCENTAGE_CURVE_LENGHT, str(self.Contour.percentage_curve_length))
        config.set(SECTION_CONTOUR, ITEM_CONTOUR_PERCENTAGE_IMAGE_WIDTH, str(self.Contour.percentage_image_width))
        config.set(SECTION_CONTOUR, ITEM_CONTOUR_PERCENTAGE_IMAGE_HEIGHT, str(self.Contour.percentage_image_height))

        config.add_section(SECTION_GRAYSCALE)
        config.set(SECTION_GRAYSCALE, ITEM_GRAYSCALE_VALUE_BOUND_UPPER, str(self.GrayScale.value_bound_upper))
        config.set(SECTION_GRAYSCALE, ITEM_GRAYSCALE_DISTANCE_WIDTH, str(self.GrayScale.distance_width))

        config.add_section(SECTION_TIGHTNESS)
        config.set(SECTION_TIGHTNESS, ITEM_TIGHTNESS_RADIUS, str(self.tightness.radius))

        config.add_section(SECTION_VERTEX)
        config.set(SECTION_VERTEX, ITEM_VERTEX_DISTANCE_HORIZONTAL, str(self.vertex.distance_horizontal))
        config.set(SECTION_VERTEX, ITEM_VERTEX_DISTANCE_VERTICAL, str(self.vertex.distance_vertical))

        try:
            with open(self.filePath, 'w') as configfile:
                config.write(configfile)
            configfile.close()
        except OSError as err:
            msg = ParameterSettings.__name__ + '.save: OS error: {0}'.format(err)
            logger.logger.critical(msg)


# global variable in scope of the package
parameters = ParameterSettings()
