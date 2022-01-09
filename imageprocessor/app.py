import cv2
import numpy
import numpy as np
import math

from global_def import parameters
from log_book import logger
from helper import *
import os

from PIL import Image as PILImage, UnidentifiedImageError

DEFAULT_IMAGE_FORMAT = 'tiff'
DEFAULT_IMAGE_FILEEXTENSION = '.' + DEFAULT_IMAGE_FORMAT
DEFAULT_PROCESSED_IMAGE_FILENAME = 'processed_image'
DEFAULT_SOURCE_IMAGE_FILENAME = 'source_image'

if __debug__:
    import time

OPENCV_IMAGE_TYPE = ['.MBP', 'DIB', '.JPEG', '.JPG', '.JPE', '.PNG', '.PBM', 'PGM',
                     '.PPM', '.SR', '.RAS', '.TIFF', '.TIF']


class ContourPonts2D:
    def __init__(self, point_list):
        # point has the form [width,height]
        self.points = np.array(point_list)
        if len(self.points) > 1:
            self.points = np.squeeze(np.array(point_list))
        #

    def find(self, point):

        if point is None:
            msg = ContourPonts2D.__name__ + '.find: no point found'
            return -1
        ind = 0
        for entry in self.points:
            if entry[0] == point[0] and entry[1] == point[1]:
                return ind
            else:
                ind = ind + 1
        msg = ContourPonts2D.__name__ + '.find: no point found'
        self.logger.info(msg)
        return -1

    def delete(self, ind):
        if ind >= 0 or ind < len(self.points):
            self.points = np.delete(self.points, ind, axis=0)
        else:
            msg = ContourPonts2D.__name__ + '.delete: no point to be deleted'
            self.logger.info(msg)

    def min_x_values(self):
        arr = np.array(self.get_x_values())
        return np.min(arr)

    def max_x_values(self):
        arr = np.array(self.get_x_values())
        return np.max(arr)

    def min_y_values(self):
        arr = np.array(self.get_y_values())
        return np.min(arr)

    def max_y_values(self):
        arr = np.array(self.get_y_values())
        return np.max(arr)

    def mean_x_value(self):
        arr = self.get_x_values()
        return np.mean(arr)

    def mean_y_value(self):
        arr = self.get_y_values()
        return np.mean(arr)

    def get_x_values(self):
        res = []
        for point in self.points:
            res.append(point[0])
        return (res)

    def get_y_values(self):
        res = []
        for point in self.points:
            res.append(point[1])
        return (res)

    def rect_boundary(self):
        # find area that is a rect:
        # - top boundary: min. y of all points
        # - bottom boundary: max. y of all points
        # - left boundary: min. x of all points
        # - right boundary: max. x of all points
        #
        # top left corner: (min_x, max_width)
        # top right corner: (min_height, max_width)
        # bottom left corner: (max_height, min_width)
        # bottom right corner: (max_height, max_width)
        x = np.array(self.get_x_values())
        y = np.array(self.get_y_values())
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        corner_lt = [x_min, y_max]
        corner_rt = [x_max, y_max]
        corner_rb = [x_max, y_min]
        corner_lb = [x_min, y_min]
        return [corner_lt, corner_rt, corner_rb, corner_lb]

    def slice_x(self, leftBound, rightBound):
        keptPoints = []
        removedPoints = []
        for point in self.points:
            x = point[0]
            if x > leftBound and x < rightBound:
                keptPoints.append(point)
            else:
                removedPoints.append(point)

        return keptPoints, removedPoints


class Image:
    def __init__(self, imageArray=None):
        msgHeader = Image.__name__ + '.__init__: '
        self.image = None
        self.height = 0
        self.width = 0
        self.channels = 0
        self.resolution_dpi = parameters.scanner.resolution_dpi
        self.format = DEFAULT_IMAGE_FORMAT

        if imageArray is not None:
            imageArray = np.array(imageArray)

            size = imageArray.shape
            dims = len(size)
            if dims == 1:
                msg = msgHeader + "1D-size expaned to [1, width]"
                logger.logger.warning(msg)
                imageArray = np.array([imageArray])
                size = imageArray.shape
                dims = len(size)

            if dims > 3:
                msg = msgHeader + 'image size must be [height, width] or [height, width, channels]'
                logger.logger.error(msg)
            else:
                self.image = imageArray
                if dims == 2:
                    self.height, self.width = size
                    self.channels = 1
                else:
                    self.height, self.width, self.channels = size

    @classmethod
    def load(self, buff, width, height, channels):
        flatNumpyArray = numpy.array(buff)
        imgarray = flatNumpyArray.reshape(width, height, channels)
        return Image(imgarray)


    @classmethod
    def load(self, filepath):
        msgHeader = Image.__name__ + ".load: "
        if os.path.isfile(filepath):
            msg = msgHeader + "load iamge from " + filepath
            logger.logger.info(msg)

            # load image from file 
            try:
                im = PILImage.open(filepath)
            except UnidentifiedImageError as err:
                msg = msgHeader + str(err.args[0])
                logger.logger.error(msg)
                return Image()

            if im is not None:
                imag = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                if imag is None:
                    msg = msgHeader + "loading image from file failed"
                    logger.logger.error(msg)

                res = Image(imag)
                if im.format == '':
                    msg = msgHeader + "image format not found, use default value"
                    logger.logger.warning(msg)
                else:
                    res.format = im.format

                msg = msgHeader + "color resolution using the default value"
                logger.logger.info(msg)
                return res
        else:
            msg = msgHeader + "file not found: " + filepath
            logger.logger.error(msg)
            return Image()

    def clone(self):
        res = Image(self.image)
        res.resolution_dpi = self.resolution_dpi
        res.format = self.format
        return res

    def super_resolution_FSRCNN_2x(self):
        # reference:
        # 1. "Image Super-Resolution Using Deep Convolutional Networks", Chao Dong
        # 2. "Accelerating the Super-Resolution Convolutional Neural Network", Chao Dong
        # 3. https://learnopencv.com/super-resolution-in-opencv/#sec5
        # 4. https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
        msgHeader = Image.__name__ + ".super_resolution_FSRCNN_2x: "
        pbFilePath = os.getcwd() + os.path.sep + "src" + os.path.sep + "FSRCNN_x2.pb"
        if not os.path.isfile(pbFilePath):
            msg = msgHeader + "model file not found, return unscaled image: " + pbFilePath
            logger.logger.error(msg)
            if __debug__:
                print(msg)

            return self.clone()

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(pbFilePath)
        sr.setModel("fsrcnn", 2)
        img = sr.upsample(self.image)

        result = Image(img)
        result.resolution_dpi = self.resolution_dpi * 2
        return result

    def super_resolution_FSRCNN_3x(self):
        # reference:
        # 1. "Image Super-Resolution Using Deep Convolutional Networks", Chao Dong
        # 2. "Accelerating the Super-Resolution Convolutional Neural Network", Chao Dong
        # 3. https://learnopencv.com/super-resolution-in-opencv/#sec5
        # 4. https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
        msgHeader = Image.__name__ + ".super_resolution_FSRCNN_3x: "
        pbFilePath = os.getcwd() + os.path.sep + "src" + os.path.sep + "FSRCNN_x3.pb"
        if not os.path.isfile(pbFilePath):
            msg = msgHeader + "model file not found, return unscaled image: " + pbFilePath
            logger.logger.error(msg)
            if __debug__:
                print(msg)

            return self.clone()

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(pbFilePath)
        sr.setModel("fsrcnn", 3)
        img = sr.upsample(self.image)

        result = Image(img)
        result.resolution_dpi = self.resolution_dpi * 3
        return result

    def super_resolution_FSRCNN_4x(self):
        # reference:
        # 1. "Image Super-Resolution Using Deep Convolutional Networks", Chao Dong
        # 2. "Accelerating the Super-Resolution Convolutional Neural Network", Chao Dong
        # 3. https://learnopencv.com/super-resolution-in-opencv/#sec5
        # 4. https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models
        msgHeader = Image.__name__ + ".super_resolution_FSRCNN_4x: "
        pbFilePath = os.getcwd() + os.path.sep + "src" + os.path.sep + "FSRCNN_x4.pb"
        if not os.path.isfile(pbFilePath):
            msg = msgHeader + "model file not found, return unscaled image: " + pbFilePath
            logger.logger.error(msg)
            if __debug__:
                print(msg)

            return self.clone()

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(pbFilePath)
        sr.setModel("fsrcnn", 4)
        img = sr.upsample(self.image)

        result = Image(img)
        result.resolution_dpi = self.resolution_dpi * 4
        return result

    def schrink(self, scale_vaule):
        msgHeader = Image.__name__ + ".schrink: "
        res = self.clone()
        if scale_vaule < 0 or scale_vaule > 1:
            msg = msgHeader + " scale must be in [0,1]: " + str(scale_vaule)
            logger.logger.error(msg)
        else:
            img = PILImage.fromarray(res.image)
            width = math.ceil(self.width * scale_vaule)
            height = math.ceil(self.height * scale_vaule)
            img.thumbnail((width, height), PILImage.ANTIALIAS)
            res = Image(imageArray=np.array(img))
            # update resolution
            res.resolution_dpi = int(math.ceil(self.resolution_dpi * scale_vaule))
        return res

    def change_resolution(self, dpi_value):
        msgHeader = Image.__name__ + ".change_resolution: "
        res = self.clone()
        if dpi_value < 1:
            msg = msgHeader + "dpi value must be a positive integer: " + str(dpi_value)
            logger.logger.error(msg)
            return res

        scale_value = dpi_value / self.resolution_dpi
        expected_width = math.ceil(self.width * scale_value)
        expected_height = math.ceil(self.height * scale_value)

        if scale_value <= 1:
            msg = msgHeader + "image size will be schrinked from (" + str(self.width) + "," + str(
                self.height) + ") to (" + str(expected_width) + "," + str(expected_height) + ")"
            logger.logger.info(msg)
            res = self.schrink(scale_vaule=scale_value)  #
        else:
            msg = msgHeader + "image size will be enlarged from (" + str(self.width) + "," + str(
                self.height) + ")to (" + str(expected_width) + "," + str(expected_height) + ")"
            logger.logger.info(msg)

            steps = math.ceil(math.log2(scale_value))

            for itr in range(steps):
                res = res.super_resolution_FSRCNN_2x()
            res = res.schrink(dpi_value / res.resolution_dpi)

        return res

    def __eq__(self, other):
        # compare size of image       
        res = self.height == other.height
        res = res and (self.width == other.width)
        res = res and (self.channels == other.channels)
        res = res and (self.resolution_dpi == other.resolution_dpi)

        # compare values of image 
        if res:
            if self.image is None and other.image is None:
                res = res and True
            else:
                res_array = np.equal(self.image, other.image)
                num = self.height * self.width * self.channels
                res_array = np.unique(np.reshape(res_array, num))
                if len(res_array) == 1:  # comparision result: either all elements in image are same or all elements are differnet
                    res = bool(res_array[0])
                else:  # comparision result: not all elemments are same
                    res = False
        return res

    def save(self, filePath):
        msgHeader = Image.__name__ + '.save: '
        if self.image is None:
            msg = msgHeader + " image array is empty, nothing to save"
            logger.logger.warning(msg)
        else:
            # validate file extension
            dirname = os.path.dirname(filePath)
            name = os.path.splitext(os.path.basename(filePath))
            file_name = name[0]
            file_ext = name[-1]
            if file_ext.upper() not in OPENCV_IMAGE_TYPE:
                msg = Image.__name__ + '.save: image type *' + file_ext + ' not support'
                logger.logger.warning(msg)

                file_ext = DEFAULT_IMAGE_FILEEXTENSION
                filePath = dirname + os.path.sep + file_name + file_ext

            msg = Image.__name__ + '.save: save image in ' + filePath
            logger.logger.info(msg)
            cv2.imwrite(filePath, self.image)

            imag = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            im = PILImage.fromarray(imag)
            im.save(filePath, dpi=(self.resolution_dpi, self.resolution_dpi))

    def show(self):
        msgHeader = Image.__name__ + '.show: '
        if self.image is None:
            msg = msgHeader + " image array is empty, nothing to show"
            logger.logger.warning(msg)
        else:
            cv2.imshow('Image', self.image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_edges_Canny(self):
        msgHeader = Image.__name__ + '.detect_edges_Canny: '
        img = None

        if self.image is None:
            msg = msgHeader + " image array is empty, cannot detect edges"
            logger.logger.warning(msg)
        else:
            threshold1 = parameters.Canny.intensity_gradient_upper_bound
            threshold2 = parameters.Canny.intensity_gradient_lower_bound
            img = cv2.Canny(self.image, threshold1, threshold2)
        return Image(img)

    def denoise_Gaussian(self):
        msgHeader = Image.__name__ + '.denoise_Gaussian: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot denosie image'
            logger.logger.warning(msg)
        else:
            kernel_x = parameters.GaussianBlur.kernel_x
            kernel_y = parameters.GaussianBlur.kernel_y
            sigma_x = parameters.GaussianBlur.standard_deviation_x
            sigma_y = parameters.GaussianBlur.standard_deviation_y

            # validate parameter: kernel_x must be positv and odd
            if kernel_x <= 0:
                msg = msgHeader + 'kernel_x must be positive'
                logger.logger.error(msg)
                return Image(image=self.image)
            elif kernel_x % 2 == 0:
                kernel_x = kernel_x + 1
                msg = msgHeader + 'kernel_x must be odd'
                logger.logger.warning(msg)

            # validate parameter: kernel_y must be positv and odd
            if kernel_y <= 0:
                msg = msgHeader + 'kernel_y must be positive'
                logger.logger.error(msg)
                return Image(image=self.image)
            elif kernel_y % 2 == 0:
                kernel_y = kernel_y + 1
                msg = msgHeader + 'kernel_y must be odd'
                logger.logger.warning(msg)

            # validate parameter: sigma_y must be nonnegative
            if sigma_y < 0:
                msg = msgHeader + 'sigma_y must be nonnegative'
                logger.logger.error(msg)
                return Image(image=self.image)

            # validate parameter: sigma_x must be nonnegative
            if sigma_x < 0:
                msg = msgHeader + 'sigma_x must be nonnegative'
                logger.logger.error(msg)
                return Image(image=self.image)

            img = cv2.GaussianBlur(self.image, (kernel_x, kernel_y), sigmaX=sigma_x, sigmaY=sigma_y)

        return Image(imageArray=img)

    def denoise_Birateral(self):
        msgHeader = Image.__name__ + '.denoise_Birateral: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot denosie image'
            logger.logger.warning(msg)
        else:
            ksize = parameters.Bilateral.kernel_size
            sigma_color = parameters.Bilateral.sigma_color
            sigma_space = parameters.Bilateral.sigma_space

            # validate parameter: ksize must be positv and odd
            if ksize <= 0:
                msg = msgHeader + 'kernel_x must be positive'
                logger.logger.error(msg)
                return Image(image=self.image)
            elif ksize % 2 == 0:
                ksize = ksize + 1
                msg = msgHeader + 'kernel_x must be odd'
                logger.logger.warning(msg)

            # validate parameter: sigma_color must be nonnegative
            if sigma_color < 0:
                msg = msgHeader + 'sigma_color must be nonnegative'
                logger.logger.error(msg)
                return Image(image=self.image)

            # validate parameter: sigma_x must be nonnegative
            if sigma_space < 0:
                msg = msgHeader + 'sigma_space must be nonnegative'
                logger.logger.error(msg)
                return Image(image=self.image)

            img = cv2.bilateralFilter(src=self.image, d=ksize, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return Image(imageArray=img)

    def convert2GRAYSCALE(self):
        msgHeader = Image.__name__ + '.convert2GRAYSCALE: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot convert image into gray scale'
            logger.logger.warning(msg)
        else:
            if self.channels == 3:
                img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                img = self.image
                msg = msgHeader + "Converting in grayscale failed"
                logger.logger.error(msg)

        return Image(imageArray=img)

    def validate_segment_area(self, area):
        # area: a list of corners [top left, top right, bottom right, bottom left]
        msgHeader = Image.__name__ + ".validate_segment_area: "
        # init 
        points = [[0, 0], [self.width, self.height]]
        result = ContourPonts2D(point_list=points)
        result = result.rect_boundary()

        num_corners = len(area)
        if num_corners <= 1:
            msg = msgHeader + " couldnot reconstruct area with less than one point, return corners of the original image "
            logger.logger.warning(msg)

        else:
            keptPoints = ContourPonts2D(point_list=area)
            x_min = keptPoints.min_x_values()
            x_max = keptPoints.max_x_values()
            y_min = keptPoints.min_y_values()
            y_max = keptPoints.max_y_values()

            x_min = max(0, x_min)
            x_max = min(x_max, self.width)
            y_min = max(y_min, 0)
            y_max = min(y_max, self.height)

            if x_max == x_min or y_max == y_min:  # area is a horizontal or vertcial line
                msg = msgHeader + ' area is a vertical/horizontal line, return corners of the original image'
                logger.logger.warning(msg)
            else:
                points = [[x_min, y_min], [x_max, y_max]]
                result = ContourPonts2D(point_list=points)
                result = result.rect_boundary()

        return result

    def segmentation(self, area):
        # area: a list of corners [top left, top right, bottom right, bottom left]
        msgHeader = Image.__name__ + '.segmentation: '
        resultImg = None
        num_corners = len(area)
        if num_corners <= 1:
            msg = msgHeader + " couldnot reconstruct area with less than one point, return original image "
            logger.logger.warning(msg)

        else:
            keptPoints = ContourPonts2D(point_list=area)
            x_min = keptPoints.min_x_values()
            x_max = keptPoints.max_x_values()
            y_min = keptPoints.min_y_values()
            y_max = keptPoints.max_y_values()

            if x_min < 0 or x_max > self.width or \
                    y_min < 0 or y_max > self.height:
                msg = msgHeader + ' area is out of image area, return original image'
                logger.logger.warning(msg)
                resultImg = self.image.copy()
            else:
                if x_max == x_min or y_max == y_min:  # crop area is a horizontal or vertcial line
                    msg = msgHeader + ' area is a vertical/horizontal line, return original image'
                    logger.logger.info(msg)
                    resultImg = self.image.copy()
                else:
                    img = self.image[y_min:y_max + 1, x_min:x_max + 1, :]

                    # # show 
                    # image = self.sourceImage.image.copy()
                    # for point in corners_crop_area:
                    #     image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
                    # result = Image(image)
                    # result.show()

        return Image(imageArray=resultImg)


class SourceImage(Image):
    def __init__(self, filePath=None, imageArray=None):
        msgHeader = SourceImage.__name__ + ".__init__: "
        self.filePath = ''
        if filePath is None and imageArray is not None:
            msg = msgHeader + "load image from array"
            logger.logger.info(msg)
            super().__init__(imageArray=imageArray)

        if filePath is not None and imageArray is None:
            msg = msgHeader + "load image from file: " + filePath
            logger.logger.info(msg)
            self.load(filePath)

        if filePath is None and imageArray is None:
            super().__init__(imageArray=imageArray)

        if filePath is not None and imageArray is not None:
            msg = msgHeader + "multiple sources provided,  prefer to load image from file: " + filePath
            logger.logger.warning(msg)
            self.load(filePath)

    def load(self, filePath):
        img = Image.load(filePath)
        self.filePath = filePath
        self.format = img.format
        self.width = img.width
        self.height = img.height
        self.channels = img.channels
        self.resolution_dpi = img.resolution_dpi
        self.image = img.image

    def set_default_filePath(self):
        file_dir = parameters.common.original_image_dirPath
        file_name = DEFAULT_SOURCE_IMAGE_FILENAME + '_' + dateTime2str() + DEFAULT_IMAGE_FILEEXTENSION
        file_path = file_dir + os.path.sep + file_name
        self.filePath = file_path


class ProcessedImage(Image):
    def __init__(self, imageArray=None):
        self.filePath = ""
        super().__init__(imageArray=imageArray)

    def save(self, filePath):
        self.filePath = filePath
        super().save(filePath=filePath)

    def set_filePath(self, source_filePath=None):
        filePath = parameters.common.processed_image_dirPath + os.path.sep + DEFAULT_PROCESSED_IMAGE_FILENAME + '_' + dateTime2str()
        fileExt = DEFAULT_IMAGE_FILEEXTENSION

        if source_filePath is not None:
            name = os.path.splitext(os.path.basename(self.sourceImageFile))
            file_name = name[0]
            # file_ext = name[-1] 
            if file_name == '':
                file_name = DEFAULT_PROCESSED_IMAGE_FILENAME
            else:
                file_name = "processed_" + file_name
            filePath = parameters.common.processed_image_dirPath + os.path.sep + file_name

        self.filePath = filePath + fileExt


class ImageProcessingReuslts:
    def __init__(self):
        self.filePath_sourceImage = ""
        self.filePath_destIamge = " "
        self.crop_area = []  # crop area is given by its corners in consequence of left top, right top, right bottom and left bottom


class ImageProcess:
    def __init__(self, sourceImageFile=None, sourceImageArray=None):
        msgHeader = ImageProcess.__name__ + ".__init__: "
        self.sourceImage = SourceImage(filePath=sourceImageFile, imageArray=sourceImageArray)
        self.processedImage = ProcessedImage()

    def reset(self):
        self.sourceImage = None
        self.processedImage = None

    def detect_contours(self):
        contourSorted = []
        # convert into gray 
        grayImage = self.sourceImage.convert2GRAYSCALE()
        # remove noise 
        denoisedImage = grayImage.denoise_Gaussian()
        #        denoisedImage.show()
        # detect edges using canny 
        edgeImage = denoisedImage.detect_edges_Canny()
        #        edgeImage.show()
        # find contours  
        contours, hierarchy = cv2.findContours(edgeImage.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # sort contoures descently with respect to their lengths 
            contourLength_arr = []
            for entry in contours:
                contourLength_arr.append(cv2.arcLength(entry, True))
            contourLength_arr = np.array(contourLength_arr)
            ind_arr = np.flip(np.argsort(contourLength_arr))
            for ind in ind_arr:
                contourSorted.append(contours[ind])

        return contourSorted

    def crop_background_automatic(self):
        msgHeader = ImageProcess.__name__ + '.crop_background_automatic: '
        msg = msgHeader + 'atuomatic crop background'
        logger.logger.info(msg)

        self.processedImage = None
        # intialize corners of crop area  
        edgePoints = [[0, 0], [self.sourceImage.width, self.sourceImage.height]]
        keptPoints = ContourPonts2D(point_list=edgePoints)
        corners_crop_area = keptPoints.rect_boundary()
        # # show 
        # image = self.sourceImage.image.copy()
        # for point in corners_crop_area:
        #     image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
        # result = Image(image)
        # result.show()

        # find contours
        contours = self.detect_contours()

        if len(contours) > 0:
            contourUsed = []
            # use edges having max length to detect the area of the scanned book
            threshold_num_edge = parameters.Contour.number_edge_upper_bound
            if len(contours) < threshold_num_edge:
                contourUsed = contours
            else:
                contourUsed = contours[:threshold_num_edge]

            edges = []
            edgeEndpoints = []
            for entity in contourUsed:
                # enclose the contour
                peri = cv2.arcLength(entity, True)
                factor = parameters.Contour.percentage_curve_length
                edge = cv2.approxPolyDP(entity, factor * peri, True)
                edges.append(edge)
                for points in edges:
                    for point in points:
                        edgeEndpoints.append(point)

                        # # show
            # CornerFrame = self.sourceImage.image.copy()
            # for edge in edges:
            #     CornerFrame = cv2.drawContours(CornerFrame, edge, -1, (255, 0, 255), 5)
            # result = Image(CornerFrame)
            # result.show()

            # remove the points that belong to background with respect to image width:
            # There is a thick horizontal strip (green) in the background (black). The  points alone its contours 
            # are also detected, e.g. the points lay near to the left and right boundaries. These points enlarge the width of 
            # "real area" and hence must be removed. 
            #                            ------------------------------------
            #                            |                                 |
            #                            |                                 |
            #                            |        *    *    *    *         |
            #                            |        *              *         |
            #                            | *      *              *       * |
            #                            | *      *              *       * |
            #                            |        *              *         |
            #                            |        *   *     *    *         |
            #                            |                                 |
            #                            ------------------------------------
            #
            edgeEndpoints_list = []
            for entry in edgeEndpoints:
                edgeEndpoints_list.append([entry[0][0], entry[0][1]])
            # The contour was approximated by connected polygon edges given by points at edge bounds. Therefore, the end points 
            # of innen polygon edges were given twice. 
            edgeEndpoints = np.unique(np.array(edgeEndpoints_list), axis=0)

            contourPoints = ContourPonts2D(point_list=edgeEndpoints)
            bound_factor = parameters.Contour.percentage_image_width
            lef_bound = math.ceil(self.sourceImage.width * bound_factor)
            right_bound = math.ceil(self.sourceImage.width * (1 - bound_factor))
            keptPoints, removedPoints = contourPoints.slice_x(leftBound=lef_bound, rightBound=right_bound)

            #    # show
            #     image = self.sourceImage.image.copy()
            #     for point in keptPoints:
            #         image = cv2.circle(image, point, radius=3, color=(255, 0, 255), thickness=-1)
            #     result = Image(image)
            #     result.show()

            # validate the kept points with respect to removed points:
            # the removed points should lay on the boundaries of the green strip in the background. Using their grayscale 
            # pixel-values colud remove the points nearby the book boundaries.
            #                            ------------------------------------
            #                            |                                 |
            #                            |                                 |
            #                            |        *    *    *    *         |
            #                            |        *              *         |
            #                            |      * *              * *       |
            #                            |      * *              * *       |
            #                            |        *              *         |
            #                            |        *   *     *    *         |
            #                            |                                 |
            #                            ------------------------------------

            threshold_gray = parameters.GrayScale.value_bound_upper
            #        distance_width = parameters.GrayScale.distance_width
            validPoints = []
            if len(removedPoints) > 0 and len(keptPoints) > 0:
                grayImage = self.sourceImage.convert2GRAYSCALE()
                for kept_point in keptPoints:
                    flag_kept = True
                    # ATTENTION: point has coordinate [width, height], iamge has coordinate [height, width]
                    value_kept = grayImage.image[kept_point[1], kept_point[0]]
                    for removed_point in removedPoints:
                        #    if np.abs(kept_point[1]-removed_point[1]) <= distance_width:
                        # ATTENTION: point has coordinate [width, height], iamge has coordinate [height, width]
                        value_removed = grayImage.image[removed_point[1], removed_point[0]]
                        if abs(int(value_kept) - int(value_removed)) <= threshold_gray:
                            flag_kept = False
                    if flag_kept:
                        validPoints.append(kept_point)
            else:
                validPoints = keptPoints.copy()

            #    # show
            #     image = self.sourceImage.image.copy()
            #     for point in validPoints:
            #         image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
            #     result = Image(image)
            #     result.show()

            # approximate points with a point that lay too tightly
            # This is preparation for detecting bottom/top/left/right -vertex
            # Since the contour were approximated by polygon, a corner of a books could be represented by more than two points 
            # laying near to each other. After this step, the corner is marked by only one point. Then, it is possible to calculate 
            # the rotation-angle of the book.     
            epsilon = parameters.tightness.radius
            points = validPoints
            pointsNotTightly = []

            while len(points) > 0:
                ind_processed = []
                cnt = 0
                foundPoints = []
                for point in points:
                    if cnt == 0:
                        foundPoints.append(point)
                        ind_processed.append(cnt)
                        point_ref = [point[0], point[1]]
                    else:
                        if np.abs(point_ref[0] - point[0]) <= epsilon and np.abs(point_ref[1] - point[1]) <= epsilon:
                            foundPoints.append(point)
                            ind_processed.append(cnt)
                    cnt = cnt + 1

                    # delete found points
                points = np.delete(points, ind_processed, axis=0)
                # aprroximate found points with one point:
                if len(foundPoints) == 1:
                    pointsNotTightly.append(foundPoints[0])
                else:
                    temp = ContourPonts2D(foundPoints)
                    x = math.ceil(temp.mean_x_value())
                    y = math.ceil(temp.mean_y_value())
                    pointsNotTightly.append([x, y])

            #    # show
            #     image = self.sourceImage.image.copy()
            #     for point in pointsNotTightly:
            #         image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
            #     result = Image(image)
            #     result.show()

            # remove the vertex points:
            # case 1: 
            #       *----------------------*
            # 
            #                * vertex=(x, y_max)
            # 
            distance_horizontal = parameters.vertex.distance_horizontal
            distance_vertical = parameters.vertex.distance_vertical
            keptPoints = ContourPonts2D(point_list=pointsNotTightly)
            y_vertex = keptPoints.max_y_values()
            min_x = keptPoints.min_x_values()
            max_x = keptPoints.max_x_values()
            for point in keptPoints.points:
                if point[1] == y_vertex:
                    vertex = point
                    break

            if vertex[0] >= min_x + distance_vertical and vertex[0] <= max_x - distance_vertical:
                count = 0
                for point in keptPoints.points:
                    if point[1] >= vertex[1] - distance_horizontal:
                        count = count + 1
                if count >= 2:
                    vertex = None
            else:
                vertex = None

            vertex_ind = keptPoints.find(point=vertex)
            keptPoints.delete(ind=vertex_ind)

            # # show 
            # image = self.sourceImage.image.copy()
            # for point in keptPoints.points:
            #     image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
            # result = Image(image)
            # result.show()

            # validate 
            num_keptPoints = len(keptPoints.points)
            if num_keptPoints < 1:
                img = img = self.sourceImage.image
                msg = msgHeader + ' found ' + str(num_keptPoints) + \
                      ' contour points, cannot build detect-area, return original image '
                logger.logger.info(msg)

            else:
                #    # show
                # rect boundaries
                #    corners = keptPoints.rect_boundary()
                #     image = self.sourceImage.image.copy()
                #     for point in corners:
                #         image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
                #     result = Image(image)
                #     result.show()

                # crop background 
                x_min = keptPoints.min_x_values()
                x_max = keptPoints.max_x_values()
                y_min = keptPoints.min_y_values()
                y_max = keptPoints.max_y_values()

                if x_max == x_min or y_max == y_min:  # crop area is a horizontal or vertcial line
                    img = img = self.sourceImage.image
                    msg = msgHeader + ' detected area is a vertical/horizontal line, return original image'
                    logger.logger.info(msg)
                else:
                    img = self.sourceImage.image[y_min:y_max + 1, x_min:x_max + 1, :]
                    corners_crop_area = keptPoints.rect_boundary()
                    # # show 
                    # image = self.sourceImage.image.copy()
                    # for point in corners_crop_area:
                    #     image = cv2.circle(image, point, radius=5, color=(255, 0, 255), thickness=-1)
                    # result = Image(image)
                    # result.show()

        else:  # case that no contours found: return original image
            img = self.sourceImage.image
            msg = msgHeader + 'not found any contour, return original image'
            logger.logger.info(msg)

        processedImage = ProcessedImage(imageArray=img)
        processedImage.set_filePath()
        processedImage.save(processedImage.filePath)
        processedImage.resolution_dpi = self.sourceImage.resolution_dpi

        self.processedImage = processedImage

        # result 
        res = ImageProcessingReuslts()
        res.filePath_sourceImage = self.sourceImage.filePath
        res.filePath_destIamge = self.processedImage.filePath
        res.crop_area = corners_crop_area
        return res

    def crop_background_manual(self, crop_area):
        msgHeader = ImageProcess.__name__ + '.crop_background_manual: '
        msg = msgHeader + 'manual crop background'
        logger.logger.info(msg)
        result = self.sourceImage.segmentation(area=crop_area)
        processedImage = ProcessedImage(imageArray=result.image)

        processedImage.set_filePath()
        processedImage.save(processedImage.filePath)
        processedImage.resolution_dpi = self.sourceImage.resolution_dpi

        self.processedImage = processedImage

        # result
        res = ImageProcessingReuslts()
        res.filePath_sourceImage = self.sourceImage.filePath
        res.filePath_destIamge = self.processedImage.filePath

        keptPoints = ContourPonts2D(point_list=crop_area)
        corners_crop_area = keptPoints.rect_boundary()
        res.crop_area = corners_crop_area
        return res
