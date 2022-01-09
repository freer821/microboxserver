import unittest
from app import SourceImage
from app import Image
from app import ImageProcess
import os
import cv2
import numpy as np
import math

testDirPath = os.getcwd() + os.path.sep + "ImageProcessing" + os.path.sep + "test_images" + os.path.sep


class TestImageClass(unittest.TestCase):
    def test_save(self):
        filePath = testDirPath + "test_1.jpg"
        sourceImage = Image.load(filePath)
        filePathSave = testDirPath + "test_1_copy.tiff"
        sourceImage.save(filePathSave)
        savedImage = Image.load(filePathSave)
        os.remove(filePathSave)

        self.assertEqual(sourceImage.width, savedImage.width, "image width")
        self.assertEqual(sourceImage.height, savedImage.height, "image height")
        self.assertEqual(sourceImage.channels, savedImage.channels, "image channel")
        self.assertEqual(sourceImage.resolution_dpi, savedImage.resolution_dpi, "resolution")

        sourceImage = Image(None)
        sourceImage.save(filePathSave)
        self.assertEqual(os.path.isfile(filePathSave), False, "save none")

    def test_equal(self):
        arr1 = [1, 2, 3, 4]
        image1 = Image(arr1)

        image2 = image1.clone()
        self.assertTrue(image1 == image2, 'equal iamge')

        arr2 = None
        image2 = Image(arr2)
        self.assertTrue(not (image1 == image2), 'equal iamge')

        arr2 = None
        image2 = Image(arr2)
        image1 = image2.clone()
        self.assertTrue(image1 == image2, 'equal iamge')

    def test_Image_init(self):
        arr = [1, 2, 3]
        result = Image(arr)
        self.assertEqual(result.width, 3, 'image width')
        self.assertEqual(result.height, 1, 'image height')
        self.assertEqual(result.channels, 1, 'iamge channel')

        arr = None
        result = Image(arr)
        self.assertEqual(result.width, 0, 'image width')
        self.assertEqual(result.height, 0, 'image height')
        self.assertEqual(result.channels, 0, 'iamge channel')
        self.assertEqual(result.image, None, "pixel values")

        arr = [[[[1, 2, 3]]]]
        result = Image(arr)
        self.assertEqual(result.width, 0, 'image width')
        self.assertEqual(result.height, 0, 'image height')
        self.assertEqual(result.channels, 0, 'iamge channel')
        self.assertEqual(result.image, None, "pixel values")

    def test_load_image_from_file(self):
        filePath = testDirPath + "test_1.jpg"
        sourceImage = Image.load(filePath)
        self.assertEqual(sourceImage.width, 225, 'image width')
        self.assertEqual(sourceImage.height, 225, 'image height')
        self.assertEqual(sourceImage.channels, 3, 'iamge channel')
        self.assertEqual(sourceImage.resolution_dpi, 400, 'resoultion')

    def test_load_image_nonexist_file(self):
        filePath = testDirPath + "NonExist.dng"
        source = Image.load(filePath)
        self.assertEqual(source.image, None, "pixel values")
        self.assertTrue(source.width == 0, "width")
        self.assertTrue(source.height == 0, "height")
        self.assertTrue(source.channels == 0, "channel")

    def test_load_image_invalid_file(self):
        filePath = testDirPath + "invalid.txt"
        source = Image.load(filePath)
        self.assertEqual(source.image, None, "pixel values")
        self.assertTrue(source.width == 0, "width")
        self.assertTrue(source.height == 0, "height")
        self.assertTrue(source.channels == 0, "channel")

    def test_super_resolution_FSRCNN_2x(self):
        filePath = testDirPath + "test_1.jpg"
        sourceImage = Image.load(filePath)
        result = sourceImage.super_resolution_FSRCNN_2x()
        self.assertEqual(sourceImage.channels, result.channels, 'image channel')
        self.assertEqual(sourceImage.width * 2, result.width, "image width")
        self.assertEqual(sourceImage.height * 2, result.height, "image height")
        self.assertEqual(sourceImage.resolution_dpi * 2, result.resolution_dpi, "resolution")

    def test_super_resolution_FSRCNN_3x(self):
        filePath = testDirPath + "test_1.jpg"
        sourceImage = Image.load(filePath)
        result = sourceImage.super_resolution_FSRCNN_3x()
        #    result.show()
        self.assertEqual(sourceImage.channels, result.channels, 'image channel')
        self.assertEqual(sourceImage.width * 3, result.width, "image width")
        self.assertEqual(sourceImage.height * 3, result.height, "image height")
        self.assertEqual(sourceImage.resolution_dpi * 3, result.resolution_dpi, "resolution")

    def test_super_resolution_FSRCNN_4x(self):
        filePath = testDirPath + "test_1.jpg"
        sourceImage = Image.load(filePath)
        result = sourceImage.super_resolution_FSRCNN_4x()
        #        result.show()
        self.assertEqual(sourceImage.channels, result.channels, 'image channel')
        self.assertEqual(sourceImage.width * 4, result.width, "image width")
        self.assertEqual(sourceImage.height * 4, result.height, "image height")
        self.assertEqual(sourceImage.resolution_dpi * 4, result.resolution_dpi, "resolution")

    def test_schrink(self):
        filePath = testDirPath + "test_1.jpg"
        sourceImage = Image.load(filePath)
        # sourceImage.show()
        factor = 0.5
        result = sourceImage.schrink(scale_vaule=factor)
        # result.show()
        self.assertTrue(abs(sourceImage.width * factor - result.width) <= 1, "width")
        self.assertTrue(abs(sourceImage.height * factor - result.height) <= 1, "height")
        self.assertTrue(abs(sourceImage.resolution_dpi * factor - result.resolution_dpi) <= 1, "resolution")

    def test_change_resolution(self):
        filePath = testDirPath + "test_1.jpg"
        source = Image.load(filePath)
        dpi = 300
        width = math.ceil(source.width * dpi / source.resolution_dpi)
        height = math.ceil(source.height * dpi / source.resolution_dpi)

        result = source.change_resolution(dpi_value=dpi)
        # result.show()
        self.assertTrue(abs(width - result.width) <= 1, "width")
        self.assertTrue(abs(height - result.height) <= 1, "height")
        self.assertTrue(abs(dpi - result.resolution_dpi) <= 1, "resolution")

        dpi = 450
        width = math.ceil(source.width * dpi / source.resolution_dpi)
        height = math.ceil(source.height * dpi / source.resolution_dpi)

        result = source.change_resolution(dpi_value=dpi)
        # result.show()
        self.assertTrue(abs(width - result.width) <= 1, "width")
        self.assertTrue(abs(height - result.height) <= 1, "height")
        self.assertTrue(abs(dpi - result.resolution_dpi) <= 1, "resolution")


class TestSourceImageClass(unittest.TestCase):
    def test_init(self):
        filePath = None
        imageArray = None
        image = SourceImage(filePath=filePath, imageArray=imageArray)
        self.assertEqual(image.image, None, "pixel values")
        self.assertEqual(image.width, 0, "width")
        self.assertEqual(image.height, 0, "height")
        self.assertEqual(image.channels, 0, "channels")

        filePath = testDirPath + "test_1.jpg"
        imageArray = None
        image = SourceImage(filePath=filePath, imageArray=imageArray)
        self.assertEqual(image.width, 225, "width")
        self.assertEqual(image.height, 225, "height")
        self.assertEqual(image.channels, 3, "channels")
        self.assertEqual(image.resolution_dpi, 400, 'resoultion')

        filePath = testDirPath + "test_1.jpg"
        imag = cv2.imread(filePath)
        filePath = None
        imageArray = imag
        image = SourceImage(filePath=filePath, imageArray=imageArray)
        self.assertEqual(image.width, 225, "width")
        self.assertEqual(image.height, 225, "height")
        self.assertEqual(image.channels, 3, "channels")

        imageArray = cv2.imread(testDirPath + "test_1.jpg")
        filePath = testDirPath + "test_2.jpg"
        image = SourceImage(filePath=filePath, imageArray=imageArray)
        self.assertEqual(image.width, 280, "width")
        self.assertEqual(image.height, 180, "height")
        self.assertEqual(image.channels, 3, "channels")


class TestImageProcessClass(unittest.TestCase):
    def test_auto_crop(self):
        filePath = testDirPath + "FG1.dng"
        processingUnit = ImageProcess(sourceImageFile=filePath)
        result = processingUnit.crop_background_automatic()

    def test_manual_crop(self):
        filePath = testDirPath + "FG1.dng"
        crop_area = np.array([[136, 353], [460, 353], [460, 122], [136, 122]])
        processingUnit = ImageProcess(sourceImageFile=filePath)
        result = processingUnit.crop_background_manual(crop_area=crop_area)


if __name__ == "__main__":
    unittest.main()
