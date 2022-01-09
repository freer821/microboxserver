import cv2
from image_processing import Image, ContourPonts2D
import os
from log_book import logger
import numpy as np


# Lit.: https://www.its404.com/article/qq_41562704/88975569

class GestureImage(Image):
    def __init__(self, imageArray=None):
        msgHeader = GestureImage.__name__ + ".__init__: "
        super().__init__(imageArray=imageArray)

    def skinMask_HSV(self):
        # 0<=H<=10，S>=48，V>=50
        msgHeader = GestureImage.__name__ + '.skinMask_HSV: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot detect skin'
            logger.logger.warning(msg)
        else:
            low = np.array([0, 48, 50])
            high = np.array([10, 255, 255])
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            skin = cv2.inRange(hsv, low, high)
            img = cv2.bitwise_and(self.image, self.image, mask=skin)

        return GestureImage(imageArray=img)

    def skinMask_CrCb(self):
        # 146<=Cr<=173 77<=Cb<=127
        msgHeader = GestureImage.__name__ + '.skinMask_CrCb: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot detect skin'
            logger.logger.warning(msg)
        else:
            YCrCb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
            (y, cr, cb) = cv2.split(YCrCb)
            skin = np.zeros(cr.shape, dtype=np.uint8)
            (x, y) = cr.shape
            for i in range(0, x):
                for j in range(0, y):
                    if (cr[i][j] > 146) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
                        skin[i][j] = 255

            img = cv2.bitwise_and(self.image, self.image, mask=skin)

        return GestureImage(imageArray=img)

    def skinMask_YCrCb(self):
        msgHeader = GestureImage.__name__ + '.skinMask_YCrCb: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot detect skin'
            logger.logger.warning(msg)
        else:
            YCrCb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
            (y, cr, cb) = cv2.split(YCrCb)
            cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
            _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.bitwise_and(self.image, self.image, mask=skin)

        return GestureImage(imageArray=img)

    def skinMask_Ellipse(self):
        msgHeader = GestureImage.__name__ + '.skinMask_Ellipse: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot detect skin'
            logger.logger.warning(msg)
        else:
            skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
            # cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1)
            cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)
            YCrCb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
            (y, Cr, Cb) = cv2.split(YCrCb)
            skin = np.zeros(Cr.shape, dtype=np.uint8)
            (x, y) = Cr.shape
            for i in range(0, x):
                for j in range(0, y):
                    if skinCrCbHist[Cr[i][j], Cb[i][j]] > 0:
                        skin[i][j] = 255

            img = cv2.bitwise_and(self.image, self.image, mask=skin)

        return GestureImage(imageArray=img)

    def morphological_processing(self):
        msgHeader = GestureImage.__name__ + '.morphological_processing: '
        img = None
        if self.image is None:
            msg = msgHeader + ' image array is empty, cannot morphological processing'
            logger.logger.warning(msg)
        else:
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.erode(self.image, kernel)
            img = cv2.dilate(img, kernel)

        return GestureImage(imageArray=img)

    def remove_gesture(self):
        msgHeader = GestureImage.__name__ + ".remove_gesture:"
        if self.image is None:
            msg = msgHeader + "image array is empty, cannot remove gesture"
            logger.logger.warning(msg)
            return GestureImage(imageArray=None)

        # denosie 
        denoisedImage = self.denoise_Birateral()
        denoisedImage = GestureImage(imageArray=denoisedImage.image)

        # find mask for gesture 
        gestureImage = denoisedImage.skinMask_HSV()
        gestureImage = gestureImage.morphological_processing()

        # find contours for gesture 
        edgeImage = gestureImage.detect_edges_Canny()
        contours, hierarchy = cv2.findContours(edgeImage.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # mask for gesture 
        RECT_MASK_BOUND = 3
        mask = np.zeros(edgeImage.image.shape, np.uint8)
        mask_rects = []
        for contour in contours:
            contourPoints = []
            for point in contour:
                point = np.squeeze(point)
                contourPoints.append(point)

            contourPoints = ContourPonts2D(point_list=contourPoints)
            width_left = contourPoints.min_x_values()
            width_right = contourPoints.max_x_values()
            delta_width = RECT_MASK_BOUND
            width_left = max(0, width_left - delta_width)
            width_right = min(sourceImg.width, width_right + delta_width)

            height_bottom = contourPoints.min_y_values()
            height_top = contourPoints.max_y_values()
            delta_height = RECT_MASK_BOUND
            height_bottom = max(0, height_bottom - delta_height)
            height_top = min(sourceImg.height, height_top + delta_height)

            corner_lt = [width_left, height_top]
            corner_rt = [width_right, height_top]
            corner_rb = [width_right, height_bottom]
            corner_lb = [width_left, height_bottom]
            rect = [corner_lt, corner_rt, corner_rb, corner_lb]
            mask_rects.append(rect)

            for height in range(mask.shape[0]):
                if height >= height_bottom and height <= height_top:
                    for width in range(mask.shape[1]):
                        if width >= width_left and width <= width_right:
                            mask[height, width] = 255

            # res = Image(mask)
            # res.show()

        img = cv2.inpaint(sourceImg.image, mask, 3, cv2.INPAINT_NS)
        res = GestureImage(img)
        return res


if __name__ == "__main__":
    testDirPath = os.getcwd() + os.path.sep + "ImageProcessing" + os.path.sep + "test_images" + os.path.sep
    filePath = testDirPath + "FG1.dng"
    filePath = testDirPath + "FG2.tiff"
    #    filePath = testDirPath+"FG4.tiff"
    sourceImg = Image.load(filePath)
    sourceImg = GestureImage(imageArray=sourceImg.image)
    sourceImg.show()
    result = sourceImg.remove_gesture()
    result.show()

    print("ok")