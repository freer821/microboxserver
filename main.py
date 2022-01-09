import os

from flask import Flask, send_file, render_template, request
import cv2, numpy
from PIL import Image

template_dir = os.path.join(os.path.dirname(__file__), 'static')

app = Flask(__name__, template_folder=template_dir)


def convert_bytes2image(buff, width, height, channels):
    flatNumpyArray = numpy.array(buff)
    imgarray = flatNumpyArray.reshape(width, height, channels)
    # imgarray = cv2.cvtColor(imgarray, cv2.COLOR_BGRA2BGR)
    # cv2.imwrite('scan.jpeg', imgarray)
    success, encoded_image = cv2.imencode('.jpeg', imgarray)
    return encoded_image


@app.route("/scan")
def scan():
    try:
        '''
        image_buff, width, height, channels = command_scan_image()
        image_bytes = convert_bytes2image(image_buff, width, height, channels)
        return send_file(io.BytesIO(image_bytes),
                         attachment_filename='scan.jpeg',
                         mimetype='image/jpeg')
        '''
        return send_file('sources/images/test.jpg', attachment_filename='test.jpg')
    except Exception as ex:
        return 'no connection to server, try again'

@app.route("/get-original")
def get_oiriginal():
    try:
        img_name = request.args.get('img_name')
        return send_file('sources/images/test.ppm', attachment_filename='test.ppm')
    except Exception as ex:
        return 'no connection to server, try again'

@app.route("/image-ouput")
def imageoutput():
    try:
        '''
        img_name = request.args.get('img_name')
        dpi = request.args.get('dpi')
        format = request.args.get('format')
        '''
        im = Image.open('resources/images/test.tif')
        print(im.info['dpi'])
        im.save("test-600.pdf", dpi=(600, 600))
        im.save("test-1200.tif", dpi=(1200, 1200))

        return send_file('resources/images/test.jpg', attachment_filename='test.jpg')
    except Exception as ex:
        print (ex)
        return 'no connection to server, try again'

@app.route("/")
def home():
    return render_template("index.html")
