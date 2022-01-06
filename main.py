import os

from flask import Flask, send_file, render_template
import cv2, numpy

template_dir = os.path.join(os.path.dirname(__file__), 'static')
print(template_dir)

app = Flask(__name__, template_folder=template_dir)

def convert_bytes2image(buff, width, height, channels):
    flatNumpyArray = numpy.array(buff)
    imgarray = flatNumpyArray.reshape(width, height, channels)
    # imgarray = cv2.cvtColor(imgarray, cv2.COLOR_BGRA2BGR)
    #cv2.imwrite('scan.jpeg', imgarray)
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
        return send_file('sources/images/test.tif', attachment_filename='test.tif')
    except Exception as ex:
        client.is_connected = False
        client.connect()
        print(ex)
        return 'no connection to server, try again'


@app.route("/")
def home():
    return render_template("index.html")