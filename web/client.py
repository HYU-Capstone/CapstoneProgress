from picamera import PiCamera
import time
from PIL import Image
import socket
import base64
import requests
from io import BytesIO

IP = '192.168.29.159'
USE_PROXY = True
PORT = 5001
PROXY_IP = '192.168.29.159'
PROXY_PORT = 8888
IMAGE_SIZE = (1920, 1080)

with PiCamera(resolution=IMAGE_SIZE) as camera:
    stream = BytesIO()
    s = time.time()
    for foo in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        # Truncate the stream to the current position (in case
        # prior iterations output a longer image)
        elapsed = time.time() - s
        stream.truncate()
        stream.seek(0)
        barray = stream.read(stream.getbuffer().nbytes)
        requests.post('http://{}:{}/match'.format(IP, PORT), files={'image': barray}, proxies={
            'http': '{}:{}'.format(PROXY_IP, PROXY_PORT),
            'https': '{}:{}'.format(PROXY_IP, PROXY_PORT)
        })
        print('Time elapsed to get binary image from camera:', elapsed)
        e = time.time() - s
        print('Total time elapsed:', e)
        with open('test_{}.jpg'.format(time.time()), 'wb') as fw:
            fw.write(barray)
        # input()
        stream.seek(0)
        s = time.time()
