from picamera import PiCamera
import time
from PIL import Image
import socket
import base64
import requests
from io import BytesIO

#IP = '192.168.29.159'
IP = '210.89.190.5'
#IP = '0.0.0.0'
#USE_PROXY = True
USE_PROXY = False
PORT = 5000
IMAGE_SIZE = (1280, 720)

time.sleep(5)

images = []

uid = input('UserID?: ').strip()

with PiCamera(resolution=IMAGE_SIZE) as camera:
    stream = BytesIO()
    s = time.time()
    input('Press enter when ready')
    for foo in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        # Truncate the stream to the current position (in case
        # prior iterations output a longer image)
        stream.truncate()
        stream.seek(0)
        barray = stream.read(stream.getbuffer().nbytes)
        images.append(barray)
        input('Press enter when ready')
        if len(images) == 5:
            break

    result = requests.post('http://{}:{}/api/users/{}/train'.format(IP, PORT, uid), files={
      'file1': images[0],
      'file2': images[1],
      'file3': images[2],
      'file4': images[3],
      'file5': images[4],
    })
    
    print(result.json())