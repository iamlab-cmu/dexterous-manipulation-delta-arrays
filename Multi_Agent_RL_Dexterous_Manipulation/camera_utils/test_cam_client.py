import socket
import pickle
import cv2
import base64
import numpy as np
from PIL import Image
import io

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000)
ip = "127.0.0.1"
port = 8080
s.bind((ip, port))

while True:
    x = s.recvfrom(2000000)
    data = x[0]
    data = pickle.loads(data)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
s.close()