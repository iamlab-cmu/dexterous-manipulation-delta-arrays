import socket
import numpy as np
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 50000))

while True:
    data = sock.recv(1024)
    print(data)
    rot_error, pos_error, is_done = data.decode().split("?")
    print(is_done)
    arr1 = float(rot_error)
    arr2 = float(pos_error)
    arr3 = is_done == "True"

    print('Received array:', arr1, arr2, arr3)
    time.sleep(0.05)


# Close the socket
sock.close()