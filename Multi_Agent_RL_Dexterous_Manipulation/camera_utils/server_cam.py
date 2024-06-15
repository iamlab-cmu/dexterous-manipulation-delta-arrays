import cv2
import threading
import pickle
import socket
import sys

# Global variable to store the current HSV frame
current_hsv_frame = None
lock = threading.Lock()

def capture_and_convert():
    global current_hsv_frame
    cap = cv2.VideoCapture(0)  # Open the first camera

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with lock:
            current_hsv_frame = hsv_frame

def start_capture_thread():
    capture_thread = threading.Thread(target=capture_and_convert)
    capture_thread.daemon = True
    capture_thread.start()

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000)
    server_socket.bind(('localhost', 8080))
    server_socket.listen(1)
    print("Server started at port 8080")

    while True:
        client_socket, addr = server_socket.accept()
        with lock:
            if current_hsv_frame is not None:
                data = pickle.dumps(current_hsv_frame)
                byte_length = sys.getsizeof(data)
                print("The byte length of the variable is:", byte_length)
                client_socket.sendall(data)
        client_socket.close()

if __name__ == "__main__":
    start_capture_thread()
    print("Capture thread started. Use another script to fetch the current frame.")        
    start_server()