import struct
import time
import cv2
import numpy as np
import socket
import pickle

# make folder called frames if it does not exist
import os
if not os.path.exists('frames'):
    os.makedirs('frames')

delimiter = b'ENDFRAME'
buffer = b""

# Initialize socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sender_ip = "192.168.0.183"
client_socket.connect((sender_ip, 8485))

data = b""
payload_size = 4096
count = 0
while True:
    try:
        while delimiter not in buffer:
            packet = client_socket.recv(4096)
            if not packet: break
            buffer += packet

        frame_data, buffer = buffer.split(delimiter, 1)
        
        data_length = struct.unpack("<L", frame_data[:4])[0]

        if len(frame_data[4:]) != data_length:
            print("Data length mismatch. Resynchronizing...")
            continue

        frame = pickle.loads(frame_data[4:])

    
        # ML processing and other tasks
        # ...
        time.sleep(1.0) 
        # save the frame to disk in a folder caled frames
        cv2.imwrite("frames/frame%d.jpg" % count, frame)
        print("Frame %d written" % count)
        count += 1

        # Optional: show the frame
        # cv2.imshow('Receiver', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Clean up
# cv2.destroyAllWindows()
client_socket.close()
