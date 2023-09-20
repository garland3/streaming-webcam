# conda activate FastSAM

from flask import Flask, Response
import cv2
import struct
import time
import cv2
import numpy as np
import socket
import pickle
import os

if not os.path.exists('frames'):
    os.makedirs('frames')


def receive_and_process_frames(sender_ip, port=8485):
    # Make folder called frames if it does not exist
    delimiter = b'ENDFRAME'
    buffer = b""
    
    # Initialize socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((sender_ip, port))

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
            # Save the frame to disk in a folder called frames
            # cv2.imwrite(f"frames/frame{count}.jpg", frame)
            if count % 100 ==0: print(f"Frame {count} written")
            count += 1
            
            yield frame  # Yield the frame for further processing or displaying
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except Exception as e:
            print(f"An error occurred: {e}")

    # Clean up
    client_socket.close()
    

model = "YOLO"
if model == "Detection2":
    from detection_wrapper import setup_detectron2_model, predict_and_visualize
    predictor = setup_detectron2_model()
elif model == "YOLO":
    from yolo_wrapper import setup_yolo_model, predict_and_visualize
    predictor = setup_yolo_model()

app = Flask(__name__)


def add_fps_to_frame(frame, fps):
    """Overlay FPS on the top-left corner of the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def generate_frames():
    # while True:
    sender_ip = "192.168.0.183"
    for frame in receive_and_process_frames(sender_ip):
        start_time = time.time()
        # Assuming `frame` is the image you got from the webcam
        processed_frame = predict_and_visualize(frame, predictor)
        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # Add FPS to frame
        add_fps_to_frame(processed_frame, fps)
        # Assume 'processed_frame' is your processed frame
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # add a hello world and a link to video. 
    respone_str = ""
    respone_str += "<h1>Hello World!</h1>"
    respone_str += "<a href='/video'>Video</a>"
    return respone_str


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
