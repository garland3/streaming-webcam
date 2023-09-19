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
            
            # ML processing and other tasks can go here
            # ...
            
            # Save the frame to disk in a folder called frames
            # cv2.imwrite(f"frames/frame{count}.jpg", frame)
            print(f"Frame {count} written")
            count += 1
            
            yield frame  # Yield the frame for further processing or displaying
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except Exception as e:
            print(f"An error occurred: {e}")

    # Clean up
    client_socket.close()
    
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_detectron2_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

# def predict_and_visualize(image, predictor):
#     outputs = predictor(image)
#     v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]), scale=1.0)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     return out.get_image()[:, :, ::-1]
import cv2
import numpy as np

def predict_and_visualize(image, predictor):
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    # Fixed color map
    fixed_colors = {
        "person": (0, 0, 255),
        "car": (0, 255, 0),
        # Add more class-color mappings as needed
    }
    
    # Convert masks to numpy arrays and get other instance attributes
    pred_masks = instances.pred_masks.numpy()
    pred_classes = instances.pred_classes.tolist()
    class_names = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]).thing_classes
    
    for i, pred_class in enumerate(pred_classes):
        mask = pred_masks[i].astype(np.uint8)
        color = np.array(fixed_colors.get(class_names[pred_class], (255, 255, 255)), dtype=np.uint8)  # Default to white

        # Convert binary mask to 3-channel mask
        colored_mask = cv2.merge([mask * color_val for color_val in color])

        # Overlay mask on image using cv2.addWeighted
        image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    return image


# Assuming `predictor` is already set up and `frame` is your input image
# processed_frame = predict_and_visualize(frame, predictor)

# Example Usage
predictor = setup_detectron2_model()

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
