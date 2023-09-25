# conda activate FastSAM

SENDER_IP = "192.168.0.115"
 
 
from flask import Flask, Response, redirect, render_template, request
import cv2
import struct
import time
import cv2
import numpy as np
import socket
import pickle
import os

class GetFrames:
    def __init__(self):

        if not os.path.exists('frames'):
            os.makedirs('frames')


    def receive_and_process_frames(self, sender_ip, port=8485, save_frames=False, break_after_1 = False):
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
                if save_frames:
                    self.filenname = f"frames/frame{count}.jpg"
                    cv2.imwrite(self.filenname, frame)
                if break_after_1:
                    break
                if count % 100 ==0: print(f"Frame {count} written")
                count += 1
                
                yield frame  # Yield the frame for further processing or displaying
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(f"An error occurred: {e}")

        # Clean up
        client_socket.close()
    

model = "Segmentation"
if model == "Detection2":
    from detection_wrapper import setup_detectron2_model, predict_and_visualize
    predictor = setup_detectron2_model()
elif model == "YOLO":
    from yolo_wrapper import setup_yolo_model, predict_and_visualize
    predictor = setup_yolo_model()
elif model == "Segmentation":
    from SAM_wrapper import (load_models_for_segmentation, get_image_embeddings, get_segmentation_masks, 
                             show_masks_on_image, load_image, read_points, save_mask_to_disk, show_points_on_image)
    model, processor, device = load_models_for_segmentation()
    
    personal_sam_obj = None
    
app = Flask(__name__)



# make 'static' folder if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# Mounting static files in Flask
app.static_folder = 'static'


def add_fps_to_frame(frame, fps):
    """Overlay FPS on the top-left corner of the frame."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def generate_frames():
    # while True:
    gf = GetFrames()
   
    for frame in gf.receive_and_process_frames(SENDER_IP):
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
    respone_str += "<p><a href='/video'>Video</a></p>"
    respone_str += "<p><a href='/annotate_image'>Annotate Image</a></p>"
    return respone_str

# allow submitting x and y array of coordinates
@app.route('/submitpoints', methods=['POST'])
def submitpoints():
    x = request.form['x']
    y = request.form['y']
    print(x)
    print(y)
    # split by comma and remove the px
    xpoints = [int(i[:-2]) for i in x.split(",")]
    ypoints = [int(i[:-2]) for i in y.split(",")]
    print("xpoints are ", xpoints)
    print("ypoints are ", ypoints)
    if len(xpoints)==0:
        # redirect to the anotate image page
        return redirect("/annotate_image")
    # make dir called 'temp' if it does not exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    # save a .csv file with the x and y coordinates
    filename = "temp/points.csv"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        for i in range(len(xpoints)):
            f.write(f"{xpoints[i]},{ypoints[i]}\n")
           
    segment_example_object() 
    return redirect("/annotate_image")
    # return "OK"


# @app.reoute("/segment_example")
def segment_example_object():
    # Load image
    raw_image = load_image("static/frame.jpg")
    # Get image embeddings
    image_embeddings = get_image_embeddings(raw_image, processor, model, device)
    # Get segmentation masks and scores
    input_points = [read_points()]
    print("The points are ", input_points)
    show_points_on_image(raw_image, input_points[0], save_filename="temp/image_with_points.png")

    masks, scores = get_segmentation_masks(raw_image, image_embeddings, input_points, processor, model, device)
    masks[0].shape
    # Save mask to disk
    save_mask_to_disk(masks[0],scores,  "static/map-segmentation.jpg")
    # Show mask on image
    show_masks_on_image(raw_image, masks[0], scores, save_file="static/segmentation.jpg")
    # copy static/frame.jpg to  static/previous_frame.jpg
    # del existing static/prevoius_frame.jpg if it exists
    if os.path.exists("static/previous_frame.jpg"):
        os.remove("static/previous_frame.jpg")
    os.rename("static/frame.jpg", "static/previous_frame.jpg")
    
@app.route("/annotate_image")
def annotate_image():
    gf = GetFrames()
    frame = next(gf.receive_and_process_frames(SENDER_IP))
    filename = "static/frame.jpg"
    cv2.imwrite(filename, frame)
    height, width, channels = frame.shape
    
    segmentation_url = None
    seg_height = None
    seg_width = None
    if os.path.exists("static/segmentation.jpg"):
        segmentation = cv2.imread("static/segmentation.jpg")
        seg_height, seg_width, _ = segmentation.shape
        reduction_ratio = 2
        seg_width = int(seg_width / reduction_ratio)
        seg_height = int(seg_height / reduction_ratio)
        segmentation_url = "/static/segmentation.jpg"

    return render_template(
        "annotate_image.html",
        image_url="/static/frame.jpg",
        width=width,
        height=height,
        segmentation_url=segmentation_url,
        seg_width=seg_width,
        seg_height=seg_height,
    )
    
@app.route("/frame_segmentation")
def frame_segmentation():
    gf = GetFrames()
    frame = next(gf.receive_and_process_frames(SENDER_IP))
    filename = "static/frame.jpg"
    cv2.imwrite(filename, frame)
    height, width, channels = frame.shape
    
   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
