

from PIL import Image
from ultralytics import YOLO

# # Load a pretrained YOLOv8n model
# model = YOLO('yolov8n.pt')

# # Run inference on 'bus.jpg'
# results = model('bus.jpg')  # results list

# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    
    
def setup_yolo_model():
    predictor = YOLO('yolov8n.pt')
    return predictor


def predict_and_visualize(image, predictor):
    result = predictor(image)
    im_array = result[0].plot()  # plot a BGR numpy array of predictions
    return im_array
    # image = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # return image