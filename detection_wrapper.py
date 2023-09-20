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