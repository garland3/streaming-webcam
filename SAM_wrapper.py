# %%
from typing import Tuple
import torch.nn.functional as F
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms.functional import resize, to_pil_image


def show_mask(mask, ax, random_color=False, save_file=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    if save_file is not None:
        plt.savefig(save_file)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None, save_filename=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis("on")
    if save_filename is not None:
        plt.savefig(save_filename)
        return 
    plt.show()



def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_masks_on_image(raw_image, masks, scores, save_file=None):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 5))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
        return
    plt.show()


# %%
def load_models_for_segmentation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    return model, processor, device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
# processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
# %%

# Function to load image
def load_image(image_path):
    return Image.open(image_path)

# Function to get image embeddings
def get_image_embeddings(raw_image, processor, model, device):
    inputs = processor(raw_image, return_tensors="pt").to(device)
    return model.get_image_embeddings(inputs["pixel_values"])



# Function to get segmentation masks and scores
def get_segmentation_masks(raw_image, image_embeddings, input_points, processor, model, device):
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    
    return masks, scores



# Function to save mask to disk
def save_mask_to_disk(mask,scores,  mask_path):
    if len(mask.shape) == 4:
        mask = mask.squeeze()
    idx_best = torch.argmax(scores).item()
    print("shape of mask is ", mask.shape)
    print("The unique values in mask are ", torch.unique(mask))
    mask = mask[idx_best,:,:]
    
    # permute to get channels last
    print("mask shape is ", mask.shape)
    # mask = mask.permute(1,2,0)
    mask = mask.numpy().astype(np.uint8)
    mask *= 255
    print("mask shape is ", mask.shape)
    # save with cv2
    cv2.imwrite(mask_path, mask)    
    
    
def read_points():
    filename = "temp/points.csv"
    with open(filename, 'r') as f:
        points = f.readlines()
    # convert to integers
    input_points = [[int(i) for i in p.split(",")] for p in points]
    return input_points

# ---------------------------------------
# personal segmentation (sam) section
# ---------------------------------------

def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def preprocess(
    x: torch.Tensor,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def prepare_mask(image, target_length=1024):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    mask = np.array(resize(to_pil_image(image), target_size))
    input_mask = torch.as_tensor(mask)
    input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_mask = preprocess(input_mask)

    return input_mask
def point_selection(mask_sim:torch.Tensor, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label

class SAMWrapper:
    def personal_sam_setup(self, processor, model, ref_image_filename, ref_image_mask_filename):
        ref_image = Image.open(ref_image_filename).convert("RGB")

        ref_mask = Image.open(ref_image_mask_filename).convert("RGB")
        ref_mask = np.array(ref_mask)
        
        pixel_values = processor(images=ref_image, return_tensors="pt").pixel_values

        # Step 1: Image features encoding
        with torch.no_grad():
            ref_feat = model.get_image_embeddings(pixel_values.to(device))
        # Step 2: interpolate reference mask
        print(f"Before prepare_mask, ref_mask shape is {ref_mask.shape}")
        ref_mask = prepare_mask(ref_mask)
        # If prepare_mask returns a mask with 3 channels, average them into one.
        if ref_mask.shape[1] == 3:
            ref_mask = ref_mask.mean(dim=1, keepdim=True)
        print(f"After prepare_mask, ref_mask shape is {ref_mask.shape}")
        print(f"Before interpolate, ref_mask shape is {ref_mask.shape}, and ref_feat shape is {ref_feat.shape}")
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]
        print(f"After interpolate, ref_mask shape is {ref_mask.shape}, and ref_feat shape is {ref_feat.shape}")

        # Step 3: Target feature extraction
        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        self.target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        print(f"target_feat shape is {self.target_feat.shape}")
        self.target_embedding = target_embedding.unsqueeze(0)

    def inference_on_new_image(self,filename_test_img=None, frame=None):
        if filename_test_img is None:
            test_image = frame
        else:
            test_image = Image.open(filename_test_img).convert("RGB")
        assert test_image is not None, "test_image is None"
        inputs = processor(images=test_image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        print(f"pixel_values shape is {pixel_values.shape} and the PIL Image test_image size is {test_image.size}")
        # image feature encoding
        with torch.no_grad():
            test_feat = model.get_image_embeddings(pixel_values).squeeze()
            print(f"test_feat shape is {test_feat.shape}")
        num_channels, height, width = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat_reshaped = test_feat.reshape(num_channels, height * width)
        sim = self.target_feat @ test_feat_reshaped
        sim = sim.reshape(1, 1, height, width)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = processor.post_process_masks(
            sim.unsqueeze(1),
            original_sizes=inputs["original_sizes"].tolist(),
            reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
            binarize=False,
        )
        sim = sim[0].squeeze()
        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)
        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)
        # prepare test image and prompts for the model
        inputs = processor(
            test_image,
            input_points=[topk_xy.tolist()],
            input_labels=[topk_label.tolist()],
            return_tensors="pt",
        ).to(device)
        # Run the loop 3 times (or however many times you want)
        n_loops = 3
        for i in range(n_loops):
            # First-step prediction or cascaded post-refinement
            with torch.no_grad():
                outputs = model(
                    input_points=inputs.input_points,
                    input_labels=inputs.input_labels,
                    input_masks=input_masks,
                    image_embeddings=test_feat.unsqueeze(0),
                    multimask_output=(input_masks is not None),
                    attention_similarity=attention_similarity if input_masks is None else None,
                    target_embedding=self.target_embedding if input_masks is None else None,
                )
            if i==0:
                input_masks = outputs.pred_masks.squeeze(1)[:, best_idx : best_idx + 1, :, :]
                continue
            
            # Post-process masks and find best index
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0].squeeze().numpy()
            best_idx = torch.argmax(outputs.iou_scores).item()
            if i== n_loops - 1:
                break
            
            v = np.nonzero(masks[best_idx])
            # print(f"V is {v}")
            y, x = v
            x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
            input_boxes = [[[x_min, y_min, x_max, y_max]]]
            
            # Update inputs
            inputs = processor(
                test_image,
                input_points=[topk_xy.tolist()],
                input_labels=[topk_label.tolist()],
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(device)
            
            # Prepare masks for the next iteration
            input_masks = outputs.pred_masks.squeeze(1)[:, best_idx : best_idx + 1, :, :]
        
        return masks[best_idx]
        
    
if __name__ == "__main__":
# %%
    # Load models
    
    model, processor, device = load_models_for_segmentation()
    print("Loaded models")
    # %%
    run_personal_sam = True
    if run_personal_sam:
        per_sam = SAMWrapper()
        per_sam.personal_sam_setup(processor, model, "static/previous_frame.jpg", "static/map-segmentation.jpg")
        print("Personal SAM setup done")
        current_frame_file = "static/frame.jpg"
        mask = per_sam.inference_on_new_image(current_frame_file)
        print("Inference done. Mask shape is ", mask.shape)
        # save the mask to temp/inference-mask.png
        save_mask_to_disk(mask, torch.Tensor([1.0]), "temp/inference-mask.png")
        
    else:

        # Load image
        raw_image = load_image("static/frame.jpg")
        # Get image embeddings
        image_embeddings = get_image_embeddings(raw_image, processor, model, device)
        # Get segmentation masks and scores
        input_points = [read_points()]
        print("The points are ", input_points)
        show_points_on_image(raw_image, input_points[0], save_filename="temp/image_with_points.png")

        # %%
        masks, scores = get_segmentation_masks(raw_image, image_embeddings, input_points, processor, model, device)

        # %%
        masks[0].shape

        # %%

        # Save mask to disk
        save_mask_to_disk(masks[0],scores,  "static/map-segmentation.jpg")
        # Show mask on image
        show_masks_on_image(raw_image, masks[0], scores, save_file="static/segmentation.jpg")
# %%
