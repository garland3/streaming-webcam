# %%
# !pip install -q git+https://github.com/huggingface/transformers.git
import matplotlib.pyplot as plt
from transformers import AutoProcessor, SamModel
import cv2
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
import torch
from typing import Tuple
from utils import (
    get_preprocess_shape,
    preprocess,
    prepare_mask,
    point_selection,
    show_mask,
)
from huggingface_hub import hf_hub_download
from PIL import Image

# %%
# from transformers import PerSamModel
processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
# model = PerSamModel.from_pretrained("facebook/sam-vit-huge")
model = SamModel.from_pretrained("facebook/sam-vit-huge")

# %%


# filename = hf_hub_download(repo_id="nielsr/persam-dog", filename="dog.jpg", repo_type="dataset")
filename = "/home/garlan/git/streamingwebcam/imgs/frame-0001.jpg"
ref_image = Image.open(filename).convert("RGB")
ref_image

# %%
filename = "/home/garlan/git/streamingwebcam/playground/mask-tensile1.png"
ref_mask = Image.open(filename).convert("RGB")
ref_mask = np.array(ref_mask)
# %%
filename_test_img = "/home/garlan/git/streamingwebcam/imgs/frame-0134.jpg"
test_image = Image.open(filename_test_img).convert("RGB").convert("RGB")
test_image
# %%
pixel_values = processor(images=ref_image, return_tensors="pt").pixel_values
print(pixel_values.shape)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# %%
# Step 1: Image features encoding
with torch.no_grad():
    ref_feat = model.get_image_embeddings(pixel_values.to(device))
    
# %%
ref_feat.shape, pixel_values.shape
# %%

ref_feat = ref_feat.squeeze().permute(1, 2, 0)

# %%
ref_feat.shape
# %%

# Step 2: interpolate reference mask
print(f"Before prepare_mask, ref_mask shape is {ref_mask.shape}")
ref_mask = prepare_mask(ref_mask)
print(f"After prepare_mask, ref_mask shape is {ref_mask.shape}")
# %%
print(f"Before interpolate, ref_mask shape is {ref_mask.shape}, and ref_feat shape is {ref_feat.shape}")
ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
ref_mask = ref_mask.squeeze()[0]
print(f"After interpolate, ref_mask shape is {ref_mask.shape}, and ref_feat shape is {ref_feat.shape}")
# %%

# Step 3: Target feature extraction
target_feat = ref_feat[ref_mask > 0]
target_embedding = target_feat.mean(0).unsqueeze(0)
target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
target_embedding = target_embedding.unsqueeze(0)
print(target_embedding.shape)
print(f"target_feat shape is {target_feat.shape}")
# %%
plt.plot(target_embedding.cpu().numpy().squeeze())
# %%

"""## Step 2: calculate cosine similarity

Next, using this target embedding, we can acquire a location confidence map of where the target concept is present in the test image by calculating the cosine similarity between the target embedding and the image features of the test image.
"""

# %%
# ------------ process the new image. 
# %%

filename_test_img = "/home/garlan/git/streamingwebcam/imgs/frame-0134.jpg"
test_image = Image.open(filename_test_img).convert("RGB").convert("RGB")
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
sim = target_feat @ test_feat_reshaped
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
n_loops = 5
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
            target_embedding=target_embedding if input_masks is None else None,
        )
    if i== n_loops - 1:
        break
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

"""## Visualize mask

We can visualize the predicted mask on top of our test image:
"""


fig, axes = plt.subplots()

best_idx = torch.argmax(outputs.iou_scores).item()
axes.imshow(np.array(test_image))
show_mask(masks[best_idx], axes)
axes.title.set_text(f"Predicted mask")
axes.axis("off")


# %%
