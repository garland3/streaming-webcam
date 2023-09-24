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
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, SamModel
from huggingface_hub import hf_hub_download
import torch.nn.functional as F


class ImageProcessor:
    def __init__(self, model_name="facebook/sam-vit-huge"):
        self.model_name = model_name
        # self.load_models()

    def load_models(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SamModel.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    @staticmethod
    def load_image(filename):
        return torch.Tensor(np.array(Image.open(filename).convert("RGB")))

    @staticmethod
    def prepare_mask(image, target_length=1024):
        target_size = ImageProcessor.get_preprocess_shape(
            image.shape[0], image.shape[1], target_length
        )
        mask = np.array(resize(to_pil_image(image), target_size))
        input_mask = torch.as_tensor(mask)
        input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_mask = preprocess(input_mask)
        return input_mask

    @staticmethod
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

    def encode_features(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            ref_feat = self.model.get_image_embeddings(pixel_values.to(self.device))
        return ref_feat.squeeze().permute(1, 2, 0)

    def interpolate_mask(self, mask, ref_feat):
        mask = self.prepare_mask(mask)
        return F.interpolate(mask, size=ref_feat.shape[0:2], mode="bilinear").squeeze()[
            0
        ]

    def extract_target_features(self, ref_feat, ref_mask):
        target_feat = ref_feat[ref_mask > 0]
        target_embedding = target_feat.mean(0).unsqueeze(0)
        return target_embedding / target_embedding.norm(dim=-1, keepdim=True)


# Example usage
# if __name__ == "__main__":
# %%

processor = ImageProcessor()
# %%
ref_image = processor.load_image("/home/garlan/git/streamingwebcam/imgs/frame-0001.jpg")
ref_mask = processor.load_image(
    "/home/garlan/git/streamingwebcam/playground/mask-tensile1.png"
)
test_image = processor.load_image(
    "/home/garlan/git/streamingwebcam/imgs/frame-0033.jpg"
)

ref_feat = processor.encode_features(ref_image)
ref_mask = processor.interpolate_mask(ref_mask, ref_feat)

target_embedding = processor.extract_target_features(ref_feat, ref_mask)
print(target_embedding.shape)

# %%

"""## Step 2: calculate cosine similarity

Next, using this target embedding, we can acquire a location confidence map of where the target concept is present in the test image by calculating the cosine similarity between the target embedding and the image features of the test image.
"""

# prepare test image for the model
inputs = processor(images=test_image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

# image feature encoding
with torch.no_grad():
    test_feat = model.get_image_embeddings(pixel_values).squeeze()

# Cosine similarity
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

"""## Step 3: obtain location priors

The next step is getting good "prompts" for SAM in the form of coordinates (point locations). These prompts are also called "location priors", which indicate where the target concept (in this case, a dog) might be or not be at all in the test image.

The PerSAM authors select two pixel coordinates with the highest and lowest similarity values. The former represents the most likely foreground position of the target object, while the latter inversely indicates the background.

As we now, SAM is very good at generating a mask given a "prompt". So we basically come up with very good "prompts" to make sure SAM will be able to segment the dog in the new image.
"""


# Positive-negative location prior
topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

# Obtain the target guidance for cross-attention layers
sim = (sim - sim.mean()) / torch.std(sim)
sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)
print("Shape of attention_similarity:", attention_similarity.shape)
print("First values of attention_similarity:", attention_similarity[0, 0, :3, :3])

test_feat[0, :3, :3]

test_feat[0, :3, :3]

"""## Step 4: first step prediction

Next, we can prompt SAM to generate a mask for the target concept (the dog) in the test image, using the prompts we prepared below. Interestingly here is that the authors add "Target-guided Attention" and "Target-semantic prompting" to the decoder. This allows for more explicit guidance to the cross-attention mechanisms in SAM’s decoder, which concentrates feature aggregation within foreground target regions. This ultimately leads to more accurate masks when personalizing SAM in a training-free manner.
"""

# prepare test image and prompts for the model
inputs = processor(
    test_image,
    input_points=[topk_xy.tolist()],
    input_labels=[topk_label.tolist()],
    return_tensors="pt",
).to(device)
for k, v in inputs.items():
    print(k, v.shape)

attention_similarity.shape

target_embedding.shape

# First-step prediction
with torch.no_grad():
    outputs = model(
        input_points=inputs.input_points,
        input_labels=inputs.input_labels,
        image_embeddings=test_feat.unsqueeze(0),
        multimask_output=False,
        attention_similarity=attention_similarity,  # Target-guided Attention
        target_embedding=target_embedding,  # Target-semantic Prompting
    )
    best_idx = 0

"""## Step 5: 2-step cascaded post-refinement

Finally, the authors propose to further improve the segmentation mask using cascaded post-refinement. The initial mask (predicted above) might include some rough edges and isolated noises in the background. The authors propose to iteratively feed the mask back into SAM’s decoder for a two-step post-processing.
"""

# Cascaded Post-refinement-1
with torch.no_grad():
    outputs_1 = model(
        input_points=inputs.input_points,
        input_labels=inputs.input_labels,
        input_masks=outputs.pred_masks.squeeze(1)[best_idx : best_idx + 1, :, :],
        image_embeddings=test_feat.unsqueeze(0),
        multimask_output=True,
    )

# Cascaded Post-refinement-2
masks = (
    processor.image_processor.post_process_masks(
        outputs_1.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    .squeeze()
    .numpy()
)

best_idx = torch.argmax(outputs_1.iou_scores).item()
y, x = np.nonzero(masks[best_idx])
x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()
input_boxes = [[[x_min, y_min, x_max, y_max]]]

inputs = processor(
    test_image,
    input_points=[topk_xy.tolist()],
    input_labels=[topk_label.tolist()],
    input_boxes=input_boxes,
    return_tensors="pt",
).to(device)

final_outputs = model(
    input_points=inputs.input_points,
    input_labels=inputs.input_labels,
    input_boxes=inputs.input_boxes,
    input_masks=outputs_1.pred_masks.squeeze(1)[:, best_idx : best_idx + 1, :, :],
    image_embeddings=test_feat.unsqueeze(0),
    multimask_output=True,
)

masks = (
    processor.image_processor.post_process_masks(
        final_outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    .squeeze()
    .numpy()
)

"""## Visualize mask

We can visualize the predicted mask on top of our test image:
"""


fig, axes = plt.subplots()

best_idx = torch.argmax(final_outputs.iou_scores).item()
axes.imshow(np.array(test_image))
show_mask(masks[best_idx], axes)
axes.title.set_text(f"Predicted mask")
axes.axis("off")


# %%
