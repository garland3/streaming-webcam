# %%
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
from PIL import Image

from utils import (
    get_preprocess_shape,
    preprocess,
    prepare_mask,
    point_selection,
    show_mask,
)


class OneShotSegmentation:
    def __init__(self):
        pass

    def load_model(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def open_image_and_mask(self, img_path, mask_path):
        self.ref_image = Image.open(img_path).convert("RGB")
        self.ref_mask = Image.open(mask_path).convert("RGB")
        self.ref_mask = np.array(self.ref_mask)

    def show_image_and_mask(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.ref_image)
        ax[1].imshow(self.ref_mask)
        plt.show()

    def encode_image(self, image):
        self.inputs = self.processor(images=image, return_tensors="pt")
        # pixel_values
        pixel_values = self.inputs.pixel_values
        with torch.no_grad():
            ref_feat = self.model.get_image_embeddings(pixel_values.to(self.device))
        return ref_feat.squeeze()

    def process_one_shot_img_and_mask(self):
        self.ref_feat = self.encode_image(self.ref_image)
        self.ref_feat = self.ref_feat.squeeze().permute(1, 2, 0)
        self.ref_mask = prepare_mask(self.ref_mask)
        self.ref_mask = F.interpolate(
            self.ref_mask, size=self.ref_feat.shape[0:2], mode="bilinear"
        )
        # calculate similarity, the mean normalized
        self.ref_mask = self.ref_mask.squeeze()[0]

        # Step 3: Target feature extraction
        self.target_feat = self.ref_feat[self.ref_mask > 0]
        self.target_embedding = self.target_feat.mean(0).unsqueeze(0)
        self.target_feat = self.target_embedding / self.target_embedding.norm(
            dim=-1, keepdim=True
        )
        # self.target_embedding = self.target_embedding.unsqueeze(0)

    def process_new_img(self, filename):
        test_image = Image.open(filename).convert("RGB")
        self.test_feat = self.encode_image(test_image)
        # Step 4: Similarity calculation
        self.test_feat = self.test_feat / self.test_feat.norm(dim=-1, keepdim=True)
        print(f"test_feat shape is {self.test_feat.shape}")
        num_channels, height, width = self.test_feat.shape
        self.test_feat_reshaped = self.test_feat.reshape(num_channels, height * width)
        print(f"test_feat_reshaped shape is {self.test_feat_reshaped.shape}")

        print("Self.target_feat shape is ", self.target_feat.shape)
        sim = self.target_feat @ self.test_feat_reshaped

        sim = sim.reshape(1, 1, height, width)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")

        sim = self.processor.post_process_masks(
            sim.unsqueeze(1),
            original_sizes=self.inputs["original_sizes"].tolist(),
            reshaped_input_sizes=self.inputs["reshaped_input_sizes"].tolist(),
            binarize=False,
        )
        sim = sim[0].squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear"
        )
        attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)
        print("Shape of attention_similarity:", attention_similarity.shape)
        print(
            "First values of attention_similarity:", attention_similarity[0, 0, :3, :3]
        )

        ## Step 4: first step prediction
        # Next, we can prompt SAM to generate a mask for the target concept (the dog) in the test image, using the prompts we prepared below. Interestingly here is that the authors add "Target-guided Attention" and "Target-semantic prompting" to the decoder. This allows for more explicit guidance to the cross-attention mechanisms in SAM’s decoder, which concentrates feature aggregation within foreground target regions. This ultimately leads to more accurate masks when personalizing SAM in a training-free manner.

        # prepare test image and prompts for the model
        inputs = self.processor(
            test_image,
            input_points=[topk_xy.tolist()],
            input_labels=[topk_label.tolist()],
            return_tensors="pt",
        ).to(self.device)
        # for k, v in inputs.items():
        # print(k, v.shape)

        # First-step prediction
        with torch.no_grad():
            outputs = self.model(
                input_points=inputs.input_points,
                input_labels=inputs.input_labels,
                image_embeddings=self.test_feat.unsqueeze(0),
                multimask_output=False,
                attention_similarity=attention_similarity,  # Target-guided Attention
                target_embedding=self.target_embedding,  # Target-semantic Prompting
            )
            best_idx = 0

        ## Step 5: 2-step cascaded post-refinement
        # Finally, the authors propose to further improve the segmentation mask using cascaded post-refinement. The initial mask (predicted above) might include some rough edges and isolated noises in the background. The authors propose to iteratively feed the mask back into SAM’s decoder for a two-step post-processing.

        # Cascaded Post-refinement-1
        with torch.no_grad():
            outputs_1 = self.model(
                input_points=inputs.input_points,
                input_labels=inputs.input_labels,
                input_masks=outputs.pred_masks.squeeze(1)[
                    best_idx : best_idx + 1, :, :
                ],
                image_embeddings=self.test_feat.unsqueeze(0),
                multimask_output=True,
            )

        # Cascaded Post-refinement-2
        masks = (
            self.processor.image_processor.post_process_masks(
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

        inputs = self.processor(
            test_image,
            input_points=[topk_xy.tolist()],
            input_labels=[topk_label.tolist()],
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        final_outputs = self.model(
            input_points=inputs.input_points,
            input_labels=inputs.input_labels,
            input_boxes=inputs.input_boxes,
            input_masks=outputs_1.pred_masks.squeeze(1)[
                :, best_idx : best_idx + 1, :, :
            ],
            image_embeddings=self.test_feat.unsqueeze(0),
            multimask_output=True,
        )

        masks = (
            self.processor.image_processor.post_process_masks(
                final_outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
            .squeeze()
            .numpy()
        )

        best_idx = torch.argmax(final_outputs.iou_scores).item()
        best_mask = masks[best_idx]
        return best_mask


# %%
oneshot = OneShotSegmentation()
oneshot.load_model("facebook/sam-vit-huge")
# %%
filename = "/home/garlan/git/streamingwebcam/imgs/frame-0001.jpg"
mask_filename = "/home/garlan/git/streamingwebcam/playground/mask-tensile1.png"
oneshot.open_image_and_mask(filename, mask_filename)
oneshot.show_image_and_mask()
# %%

# %%
oneshot.process_one_shot_img_and_mask()

# %%
# fig, ax = plt.subplots(5,5, figsize=(10,10))
# ax = ax.flatten()
# start = 50
# for i,j in enumerate(range(start, start+25)):
#     ax[i].imshow(oneshot.ref_feat[:,:,j].cpu().numpy())
# %%
test_img_fiie_name = "/home/garlan/git/streamingwebcam/imgs/frame-0002.jpg"

mask = oneshot.process_new_img(test_img_fiie_name)
# %%
print(oneshot.test_feat_reshaped.shape, oneshot.target_feat.shape)
# %%
# show results
test_img_np = np.array(Image.open(test_img_fiie_name).convert("RGB"))
fig, axes = plt.subplots(1, 3)
axes[0].imshow(test_img_np)
axes[1].imshow(mask)
axes[2].imshow(test_img_np)
show_mask(mask, axes[2])
_ = [a.axis("off") for a in axes]
# %%
