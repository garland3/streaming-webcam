from transformers import AutoProcessor, SamModel
import cv2
import numpy as np
import torch


from transformers import AutoProcessor, SamModel
import cv2
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
import torch
from typing import Tuple


class MaskGenerator:
    def __init__(self, model_path, processor_path, long_side_length=1024):
        self.model = SamModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.long_side_length = long_side_length

    # @staticmethod
    def get_preprocess_shape(self, oldh, oldw):
        """
        Compute the output size given input size and target long side length.
        """
        scale = self.long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale), int(oldw * scale)
        return newh, neww

    def preprocess(
        self,
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

    def prepare_mask(
        self,
        image,
    ):
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1])
        mask = np.array(resize(to_pil_image(image), target_size))

        input_mask = torch.as_tensor(mask)
        input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]

        input_mask = self.preprocess(input_mask)

        return input_mask

    def generate_mask(self, image):
        """
        Generate mask for a given image.
        """
        preprocessed_image = self.preprocess(image)
        inputs = self.processor(images=preprocessed_image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        mask = self.prepare_mask(logits)
        return mask
