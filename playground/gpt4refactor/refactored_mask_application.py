import cv2
from refactored_mask_generator import MaskGenerator


class MaskApplication:
    def __init__(self, model_path, processor_path):
        self.mask_generator = MaskGenerator(model_path, processor_path)

    def load_images(self, image_paths):
        """
        Load multiple images from given paths.
        """
        images = [cv2.imread(path) for path in image_paths]
        return images

    def generate_and_save_masks(self, image_paths, save_dir):
        """
        Generate masks for multiple images and save them.
        """
        images = self.load_images(image_paths)
        for i, image in enumerate(images):
            mask = self.mask_generator.generate_mask(image)
            save_path = f"{save_dir}/mask_{i}.png"
            cv2.imwrite(save_path, mask)


# Usage example:
# mask_app = MaskApplication("model_path", "processor_path")
# image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
# mask_app.generate_and_save_masks(image_paths, "save_directory")
