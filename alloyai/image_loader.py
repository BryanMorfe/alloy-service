import PIL.Image
import os
from typing import Union
from diffusers.utils.loading_utils import load_image

class ImageLoader:

    @staticmethod
    def load(image_path: Union[str, os.PathLike]) -> PIL.Image.Image:
        """
        Loads an image from a given path or URL.
        """
        return load_image(image_path)
