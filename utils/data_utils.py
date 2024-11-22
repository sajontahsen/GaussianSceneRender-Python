import os
from typing import Dict

from .read_write_model import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)

def read_camera_file(colmap_path: str) -> Dict:
    binary_path = os.path.join(colmap_path, "cameras.bin")
    text_path = os.path.join(colmap_path, "cameras.txt")
    if os.path.exists(binary_path):
        cameras = read_cameras_binary(binary_path)
    elif os.path.exists(text_path):
        cameras = read_cameras_text(text_path)
    else:
        raise ValueError
    return cameras


def read_image_file(colmap_path: str) -> Dict:
    binary_path = os.path.join(colmap_path, "images.bin")
    text_path = os.path.join(colmap_path, "images.txt")
    if os.path.exists(binary_path):
        images = read_images_binary(binary_path)
    elif os.path.exists(text_path):
        images = read_images_text(text_path)
    else:
        raise ValueError
    return images

