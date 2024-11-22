import os

import matplotlib.pyplot as plt
import pycolmap
import torch

from gaussian_splatting import GaussianScene
from gaussian_splatting import Gaussians


def filter_points3D(reconstruction):
    """
    Filters 3D points based on their track length and extracts their positions and colors.

    Args:
        reconstruction: pycolmap.Reconstruction object.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered 3D points and their colors.
    """
    all_points3d = []
    all_point_colors = []

    for idx, point in enumerate(reconstruction.points3D.values()):
        if point.track.length() >= 2:  # Only keep points observed in at least 2 images
            all_points3d.append(point.xyz)
            all_point_colors.append(point.color)

    return torch.Tensor(all_points3d), torch.Tensor(all_point_colors)

def initialize_scene(colmap_path):
    """
    Initializes the GaussianScene from a COLMAP reconstruction.

    Args:
        colmap_path (str): Path to COLMAP reconstruction output.
        model_path (str): Path to save the point cloud model.

    Returns:
        GaussianScene: Initialized GaussianScene object.
    """
    # Load COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction(colmap_path)

    # Filter points
    points, colors = filter_points3D(reconstruction)

    # Initialize Gaussians
    gaussians = Gaussians(points, colors)

    # Initialize Scene
    return GaussianScene(colmap_path=colmap_path, gaussians=gaussians)

def render_and_save_image(colmap_path: str, image_idx: int, output_path: str = "./output"):
    """
    Renders the image using Gaussian Splatting and saves the output.

    Args:
        colmap_path (str): Path to the COLMAP reconstruction directory.
        image_idx (int): Index of the image to render.
        output_path (str): Directory to save the rendered images.
    """

    os.makedirs(output_path, exist_ok=True)

    print(f"Loading COLMAP data from {colmap_path}...")
    scene = initialize_scene(colmap_path)

    print(f"Rendering image with ID {image_idx}...")
    with torch.no_grad():
        rendered_image = scene.render_image(image_idx=image_idx)

    rendered_image_path = os.path.join(output_path, f"rendered_image_{image_idx}.png")
    plt.imsave(rendered_image_path, rendered_image.cpu().detach().transpose(0, 1) * 255)
    print(f"Rendered image saved to {rendered_image_path}")
