from tqdm import tqdm

import torch
from torch import nn

from gaussian_splatting import GaussianImage  
from gaussian_splatting import Gaussians

from utils.schema import PreprocessedScene

from utils import (
    read_image_file, 
    read_camera_file,
    in_view_frustum,
    ndc2Pix,
    compute_2d_covariance,
    compute_inverted_covariance,
    compute_extent_and_radius,
    compute_gaussian_weight
)

class GaussianScene(nn.Module):
    def __init__(
            self, 
            colmap_path: str, 
            gaussians: Gaussians,
        ) -> None:
        """
        Initializes the GaussianScene with COLMAP camera and image data.

        Args:
            colmap_path (str): Path to the COLMAP output directory.
            gaussians: An instance of the Gaussians class.
        """
        super().__init__()
        self.gaussians = gaussians
        self.cameras = self.load_cameras(colmap_path)
        self.images = self.load_images(colmap_path)

    def load_cameras(self, colmap_path: str):
        """Loads camera data from COLMAP and initializes GaussianImage instances."""
        camera_dict = read_camera_file(colmap_path)
        return camera_dict

    def load_images(self, colmap_path: str):
        image_dict = read_image_file(colmap_path)
        images = {}
        for idx, image in image_dict.items():
            camera = self.cameras[image.camera_id]
            images[idx] = GaussianImage(camera=camera, image=image)
        return images
    
    def preprocess(self, image_idx: int) -> PreprocessedScene:
        """
        Preprocesses Gaussians for rendering in the given image.

        Args:
            image_idx (int): Index of the image to preprocess.

        Returns:
            PreprocessedScene: Contains preprocessed data for rendering.
        """
        image = self.images[image_idx]

        # Cull points outside the frustum
        in_view = in_view_frustum(
            points=self.gaussians.points,
            view_matrix=image.world_view_transform,
        )
        points = self.gaussians.points[in_view]
        covariance_3d = self.gaussians.get_3d_covariance_matrix()[in_view]

        # Transform points into view space
        points_homogeneous = torch.cat(
            [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
        )
        points_view = (
            points_homogeneous
            @ image.world_view_transform.to(points_homogeneous.device)
        )[:, :3] # Transform to camera space

        # Transform points into NDC (normalized device coordinates)
        points_ndc = points_homogeneous @ image.full_proj_transform.to(
            points_homogeneous.device
        )
        points_ndc = points_ndc[:, :3] / points_ndc[:, 3].unsqueeze(1)

        # Convert NDC to pixel coordinates
        points_xy = points_ndc[:, :2]
        points_xy[:, 0] = ndc2Pix(points_xy[:, 0], image.width)
        points_xy[:, 1] = ndc2Pix(points_xy[:, 1], image.height)

        # Compute 2D covariance matrices
        covariance_2d = compute_2d_covariance(
            points=points,
            extrinsic_matrix=image.world_view_transform,
            covariance_3d=covariance_3d,
            tan_fovX=image.tan_fovX,
            tan_fovY=image.tan_fovY,
            focal_x=image.f_x,
            focal_y=image.f_y,
        )

        # Compute inverse covariance and radii
        inverse_covariance = compute_inverted_covariance(covariance_2d)
        radius = compute_extent_and_radius(covariance_2d)

        # Compute min/max bounds for each Gaussian
        min_x = torch.floor(points_xy[:, 0] - radius)
        min_y = torch.floor(points_xy[:, 1] - radius)
        max_x = torch.ceil(points_xy[:, 0] + radius)
        max_y = torch.ceil(points_xy[:, 1] + radius)

        # Sort points by depth (front to back)
        colors = self.gaussians.colors[in_view]
        opacity = self.gaussians.opacity[in_view]

        indices_by_depth = torch.argsort(points_view[:, 2])
        points_view = points_view[indices_by_depth]
        colors = colors[indices_by_depth]
        opacity = opacity[indices_by_depth]
        points_xy = points_xy[indices_by_depth]
        covariance_2d = covariance_2d[indices_by_depth]
        inverse_covariance = inverse_covariance[indices_by_depth]
        radius = radius[indices_by_depth]
        min_x = min_x[indices_by_depth]
        min_y = min_y[indices_by_depth]
        max_x = max_x[indices_by_depth]
        max_y = max_y[indices_by_depth]

        # Return preprocessed data
        return PreprocessedScene(
            points=points_xy,
            colors=colors,
            covariance_2d=covariance_2d,
            depths=points_view[:, 2],
            inverse_covariance_2d=inverse_covariance,
            radius=radius,
            points_xy=points_xy,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            sigmoid_opacity=torch.sigmoid(opacity),
        )

    def render_pixel(
        self,
        pixel_coords: torch.Tensor,
        points_in_tile_mean: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        inverse_covariance: torch.Tensor,
        min_weight: float = 0.000001,
    ) -> torch.Tensor:
        total_weight = torch.ones(1).to(points_in_tile_mean.device)
        pixel_color = torch.zeros((1, 1, 3)).to(points_in_tile_mean.device)
        for point_idx in range(points_in_tile_mean.shape[0]):
            point = points_in_tile_mean[point_idx, :].view(1, 2)
            weight = compute_gaussian_weight(
                pixel_coord=pixel_coords,
                point_mean=point,
                inverse_covariance=inverse_covariance[point_idx],
            )
            alpha = weight * torch.sigmoid(opacities[point_idx])
            test_weight = total_weight * (1 - alpha)
            if test_weight < min_weight:
                return pixel_color
            pixel_color += total_weight * alpha * colors[point_idx]
            total_weight = test_weight
        # in case we never reach saturation
        return pixel_color

    def render_tile(
        self,
        x_min: int,
        y_min: int,
        points_in_tile_mean: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        inverse_covariance: torch.Tensor,
        tile_size: int = 16,
    ) -> torch.Tensor:
        """Points in tile should be arranged in order of depth"""

        tile = torch.zeros((tile_size, tile_size, 3))

        for pixel_x in range(x_min, x_min + tile_size):
            for pixel_y in range(y_min, y_min + tile_size):
                tile[pixel_x % tile_size, pixel_y % tile_size] = self.render_pixel(
                    pixel_coords=torch.Tensor([pixel_x, pixel_y])
                    .view(1, 2)
                    .to(points_in_tile_mean.device),
                    points_in_tile_mean=points_in_tile_mean,
                    colors=colors,
                    opacities=opacities,
                    inverse_covariance=inverse_covariance,
                )
        return tile

    def render_image(self, image_idx: int, tile_size: int = 16) -> torch.Tensor:
        """For each tile have to check if the point is in the tile"""
        preprocessed_scene = self.preprocess(image_idx)
        height = int(self.images[image_idx].height.item())
        width = int(self.images[image_idx].width.item())

        image = torch.zeros((width, height, 3))

        for x_min in tqdm(range(0, width - tile_size, tile_size)):
            x_in_tile = (preprocessed_scene.min_x <= x_min + tile_size) & (
                preprocessed_scene.max_x >= x_min
            )
            if x_in_tile.sum() == 0:
                continue
            for y_min in range(0, height - tile_size, tile_size):
                y_in_tile = (preprocessed_scene.min_y <= y_min + tile_size) & (
                    preprocessed_scene.max_y >= y_min
                )
                points_in_tile = x_in_tile & y_in_tile
                if points_in_tile.sum() == 0:
                    continue
                points_in_tile_mean = preprocessed_scene.points[points_in_tile]
                colors_in_tile = preprocessed_scene.colors[points_in_tile]
                opacities_in_tile = preprocessed_scene.sigmoid_opacity[points_in_tile]
                inverse_covariance_in_tile = preprocessed_scene.inverse_covariance_2d[
                    points_in_tile
                ]
                image[x_min : x_min + tile_size, y_min : y_min + tile_size] = (
                    self.render_tile(
                        x_min=x_min,
                        y_min=y_min,
                        points_in_tile_mean=points_in_tile_mean,
                        colors=colors_in_tile,
                        opacities=opacities_in_tile,
                        inverse_covariance=inverse_covariance_in_tile,
                        tile_size=tile_size,
                    )
                )
        return image

