import torch

def in_view_frustum(
    points: torch.Tensor, view_matrix: torch.Tensor, minimum_z: float = 0.2
) -> torch.Tensor:
    """
    Checks if 3D points are within the view frustum of the camera.

    Args:
        points (torch.Tensor): Nx3 tensor of 3D points in world space.
        view_matrix (torch.Tensor): 4x4 world-to-view transformation matrix.
        minimum_z (float): Minimum z-distance for a point to be considered in the frustum.

    Returns:
        torch.Tensor: Boolean tensor (N) indicating whether each point is in the frustum.
    """
    homogeneous = torch.ones((points.shape[0], 4), device=points.device)
    homogeneous[:, :3] = points
    projected_points = homogeneous @ view_matrix
    z_component = projected_points[:, 2]
    truth = z_component >= minimum_z
    return truth

def ndc2Pix(ndc_coords: torch.Tensor, dimensions: torch.Tensor) -> torch.Tensor:
    """
    Converts normalized device coordinates (NDC) to pixel-space coordinates.

    Args:
        ndc_coords (torch.Tensor): Nx2 tensor of NDC points.
        dimensions (torch.Tensor): Tensor with [width, height] of the image.

    Returns:
        torch.Tensor: Nx2 tensor of pixel-space points.
    """
    return (ndc_coords + 1) * (dimensions - 1) * 0.5