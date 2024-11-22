import torch
from torch import nn

from utils import build_rotation, inverse_sigmoid

class Gaussians(nn.Module):
    def __init__(self, points: torch.Tensor, colors: torch.Tensor) -> None:
        """
        Initialize the 3D Gaussians with their positions, colors, scales, quaternions, and opacity.

        Args:
            points (torch.Tensor): Nx3 tensor of 3D points representing Gaussian centers.
            colors (torch.Tensor): Nx3 tensor of RGB colors for the Gaussians.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize 3D points (Gaussian centers)
        self.points = points.clone().requires_grad_(True).to(self.device).float()
        
        # Initialize colors (normalized to [0, 1])
        self.colors = (colors / 255.0).clone().requires_grad_(True).to(self.device).float()
        
        # Initialize scales (default small values for now, later adjustable)
        self.scales = torch.ones((len(self.points), 3)).to(self.device).float() * 0.001
        # self.initialize_scale()
        
        # Initialize quaternions (identity quaternion for no rotation)
        self.quaternions = torch.zeros((len(self.points), 4)).to(self.device)
        self.quaternions[:, 0] = 1.0  
        
        # Initialize opacity (default high opacity)
        self.opacity = inverse_sigmoid(
            0.9999 * torch.ones((self.points.shape[0], 1), dtype=torch.float)
        ).to(self.device)

    def initialize_scale(self):
        """
        Initialize the scales for each Gaussian based on the average of the three smallest
        nonzero distances to neighbors. This helps adapt the Gaussian size to the scene.
        """
        # Pairwise distances
        diffs = self.points.unsqueeze(0) - self.points.unsqueeze(1)
        distances = torch.linalg.norm(diffs, dim=2)
        
        # Ignore self-distance (diagonal)
        distances.fill_diagonal_(float("inf"))
        
        # Find the mean of the three smallest distances
        nearest_distances = distances.topk(3, largest=False).values
        mean_scale = nearest_distances.mean(dim=1).clamp(min=0.00001)
        
        # Initialize scales
        self.scales *= mean_scale.unsqueeze(1)

    def get_3d_covariance_matrix(self) -> torch.Tensor:
        """
        Compute the 3D covariance matrix for each Gaussian based on its scale and rotation.

        Returns:
            torch.Tensor: Nx3x3 tensor of covariance matrices for all Gaussians.
        """
        # Normalize quaternions
        quaternions = nn.functional.normalize(self.quaternions, p=2, dim=1)
        
        # Build rotation matrices from quaternions
        rotation_matrices = build_rotation(quaternions)
        
        # Create scale matrices
        scale_matrices = torch.zeros((len(self.points), 3, 3)).to(self.device)
        scale_matrices[:, 0, 0] = self.scales[:, 0]
        scale_matrices[:, 1, 1] = self.scales[:, 1]
        scale_matrices[:, 2, 2] = self.scales[:, 2]
        
        # Compute covariance matrices: R * S * (R * S)^T
        scale_rotation_matrix = rotation_matrices @ scale_matrices
        covariance_matrices = scale_rotation_matrix @ scale_rotation_matrix.transpose(1, 2)
        return covariance_matrices
    