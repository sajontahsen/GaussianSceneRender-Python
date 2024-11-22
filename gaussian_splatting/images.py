from typing import Tuple

import torch

from utils import build_rotation, in_view_frustum, ndc2Pix

class GaussianImage(torch.nn.Module):
    def __init__(self, camera, image) -> None:
        """
        Represents a camera and its ability to project points onto the image plane.

        Args:
            camera (dict): Intrinsic camera parameters (focal length, principal points, resolution).
            image (dict): Extrinsic image parameters (rotation, translation, image name).
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Camera intrinsics
        self.f_x = torch.tensor([camera.params[0]]).to(self.device)
        self.f_y = torch.tensor([camera.params[1]]).to(self.device)
        self.c_x = torch.tensor([camera.params[2]]).to(self.device)
        self.c_y = torch.tensor([camera.params[3]]).to(self.device)
        self.intrinsic_matrix = self.get_intrinsic_matrix(
            f_x=self.f_x, f_y=self.f_y, c_x=self.c_x, c_y=self.c_y
        ).to(self.device)
        self.height = torch.tensor([camera.height]).to(self.device)
        self.width = torch.tensor([camera.width]).to(self.device)

        # Camera extrinsics
        self.R = build_rotation(torch.tensor(image.qvec, dtype=torch.float32).unsqueeze(0).to(self.device))
        self.T = torch.tensor(image.tvec, dtype=torch.float32).to(self.device)
        self.extrinsic_matrix = self.get_extrinsic_matrix(self.R[0], self.T).to(self.device)

        # Field of view and perspective projection
        self.fovX = self.focal2fov(self.f_x, self.width).to(self.device)
        self.fovY = self.focal2fov(self.f_y, self.height).to(self.device)
        self.tan_fovX = torch.tan(self.fovX / 2).to(self.device)
        self.tan_fovY = torch.tan(self.fovY / 2).to(self.device)
        self.zfar = torch.tensor([100.0]).to(self.device)
        self.znear = torch.tensor([0.001]).to(self.device)
        self.projection_matrix = self.getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fovX, fovY=self.fovY
        ).transpose(0, 1).to(self.device)

        # Combined transformation matrices
        self.world_view_transform = self.getWorld2View(R=self.R[0], T=self.T).transpose(0, 1).to(self.device)
        self.full_proj_transform = (
            (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)))
            .squeeze(0)
            .to(self.device)
        )
        # Camera properties
        self.name = image.name
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.device)
        self.projection = (self.intrinsic_matrix @ self.extrinsic_matrix).to(self.device)


    @staticmethod
    def get_intrinsic_matrix(f_x, f_y, c_x, c_y):
        """
        Get the homogenous intrinsic matrix for the camera

        Args:
            f_x: focal length in x
            f_y: focal length in y
            c_x: principal point in x
            c_y: principal point in y
        """
        
        return torch.tensor(
            [
                [f_x, 0, c_x, 0],
                [0, f_y, c_y, 0],
                [0, 0, 1, 0]
            ],
            dtype=torch.float32
        )


    @staticmethod
    def get_extrinsic_matrix(R, T):
        """
        Get the homogenous extrinsic matrix for the camera

        Args:
            R: 3x3 rotation matrix
            T: 3x1 translation vector
        """
        Rt = torch.zeros((4, 4), dtype=torch.float32)
        Rt[:3,:3] = R
        Rt[:3, 3]= T
        Rt[3, 3] = 1.0
        return Rt    
    

    @staticmethod
    def getWorld2View(R, T) -> torch.Tensor:
        """Computes the world-to-view transformation."""
        Rt = torch.zeros((4, 4))

        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        return Rt.float()

    @staticmethod
    def getProjectionMatrix(
        znear: torch.Tensor, zfar: torch.Tensor, fovX: torch.Tensor, fovY: torch.Tensor
    ) -> torch.Tensor:
        """
        znear: near plane set by user
        zfar: far plane set by user
        fovX: field of view in x, calculated from the focal length
        fovY: field of view in y, calculated from the focal length


        This is from the original repo. I just changed math.tan to torch.tan
        """
        tanHalfFovY = torch.tan((fovY / 2))
        tanHalfFovX = torch.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    @staticmethod
    def focal2fov(focal, dim) ->  torch.Tensor:
        """Converts focal length to field of view."""
        return torch.tensor([2 * torch.atan(dim / (2 * focal))])
    
    def project_point_to_camera_perspective_projection(
            self, points: torch.Tensor, colors: torch.Tensor = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects 3D points onto the 2D image plane and maps them to pixel coordinates.

        Args:
            points (torch.Tensor): Nx3 tensor of 3D points.

        Returns:
            torch.Tensor: Nx3 tensor of projected points in pixel coordinates (x, y, z).
        """
        # Frustum culling
        in_frustum_truth = in_view_frustum(
            points=points,
            view_matrix=self.world2view,
        )
        points = points[in_frustum_truth]

        # Convert to homogeneous coordinates
        points = torch.cat(
            [points, torch.ones(points.shape[0], 1, device=self.device)], dim=1
        )

        # Apply full projection transform
        four_dim_points = points @ self.full_proj_transform  # Nx4

        # Normalize by w-component
        three_dim_points = four_dim_points[:, :3] / four_dim_points[:, 3].unsqueeze(1)

        # Convert NDC to pixel coordinates
        three_dim_points[:, 0] = ndc2Pix(three_dim_points[:, 0], self.width)
        three_dim_points[:, 1] = ndc2Pix(three_dim_points[:, 1], self.height)

        if colors:
            return three_dim_points, colors[in_frustum_truth]
        
        return three_dim_points

