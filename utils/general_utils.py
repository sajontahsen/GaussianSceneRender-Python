import torch

def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse sigmoid (logit) of a tensor.
    
    Args:
        x (torch.Tensor): Tensor with values in (0, 1).
    
    Returns:
        torch.Tensor: Logit of the input tensor.
    """
    return torch.log(x / (1 - x))

def build_rotation(r: torch.Tensor) -> torch.Tensor:

    """
    Build 3x3 rotation matrices from normalized quaternions.

    Args:
        r (torch.Tensor): Nx4 tensor of normalized quaternions.

    Returns:
        torch.Tensor: Nx3x3 tensor of rotation matrices.
    """

    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R