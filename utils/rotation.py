import torch


def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to a 3x3 rotation matrix.

    Args:
        rot6d: Tensor of shape (..., 6)

    Returns:
        Tensor of shape (..., 3, 3)
    """
    rot6d = torch.as_tensor(rot6d)
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))
    matrix = torch.stack([x, y, z], dim=-2).transpose(-2, -1)
    return matrix


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D rotation representation.

    Supports:
        - Single matrix: (3, 3)
        - Batch: (..., 3, 3)

    Returns:
        Tensor of shape (..., 6)
    """
    matrix = torch.as_tensor(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid shape {matrix.shape}, expected (..., 3, 3)")
    rot6d = matrix.transpose(-2, -1).reshape(*matrix.shape[:-2], 9)[..., :6]
    return rot6d
