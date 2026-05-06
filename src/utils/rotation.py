import torch
from scipy.spatial.transform import Rotation as R


def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)


def rot6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to a 3x3 rotation matrix.

    Args:
        rot_6d: Tensor of shape (..., 6)

    Returns:
        Tensor of shape (..., 3, 3)
    """
    rot_6d = torch.as_tensor(rot_6d)
    x = normalize(rot_6d[..., 0:3])
    y = normalize(rot_6d[..., 3:6])
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

    Args:
        - matrix: Tensor of shape (..., 3, 3)

    Returns:
        Tensor of shape (..., 6)
    """
    matrix = torch.as_tensor(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid shape {matrix.shape}, expected (..., 3, 3)")
    rot6d = matrix.transpose(-2, -1).reshape(*matrix.shape[:-2], 9)[..., :6]
    return rot6d


def q_euler_to_q_6d(q_euler: torch.Tensor) -> torch.Tensor:
    """
    Convert joint angle representation from Euler (XYZ) angles to 6D rotation representation.

    Args:
        q_euler = (..., D), where D >= 6
            - [..., 0:3]: position (x, y, z)
            - [..., 3:6]: Euler angles in XYZ order (radians)
            - [..., 6:]: hand joint angles (length = 22 for canonical hand, or DoF for original hand)

    Returns:
        Tensor of shape (..., D + 3)
            - [..., 0:3]: position (x, y, z)
            - [..., 3:9]: 6D rotation representation
            - [..., 9:]: hand joint angles (length = 22 for canonical hand, or DoF for original hand)
    """
    q_euler = torch.as_tensor(q_euler)
    if q_euler.shape[-1] < 6:
        raise ValueError(f"Invalid shape {q_euler.shape}, expected last dim >= 6")
    pos = q_euler[..., :3]
    euler = q_euler[..., 3:6]
    joint = q_euler[..., 6:]

    euler_np = euler.detach().cpu().numpy().reshape(-1, 3)
    rot_matrix_np = R.from_euler("XYZ", euler_np).as_matrix()  # (N, 3, 3)
    rot_matrix = torch.from_numpy(rot_matrix_np).to(
        dtype=torch.float32,
        device=q_euler.device
    ).reshape(*euler.shape[:-1], 3, 3)

    rot_6d = matrix_to_rot6d(rot_matrix)  # (..., 6)
    q_6d = torch.cat([pos, rot_6d, joint], dim=-1)
    return q_6d


def q_6d_to_q_euler(q_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert joint angle representation from 6D rotation representation to Euler (XYZ) angles.

    Args:
        q_6d = (..., D), where D >= 9
            - [..., 0:3]: position (x, y, z)
            - [..., 3:9]: 6D rotation representation
            - [..., 9:]: hand joint angles (length = 22 for canonical hand, or DoF for original hand)

    Returns:
        Tensor of shape (..., D - 3)
            - [..., 0:3]: position (x, y, z)
            - [..., 3:6]: Euler angles in XYZ order (radians)
            - [..., 6:]: hand joint angles (length = 22 for canonical hand, or DoF for original hand)
    """
    q_6d = torch.as_tensor(q_6d)
    if q_6d.shape[-1] < 9:
        raise ValueError(f"Invalid shape {q_6d.shape}, expected last dim >= 9")
    pos = q_6d[..., :3]
    rot_6d = q_6d[..., 3:9]
    joint = q_6d[..., 9:]

    rot_matrix = rot6d_to_matrix(rot_6d)  # (..., 3, 3)
    rot_matrix_np = rot_matrix.detach().cpu().numpy().reshape(-1, 3, 3)
    euler_np = R.from_matrix(rot_matrix_np).as_euler("XYZ")  # (N, 3)
    euler = torch.from_numpy(euler_np).to(
        dtype=torch.float32,
        device=q_6d.device
    ).reshape(*rot_6d.shape[:-1], 3)

    q_euler = torch.cat([pos, euler, joint], dim=-1)
    return q_euler
