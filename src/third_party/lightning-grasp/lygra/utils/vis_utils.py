# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import open3d as o3d
import numpy as np
import trimesh 


def get_box_lineset_visual(box_min, box_max):
    points = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_max[2]],
        [box_min[0], box_max[1], box_max[2]],
    ])

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def make_arrow_lines(start_points, vectors, color=[1, 0, 0]):
    """Create an Open3D LineSet that visualizes 3D arrows."""
    assert start_points.shape == vectors.shape
    end_points = start_points + vectors

    points = np.vstack([start_points, end_points])  # [2*M, 3]
    M = start_points.shape[0]
    lines = [[i, i + M] for i in range(M)]
    colors = [color for _ in range(M)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_grasp_point(grasp_pos, grasp_normal, point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])  # gray

    vector_fields = []
    arrow_lines = make_arrow_lines(grasp_pos, grasp_normal * 0.03)
    vector_fields.append(arrow_lines)

    base = get_frame_visual()
    # Visualize everything
    o3d.visualization.draw_geometries([pcd] + vector_fields + [base])


def get_point_cloud(point_cloud, color=[0.3, 0.3, 0.3]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color(color)  # gray
    return [pcd]


def get_vector_field(vec_origin, vec_direction, color=[1, 0, 0]):
    vector_fields = []
    arrow_lines = make_arrow_lines(vec_origin, vec_direction * 0.005, color)
    vector_fields.append(arrow_lines)
    return vector_fields


def get_vector_fields(vec_origins, vec_directions, colors=[]):
    vector_fields = []
    for i, (vec_origin, vec_direction) in enumerate(zip(vec_origins, vec_directions)):
        if i < len(colors):
            color = colors[i]
        else:
            color = [1, 0, 0]
        arrow_lines = make_arrow_lines(vec_origin, vec_direction, color)
        # print(vec_origin, vec_direction)

        vector_fields.append(arrow_lines)
    return vector_fields


def get_oriented_boxes(
    box_centers: np.ndarray,
    box_xaxis: np.ndarray,
    box_yaxis: np.ndarray,
    box_zaxis: np.ndarray,
    box_widthx: np.ndarray,
    box_widthy: np.ndarray,
    box_widthz: np.ndarray,
    color=(1, 0, 0)
):
    """
    Render oriented bounding boxes in Open3D.

    Args:
        box_centers (np.ndarray): [B, 3] array of box centers.
        box_xaxis (np.ndarray): [B, 3] array of box local x axes (must be normalized).
        box_yaxis (np.ndarray): [B, 3] array of box local y axes (must be normalized).
        box_zaxis (np.ndarray): [B, 3] array of box local z axes (must be normalized).
        box_widthx (np.ndarray): [B,] array of box extents along x axis.
        box_widthy (np.ndarray): [B,] array of box extents along y axis.
        box_widthz (np.ndarray): [B,] array of box extents along z axis.
        color (tuple): RGB tuple for box color (default: red).

    Returns:
        None. Opens an Open3D visualizer window.
    """
    B = box_centers.shape[0]
    boxes = []

    if isinstance(box_widthx, float):
        box_widthx = np.array([box_widthx] * B)
        box_widthy = np.array([box_widthy] * B)
        box_widthz = np.array([box_widthz] * B)
        
    for i in range(B):
        center = box_centers[i]
        R = np.stack([
            box_xaxis[i],
            box_yaxis[i],
            box_zaxis[i]
        ], axis=1)  # Shape: [3, 3]

        extent = np.array([box_widthx[i], box_widthy[i], box_widthz[i]])
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        obb.color = color
        boxes.append(obb)

    print(len(boxes))
    return boxes


def get_frame_visual(rotation=np.eye(3), translation=np.zeros(3)):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    frame.transform(T)
    return frame


def show_visualization(geoms):
    # Visualize everything
    o3d.visualization.draw(geoms)


def visualize_vector_field(vec_origin, vec_direction, color=[1, 0, 0]):
    vector_fields = []
    arrow_lines = make_arrow_lines(vec_origin, vec_direction * 0.03, color)
    vector_fields.append(arrow_lines)

    # Visualize everything
    o3d.visualization.draw_geometries(vector_fields)

def visualize_vector_fields(vec_origins, vec_directions, colors=[]):
    vector_fields = []
    for i, (vec_origin, vec_direction) in enumerate(zip(vec_origins, vec_directions)):
        if i < len(colors):
            color = colors[i]
        else:
            color = [1, 0, 0]
        arrow_lines = make_arrow_lines(vec_origin, vec_direction * 0.001, color)
        vector_fields.append(arrow_lines)

    # Visualize everything
    o3d.visualization.draw_geometries(vector_fields)


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(mesh.vertex_normals)
    else:
        o3d_mesh.compute_vertex_normals()

    return o3d_mesh
