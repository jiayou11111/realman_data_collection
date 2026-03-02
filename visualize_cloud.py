#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blosc    
import numpy as np
import open3d as o3d

import pickle
import torch
# import pytorch3d.ops as torch3d_ops
# import torchvision
from termcolor import cprint
import re
import time
import socket


BLOSC_PATH = "dataset/demo0010.blosc"

# ===============================
# 1️⃣ 读取 blosc
# ===============================
def unpack_one(raw, offset):
    nbytes, cbytes, blocksize = blosc.get_cbuffer_sizes(raw[offset:])
    buf = raw[offset: offset + cbytes]
    arr = blosc.unpack_array(buf)
    return arr, offset + cbytes

def load_blosc_file(path):
    with open(path, "rb") as f:
        raw = f.read()

    offset = 0
    obs, offset = unpack_one(raw, offset)
    action, offset = unpack_one(raw, offset)
    timestamps, offset = unpack_one(raw, offset)

    return obs

obs = load_blosc_file(BLOSC_PATH)
num_frames = len(obs)
print("Total frames:", num_frames)

# ===============================
# 2️⃣ 相机参数 (640×480 原始参数)
# ===============================
fx = 393.23
fy = 393.23
cx = 317.85
cy = 243.23
scale = 0.001

# ===============================
# 3️⃣ depth → cloud
# ===============================
# WORK_SPACE = [
#     [-0.6, 1.3],   # X 
#     [-0.5, 0.5],   # Y
#     [0.2, 1.8]     # Z
# ]
WORK_SPACE = [
    [-0.6, 1.3],   # X 
    [-0.15, 0.5],   # Y
    [0.2, 1.8]     # Z
]

def depth_to_cloud(depth):
    depth = depth.astype(np.float32) * scale
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    cloud = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    mask = np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > 0)
    cloud = cloud[mask]
    return cloud


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    """
    points: numpy array (N, 3)
    return:
        sampled_points: (num_points, 3)
        indices: (num_points,)
    """

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    pts = torch.from_numpy(points).float().to(device)   # (N, 3)
    N = pts.shape[0]

    # 存储采样索引
    indices = torch.zeros(num_points, dtype=torch.long, device=device)

    # 存储每个点到当前已选点集的最小距离
    distances = torch.ones(N, device=device) * 1e10

    # 随机选择第一个点
    farthest = torch.randint(0, N, (1,), device=device)
    
    for i in range(num_points):
        indices[i] = farthest
        centroid = pts[farthest, :].view(1, 3)

        # 计算到新加入点的距离
        dist = torch.sum((pts - centroid) ** 2, dim=1)

        # 更新最小距离
        mask = dist < distances
        distances[mask] = dist[mask]

        # 选择最远点
        farthest = torch.argmax(distances)

    sampled_points = pts[indices]

    return sampled_points.cpu().numpy(), indices.cpu().numpy()

def preprocess_point_cloud(points, num_points=1024):
    mask = (
        (points[:, 0] > WORK_SPACE[0][0]) & (points[:, 0] < WORK_SPACE[0][1]) &
        (points[:, 1] > WORK_SPACE[1][0]) & (points[:, 1] < WORK_SPACE[1][1]) &
        (points[:, 2] > WORK_SPACE[2][0]) & (points[:, 2] < WORK_SPACE[2][1])
    )
    points_crop = points[mask]
    if len(points_crop) == 0:
        print("failed to crop point cloud")
        return np.zeros((num_points, 3), dtype=np.float32)
        
    return farthest_point_sampling(points_crop, num_points=num_points, use_cuda=True)

# 4️⃣ Open3D 可视化器
current_frame = 0

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Cloud Viewer", width=1200, height=900)

pcd_raw = o3d.geometry.PointCloud()
pcd_crop = o3d.geometry.PointCloud()

pcd_raw.paint_uniform_color([0.5, 0.5, 0.5])    # 灰色
pcd_crop.paint_uniform_color([1.0, 0, 0])       # 红色

vis.add_geometry(pcd_raw)
vis.add_geometry(pcd_crop)

render_option = vis.get_render_option()
render_option.point_size = 2.0
render_option.background_color = np.array([0, 0, 0])
render_option.light_on = True

# 5️⃣ 更新函数
def update_frame(frame_id):
    global current_frame
    frame_id = max(0, min(frame_id, num_frames - 1))
    current_frame = frame_id

    print("\n========== Frame:", current_frame, "==========")
    sample = obs[current_frame]
    # depth = np.squeeze(sample["cam_head_depth"])  # 640×480
    depth = np.squeeze(sample["cam_right_depth"])  # 640×480

    cloud_raw = depth_to_cloud(depth)
    cloud_crop,_ = preprocess_point_cloud(cloud_raw, num_points=1024)

    # print("Raw cloud shape:", cloud_raw)
    # print("Crop cloud shape:", cloud_crop.shape)

    cloud_crop = cloud_crop.astype(np.float64) 

    # pcd_raw.points = o3d.utility.Vector3dVector(cloud_raw.astype(np.float64))
    pcd_crop.points = o3d.utility.Vector3dVector(cloud_crop)

    # pcd_raw.points = o3d.utility.Vector3dVector(cloud_raw)
    pcd_crop.points = o3d.utility.Vector3dVector(cloud_crop)

    # vis.update_geometry(pcd_raw)
    vis.update_geometry(pcd_crop)

    if frame_id == 0:
        vis.reset_view_point(True)

    vis.poll_events()
    vis.update_renderer()

# 6️⃣ 键盘控制
def next_frame(vis):
    update_frame(current_frame + 1)
    return False

def prev_frame(vis):
    update_frame(current_frame - 1)
    return False

def quit_vis(vis):
    vis.close()
    return False

vis.register_key_callback(ord("N"), next_frame)
vis.register_key_callback(ord("B"), prev_frame)
vis.register_key_callback(ord("Q"), quit_vis)

# 初始化
update_frame(0)

vis.run()
vis.destroy_window()