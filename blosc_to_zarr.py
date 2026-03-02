# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import blosc
import zarr
import numpy as np
from tqdm import tqdm
import cv2


import pickle
import torch
# import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time
import socket



# def farthest_point_sampling(points, num_points=1024, use_cuda=True):
#     K = [num_points]
#     if use_cuda:
#         points = torch.from_numpy(points).cuda()
#         sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
#         sampled_points = sampled_points.squeeze(0)
#         sampled_points = sampled_points.cpu().numpy()
#     else:
#         points = torch.from_numpy(points)
#         sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
#         sampled_points = sampled_points.squeeze(0)
#         sampled_points = sampled_points.numpy()

#     return sampled_points, indices

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



def preprocess_point_cloud(points, num_points=1024, use_cuda=True):

    WORK_SPACE = [
        [-0.6, 1.3],   # X 
        [-0.3, 0.5],   # Y
        [0.2, 1.8]     # Z
    ]

    # 只保留工作空间内的点
    mask = (
        (points[:, 0] > WORK_SPACE[0][0]) & (points[:, 0] < WORK_SPACE[0][1]) &
        (points[:, 1] > WORK_SPACE[1][0]) & (points[:, 1] < WORK_SPACE[1][1]) &
        (points[:, 2] > WORK_SPACE[2][0]) & (points[:, 2] < WORK_SPACE[2][1])
    )
    points = points[mask]

    points_xyz = points[:, :3]

    points_xyz, _ = farthest_point_sampling(points_xyz, num_points=num_points, use_cuda=True)

    return points_xyz


# 1️⃣ 读取 blosc
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

    return obs, action, timestamps


# 2️⃣ depth → 点云
def depth_to_cloud(depth, fx, fy, cx, cy, scale):

    depth = depth.astype(np.float32) * scale
    H, W = depth.shape

    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    cloud = np.stack([X, Y, Z], axis=-1)  # (640,480,3)
    cloud = cloud.reshape(-1, 3)          # (640*480,3)

    return cloud.astype(np.float32)

# 3️⃣ 相机参数
cam_params = {
    "cam_head_depth": {
        "fx": 393.23 ,
        "fy": 393.23,
        "cx": 317.85,
        "cy": 243.23,
        "scale": 0.001
    },
    "cam_right_depth": {
        "fx": 389.05,
        "fy": 389.05,
        "cx": 314.73,
        "cy": 236.93,
        "scale": 0.001
    }
}

INPUT_DIR = "aoliao_dataset"
save_data_path = "aoliao_dataset_processed.zarr"

# ==========================================================
# 4️⃣ 数据容器
# ==========================================================
state_list, pose_list = [], []
action_list, action_pose_list = [], []
rgb_head_list, rgb_right_list = [], []
depth_head_list, depth_right_list = [], []
cloud_head_list, cloud_right_list = [], []
episode_ends = []

frame_count = 0  # 累计帧数
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.blosc")))

# 5️⃣ 读取数据
for file in files:

    print("Processing:", file)
    obs, action, _ = load_blosc_file(file)
    episode_length = len(obs)

    for i, sample in enumerate(tqdm(obs)):

        # ===== state =====
        q = sample["q"].astype(np.float32)
        a = action[i].astype(np.float32)

        # state / pose
        state_list.append(q[:8])
        pose_list.append(q[8:14])

        # action / action_pose
        action_list.append(a[:8])
        action_pose_list.append(a[8:14])

        # RGB
        head_rgb = cv2.resize(np.squeeze(sample["cam_head_rgb"]), (84, 84))
        right_rgb = cv2.resize(np.squeeze(sample["cam_right_rgb"]), (84, 84))
        rgb_head_list.append(head_rgb.astype(np.uint8))
        rgb_right_list.append(right_rgb.astype(np.uint8))

        # Depth
        head_depth = cv2.resize(np.squeeze(sample["cam_head_depth"]), (84, 84), interpolation=cv2.INTER_NEAREST)
        right_depth = cv2.resize(np.squeeze(sample["cam_right_depth"]), (84, 84), interpolation=cv2.INTER_NEAREST)
        depth_head_list.append(head_depth.astype(np.float32))
        depth_right_list.append(right_depth.astype(np.float32))

        # ====== 点云 =====
        # 点云 -> 下采样
        cloud_head = preprocess_point_cloud(
            depth_to_cloud(np.squeeze(sample["cam_head_depth"]), **cam_params["cam_head_depth"]),
            num_points=1024, use_cuda=False
        )
        cloud_right = preprocess_point_cloud(
            depth_to_cloud(np.squeeze(sample["cam_right_depth"]), **cam_params["cam_right_depth"]),
            num_points=1024, use_cuda=False
        )
        cloud_head_list.append(cloud_head)
        cloud_right_list.append(cloud_right)

    # 记录 episode 结束位置
    frame_count += episode_length
    episode_ends.append(frame_count)

# 6️⃣ 转 numpy

state_arrays = np.stack(state_list, axis=0)
pose_arrays = np.stack(pose_list, axis=0)
action_arrays = np.stack(action_list, axis=0)
action_pose_arrays = np.stack(action_pose_list, axis=0)
rgb_head_arrays = np.stack(rgb_head_list, axis=0)
rgb_right_arrays = np.stack(rgb_right_list, axis=0)
depth_head_arrays = np.stack(depth_head_list, axis=0)
depth_right_arrays = np.stack(depth_right_list, axis=0)
cloud_head_arrays = np.stack(cloud_head_list, axis=0)  # (N,1024,3)
cloud_right_arrays = np.stack(cloud_right_list, axis=0)  # (N,1024,3)
episode_ends_arrays = np.array(episode_ends, dtype=np.int64)

# 图像通道放最后
if rgb_head_arrays.shape[1] == 3:
    rgb_head_arrays = np.transpose(rgb_head_arrays, (0,2,3,1))
if rgb_right_arrays.shape[1] == 3:
    rgb_right_arrays = np.transpose(rgb_right_arrays, (0,2,3,1))

# 7️⃣ 保存到 zarr
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# ===== chunk size =====
img_head_chunk = (100, rgb_head_arrays.shape[1], rgb_head_arrays.shape[2], rgb_head_arrays.shape[3])
img_right_chunk = (100, rgb_right_arrays.shape[1], rgb_right_arrays.shape[2], rgb_right_arrays.shape[3])
point_cloud_chunk = (100, cloud_head_arrays.shape[1], cloud_head_arrays.shape[2])
depth_head_chunk = (100, depth_head_arrays.shape[1], depth_head_arrays.shape[2])
depth_right_chunk = (100, depth_right_arrays.shape[1], depth_right_arrays.shape[2])
action_chunk = (100, action_arrays.shape[1])
state_chunk = (100, state_arrays.shape[1])


# ===== 保存 data =====
# zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

zarr_data.create_dataset('cam_head_rgb', data=rgb_head_arrays, chunks=img_head_chunk, dtype='uint8', overwrite=True,compressor=compressor)
zarr_data.create_dataset('cam_right_rgb', data=rgb_right_arrays, chunks=img_right_chunk, dtype='uint8', overwrite=True,compressor=compressor)

zarr_data.create_dataset('cloud_cam_head', data=cloud_head_arrays, chunks=point_cloud_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('cloud_cam_right', data=cloud_right_arrays, chunks=point_cloud_chunk, dtype='float32', overwrite=True,compressor=compressor)

zarr_data.create_dataset('cam_head_depth', data=depth_head_arrays, chunks=depth_head_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('cam_right_depth', data=depth_right_arrays, chunks=depth_right_chunk, dtype='float32', overwrite=True,compressor=compressor)

zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('pose', data=pose_arrays, chunks=state_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('action_pose', data=action_pose_arrays, chunks=action_chunk, dtype='float32', overwrite=True,compressor=compressor)

# ===== 保存 meta =====
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True,compressor=compressor)
# ==========================================================
# 8️⃣ 打印信息

cprint(f'cam_head_rgb shape: {rgb_head_arrays.shape}, range: [{np.min(rgb_head_arrays)}, {np.max(rgb_head_arrays)}]', 'green')
cprint(f'cam_right_rgb shape: {rgb_right_arrays.shape}, range: [{np.min(rgb_right_arrays)}, {np.max(rgb_right_arrays)}]', 'green')
cprint(f'cloud_cam_head shape: {cloud_head_arrays.shape}, range: [{np.min(cloud_head_arrays)}, {np.max(cloud_head_arrays)}]', 'green')
cprint(f'cloud_cam_right shape: {cloud_right_arrays.shape}, range: [{np.min(cloud_right_arrays)}, {np.max(cloud_right_arrays)}]', 'green')
cprint(f'cam_head_depth shape: {depth_head_arrays.shape}, range: [{np.min(depth_head_arrays)}, {np.max(depth_head_arrays)}]', 'green')
cprint(f'cam_right_depth shape: {depth_right_arrays.shape}, range: [{np.min(depth_right_arrays)}, {np.max(depth_right_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'pose shape: {pose_arrays.shape}, range: [{np.min(pose_arrays)}, {np.max(pose_arrays)}]', 'green')
cprint(f'action_pose shape: {action_pose_arrays.shape}, range: [{np.min(action_pose_arrays)}, {np.max(action_pose_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

print("\n✅ Conversion Finished Successfully!")
