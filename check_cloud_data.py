#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zarr
import numpy as np
import open3d as o3d
import cv2
import time

ZARR_PATH = "init_rectangle_blue_weita.zarr"


# ===============================
# 1. 打开数据
# ===============================
root = zarr.open(ZARR_PATH, mode='r')

data = root['data']
meta = root['meta']

cloud_head = data['cloud_cam_head'][:]   # (N,2024,3)
cloud_right = data['cloud_cam_right'][:] # (N,2048,3)
rgb_head = data['cam_head_rgb'][:]       # (N,H,W,3)
state = data['state'][:]                 # (N,8)
action = data['action'][:]               # (N,8)
episode_ends = meta['episode_ends'][:]

print("=== 数据信息 ===")
print("cloud_head:", cloud_head.shape)
print("cloud_right:", cloud_right.shape)
print("rgb_head:", rgb_head.shape)
print("state:", state.shape)
print("action:", action.shape)
print("episode_ends:", episode_ends)


# 2. 选择 demo
demo_id = 31

start = 0 if demo_id == 0 else episode_ends[demo_id - 1]
end = episode_ends[demo_id]

print(f"\n=== Demo {demo_id} 范围: {start} ~ {end} ===")

# 3. 点云质量检查
print("\n=== 点云质量检查（前10帧）===")
for i in range(start, min(start + 10, end)):
    h = cloud_head[i]
    r = cloud_right[i]

    print(f"\nframe {i-start}")
    print(" head shape:", h.shape, "min/max:", np.min(h), np.max(h))
    print(" right shape:", r.shape, "min/max:", np.min(r), np.max(r))

    print(" head 全0:", np.all(h == 0))
    print(" right 全0:", np.all(r == 0))

    print(" head NaN:", np.isnan(h).any())
    print(" right NaN:", np.isnan(r).any())

# 4. 清理点云（关键！）
def clean_cloud(points):
    # 去掉 NaN
    mask = ~np.isnan(points).any(axis=1)
    points = points[mask]

    # 去掉全0
    mask = ~(np.all(points == 0, axis=1))
    points = points[mask]

    return points

# 5. 可视化函数
def visualize_sequence(cloud_seq, title="cloud"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)

    pcd = o3d.geometry.PointCloud()

    # ✅ 关键：先用第一帧初始化
    first = cloud_seq[0]

    # 去掉 NaN
    mask = ~np.isnan(first).any(axis=1)
    first = first[mask]

    # 去掉全0
    mask = ~(np.all(first == 0, axis=1))
    first = first[mask]

    if len(first) == 0:
        print("[FATAL] 第一帧点云就是空的")
        return

    pcd.points = o3d.utility.Vector3dVector(first)

    vis.add_geometry(pcd)

    for i in range(len(cloud_seq)):
        pts = cloud_seq[i]

        # 清理
        mask = ~np.isnan(pts).any(axis=1)
        pts = pts[mask]

        mask = ~(np.all(pts == 0, axis=1))
        pts = pts[mask]

        if len(pts) == 0:
            print(f"[WARNING] frame {i} 空，跳过")
            continue

        pcd.points = o3d.utility.Vector3dVector(pts)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.03)

    vis.destroy_window()

# 6. 可视化点云

print("\n🎯 正在播放 Head 点云...")
visualize_sequence(cloud_head[start:end], title="Head Cloud")

print("\n🎯 正在播放 Right 点云...")
visualize_sequence(cloud_right[start:end], title="Right Cloud")

# 7. 显示一张 RGB
frame_id = start

img = rgb_head[frame_id]

print("\n显示 RGB 图像（按任意键关闭）")
cv2.imshow("Head RGB", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. 打印 state (q)
print("\n=== STATE (q) ===")
for i in range(start, end):
    print(f"frame {i-start}: {state[i]}")

# 9. 打印 action
print("\n=== ACTION ===")
for i in range(start, end):
    print(f"frame {i-start}: {action[i]}")