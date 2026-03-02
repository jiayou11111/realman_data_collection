#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import cv2
import os


# ------------------------------------------------
# 打印 HDF5 结构
# ------------------------------------------------
def print_hdf5_structure(h5_path):
    print("\n========== HDF5 STRUCTURE ==========")
    with h5py.File(h5_path, "r") as f:
        f.visit(print)
    print("===================================\n")


# ------------------------------------------------
# 加载一个 demo
# ------------------------------------------------
def load_demo(h5_path, demo_name):
    with h5py.File(h5_path, "r") as f:
        grp = f[f"data/{demo_name}"]

        data = {}

        # actions
        data["actions"] = grp["actions"][:] if "actions" in grp else None

        # obs
        obs_grp = grp["obs"]
        data["agentview_image"] = obs_grp["agentview_image"][:]
        data["agentview_head_image"] = obs_grp["agentview_head_image"][:]
        data["robot0_qpos"] = obs_grp["robot0_qpos"][:]
        data["robot0_gripper_qpos"] = obs_grp["robot0_gripper_qpos"][:]
        data["robot0_end_pos"] = obs_grp["robot0_end_pos"][:]
        data["robot0_end_rxryrz"] = obs_grp["robot0_end_rxryrz"][:]

    return data


# ------------------------------------------------
# 打印 shape / dtype
# ------------------------------------------------
def print_data_info(data):
    print("\n========== DATA INFO ==========")
    for k, v in data.items():
        if v is None:
            print(f"{k:<28}: None")
        else:
            print(f"{k:<28}: shape={v.shape}, dtype={v.dtype}")
    print("================================\n")


# ------------------------------------------------
# 打印真实数据内容（前 N 帧）
# ------------------------------------------------
def print_data_preview(data, num_frames=3):
    print("\n========== DATA PREVIEW ==========")

    for k, v in data.items():
        if v is None:
            continue

        print(f"\n[{k}]")
        print(f"  shape: {v.shape}")
        print(f"  dtype: {v.dtype}")

        # 图像
        if v.ndim == 4 and v.dtype == np.uint8:
            for i in range(min(num_frames, v.shape[0])):
                img = v[i]
                print(
                    f"  frame {i}: "
                    f"min={img.min()}, max={img.max()}, mean={img.mean():.2f}"
                )

        # 向量 / action / qpos
        else:
            for i in range(min(num_frames, v.shape[0])):
                print(f"  step {i}: {v[i]}")

    print("=================================\n")


# ------------------------------------------------
# 可视化图像序列（按键切换）
# ------------------------------------------------
def visualize_images(images, win_name="image"):
    print("Press 'n' for next frame, 'q' to quit.")
    idx = 0
    T = images.shape[0]

    while True:
        img = images[idx]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        cv2.imshow(win_name, img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            idx = (idx + 1) % T

    cv2.destroyAllWindows()


# ------------------------------------------------
# main
# ------------------------------------------------
if __name__ == "__main__":

    HDF5_FILE = r"dataset/dataset.hdf5"
    DEMO_NAME = "demo_9"   # ← 可改 demo_1 / demo_2 / ...

    if not os.path.exists(HDF5_FILE):
        raise FileNotFoundError(HDF5_FILE)

    # 1️⃣ 打印 HDF5 层级
    print_hdf5_structure(HDF5_FILE)

    # 2️⃣ 加载 demo
    data = load_demo(HDF5_FILE, DEMO_NAME)

    # 3️⃣ 打印 shape / dtype
    print_data_info(data)

    # 4️⃣ 打印真实数值（前 3 帧）
    print_data_preview(data, num_frames=3)

    # 5️⃣ 可视化 agentview 图像序列
    visualize_images(data["agentview_image"], win_name="agentview")

    # 6️⃣ 可视化 head 相机（如需要）
    # visualize_images(data["agentview_head_image"], win_name="agentview_head")
