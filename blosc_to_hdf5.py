#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import blosc
import h5py
import numpy as np
import cv2


# -----------------------------
# blosc 读取
# -----------------------------
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
    actions, offset = unpack_one(raw, offset)
    timestamps, offset = unpack_one(raw, offset)

    return obs, actions, timestamps


# -----------------------------
# 单个 demo 写入 hdf5
# -----------------------------
def write_one_demo(h5file, demo_name, obs, actions):
    T = len(obs)

    # =========================
    # q 拆分 (14维)
    # =========================
    q_all = np.stack([o["q"] for o in obs]).astype(np.float32)  # (T, 14)

    robot0_qpos = q_all[:, 0:7]                      # (T, 7)
    robot0_gripper_qpos = q_all[:, 7:8]              # (T, 1)
    robot0_end_pos = q_all[:, 8:11]                  # (T, 3)
    robot0_end_rxryrz = q_all[:, 11:14]              # (T, 3)

    # =========================
    # 图像处理
    # =========================
    agentview_images = []
    agentview_head_images = []

    for o in obs:
        # right camera
        img_right = o["cam_right_rgb"][-1]  # (H, W, 3)
        img_right = cv2.resize(
            img_right, (84, 84), interpolation=cv2.INTER_AREA
        )
        agentview_images.append(img_right)

        # head camera
        img_head = o["cam_head_rgb"][-1]    # (H, W, 3)
        img_head = cv2.resize(
            img_head, (84, 84), interpolation=cv2.INTER_AREA
        )
        agentview_head_images.append(img_head)

    agentview_images = np.stack(agentview_images).astype(np.uint8)
    agentview_head_images = np.stack(agentview_head_images).astype(np.uint8)

    # =========================
    # actions：只取前 8 维
    # =========================
    actions = actions[:, :8].astype(np.float32)  # (T, 8)



    # =========================
    # 写 HDF5
    # =========================
    grp = h5file.create_group(f"data/{demo_name}")

    grp.create_dataset("actions", data=actions)

    obs_grp = grp.create_group("obs")

    obs_grp.create_dataset(
        "agentview_image",
        data=agentview_images,
        compression="gzip",
        compression_opts=4
    )
    obs_grp.create_dataset(
        "agentview_head_image",
        data=agentview_head_images,
        compression="gzip",
        compression_opts=4
    )

    obs_grp.create_dataset("robot0_qpos", data=robot0_qpos)
    obs_grp.create_dataset("robot0_gripper_qpos", data=robot0_gripper_qpos)
    obs_grp.create_dataset("robot0_end_pos", data=robot0_end_pos)
    obs_grp.create_dataset("robot0_end_rxryrz", data=robot0_end_rxryrz)


import re

def extract_number(fname):
    """
    从文件名中提取数字，用于排序
    e.g. demo_12.blosc -> 12
    """
    nums = re.findall(r"\d+", fname)
    return int(nums[-1]) if len(nums) > 0 else -1


def convert_dataset(blosc_dir, hdf5_path):
    blosc_files = [
        f for f in os.listdir(blosc_dir)
        if f.endswith(".blosc")
    ]

    if len(blosc_files) == 0:
        raise RuntimeError("No .blosc files found")

    # ========= 按数字顺序排序 =========
    blosc_files = sorted(blosc_files, key=extract_number)

    print(f"[INFO] Found {len(blosc_files)} demos")

    with h5py.File(hdf5_path, "w") as h5file:
        for idx, fname in enumerate(blosc_files, start=0):
            blosc_path = os.path.join(blosc_dir, fname)
            demo_name = f"demo_{idx}"

            obs, actions, timestamps = load_blosc_file(blosc_path)

            # ========= 按时间戳排序 =========
            order = np.argsort(timestamps)
            obs = np.asarray(obs, dtype=object)[order]
            actions = np.asarray(actions)[order]
            timestamps = np.asarray(timestamps)[order]
            # =================================

            print(f"[INFO] Writing {demo_name} <- {fname}")
            write_one_demo(h5file, demo_name, obs, actions)

    print(f"[OK] All demos saved into {hdf5_path}")


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":

    BLOSC_DIR = r"dataset"
    HDF5_FILE = r"dataset/dataset.hdf5"

    convert_dataset(
        blosc_dir=BLOSC_DIR,
        hdf5_path=HDF5_FILE
    )
