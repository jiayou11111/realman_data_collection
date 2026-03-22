#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blosc
import numpy as np
import os
import cv2


def unpack_one(raw, offset):
    """
    从 raw[offset:] 解一个 blosc array
    """
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


def print_shapes(obs, action, timestamps):
    print("\n========== DATA SHAPES ==========")
    print(f"obs type        : {type(obs)}")
    print(f"num samples     : {len(obs)}")
    print(f"action shape    : {action.shape}")
    print(f"timestamp shape : {timestamps.shape}")

    sample = obs[0]
    print("\n--- obs[0] content ---")
    for k, v in sample.items():
        if k == "q":
            print(f"{k}: shape={np.array(v).shape}, dtype={np.array(v).dtype}")
        else:
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    print("================================\n")



def print_all_q_action(obs, action, timestamps):
    all_q = np.stack([np.asarray(sample["q"]) for sample in obs], axis=0)
    all_action = np.asarray(action)

    print("\n========== ALL Q AND ACTION ==========")
    print("timestamps shape:", np.asarray(timestamps).shape)
    print("all q shape     :", all_q.shape)
    print("all q           :")
    print(all_q)
    print("\naction shape    :", all_action.shape)
    print("all action      :")
    print(all_action)
    print("\n======================================\n")


def visualize_one_sample(obs, idx=0):
    sample = obs[idx]

    for k, v in sample.items():
        if k == "q":
            continue

        img = v[-1]

        # ===== depth 图 =====
        if "depth" in k:
            depth = img.astype(np.float32)

            depth_norm = cv2.normalize(
                depth,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            cv2.imshow(k, depth_norm)

        # ===== RGB 图 =====
        else:
            rgb = img

            if rgb.dtype != np.uint8:
                rgb = rgb.astype(np.uint8)

            # OpenCV 显示需要 BGR
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow(k, bgr)

    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    BLOSC_FILE = r"dataset\demo0005.blosc"

    if not os.path.exists(BLOSC_FILE):
        raise FileNotFoundError(BLOSC_FILE)

    obs, action, timestamps = load_blosc_file(BLOSC_FILE)

    # 1️⃣ 打印整个文件的数据 shape
    print_shapes(obs, action, timestamps)

    # 2️⃣ 打印当前 demo 的所有 q 和 action
    print_all_q_action(obs, action, timestamps)

    # 4️⃣ 显示第 0 帧的相机画面
    visualize_one_sample(obs, idx=5)
