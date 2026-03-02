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


def print_one_sample(obs, action, timestamps, idx=0):
    print(f"\n========== SAMPLE {idx} ==========")
    sample = obs[idx]

    print("timestamp:", timestamps[idx])
    print("action   :", action[idx])
    print("q        :", sample["q"])

    for k, v in sample.items():
        if k == "q":
            continue
        print(f"{k} data (last obs step):")
        print(v[-1])   # 打印这一帧的完整数组

    print("=================================\n")


def visualize_one_sample(obs, idx=0):
    sample = obs[idx]

    for k, v in sample.items():
        if k == "q":
            continue

        img = v[-1]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        cv2.imshow(k, img)

    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    BLOSC_FILE = r"aoliao_dataset\demo0011.blosc"

    if not os.path.exists(BLOSC_FILE):
        raise FileNotFoundError(BLOSC_FILE)

    obs, action, timestamps = load_blosc_file(BLOSC_FILE)

    # 1️⃣ 打印整个文件的数据 shape
    print_shapes(obs, action, timestamps)

    # 2️⃣ 打印第 0 帧的所有数据
    print_one_sample(obs, action, timestamps, idx=0)

    # 3️⃣ 显示第 0 帧的相机画面
    visualize_one_sample(obs, idx=0)
