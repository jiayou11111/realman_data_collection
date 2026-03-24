import zarr
import numpy as np
import random


zarr_path = "pink_weita_delete0data_initdofchange_camerachange_BGR_cloud.zarr"


def print_zarr_structure(group, prefix=""):
    """
    递归打印 zarr 结构
    """
    for key, value in group.items():
        if isinstance(value, zarr.Group):
            print(f"\n[GROUP] {prefix}{key}/")
            print_zarr_structure(value, prefix + key + "/")
        else:
            print(f"[DATASET] {prefix}{key}")
            print(f"  shape: {value.shape}")
            print(f"  dtype: {value.dtype}")


def print_range(name, arr):
    print(f"\n--- {name} ---")
    print("shape:", arr.shape)
    print("min:", np.min(arr))
    print("max:", np.max(arr))


if __name__ == "__main__":

    root = zarr.open(zarr_path, mode='r')

    print("\n========== 📦 ZARR STRUCTURE ==========")
    print_zarr_structure(root)

    # ===== 读取数据 =====
    data = root["data"]
    meta = root["meta"]

    episode_ends = meta["episode_ends"][:]

    print("\n========== 🎯 EPISODE INFO ==========")
    print("episode_ends:", episode_ends)

    num_episodes = len(episode_ends)
    print("num_episodes:", num_episodes)

    # ===== 随机选一个 episode =====
    ep_id = random.randint(0, num_episodes - 1)

    start = 0 if ep_id == 0 else episode_ends[ep_id - 1]
    end = episode_ends[ep_id]

    print(f"\n🎲 随机选择 episode: {ep_id}")
    print(f"frame range: [{start}, {end})  length={end - start}")

    # ===== 取该 episode 数据 =====
    sample = {}

    for key in data.keys():
        sample[key] = data[key][start:end]

    print("\n========== 📊 数据范围（该 episode） ==========")

    for key, value in sample.items():

        # ⚠️ 图像/点云太大，不要全算（会慢）
        if value.ndim >= 3:
            # 只抽前10帧
            v = value[:10]
        else:
            v = value

        print_range(key, v)

    print("\n========== ✅ 检查完成 ==========")