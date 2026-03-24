# import h5py
# import numpy as np

# # 文件路径
# hdf5_file_path = "brown_weita_RGB\\brown_weita_RGB.hdf5"

# # 指定 demo
# demo_name = "demo_0"

# if __name__ == "__main__":
#     with h5py.File(hdf5_file_path, "r") as f:

#         demo_path = f"data/{demo_name}"

#         if demo_path not in f:
#             print(f"❌ {demo_name} 不存在")
#             exit()

#         print(f"\n========== 打印 {demo_name} ==========")

#         demo_group = f[demo_path]

#         # ========= 1️⃣ 打印 actions =========
#         actions = demo_group["actions"][()]
#         print("\n===== ACTIONS =====")
#         print("shape:", actions.shape)
#         print(actions)   # 完整打印，不省略

#         # ========= 2️⃣ 打印 robot0_qpos =========
#         qpos = demo_group["obs"]["robot0_qpos"][()]
#         print("\n===== robot0_qpos =====")
#         print("shape:", qpos.shape)
#         print(qpos)      # 完整打印

#         # ========= 3️⃣ 打印 robot0_gripper_qpos =========
#         gripper = demo_group["obs"]["robot0_gripper_qpos"][()]
#         print("\n===== robot0_gripper_qpos =====")
#         print("shape:", gripper.shape)
#         print(gripper)   # 完整打印

#         print("\n========== 打印完成 ==========")



# 打印结构
import h5py

# 文件路径
hdf5_file_path = "brown_weita_RGB\\brown_weita_RGB.hdf5"


def print_hdf5_structure(name, obj):
    """
    遍历 HDF5 文件结构
    - Group：打印路径
    - Dataset：打印 shape 和 dtype
    """
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name}")
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"\n[GROUP] {name}")


if __name__ == "__main__":
    with h5py.File(hdf5_file_path, "r") as f:
        print("\n========== HDF5 文件结构 ==========\n")

        # 递归遍历整个文件
        f.visititems(print_hdf5_structure)

        print("\n========== 打印完成 ==========")

