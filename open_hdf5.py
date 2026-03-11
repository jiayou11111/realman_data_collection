# import h5py

# # 文件路径
# hdf5_file_path = "data/robomimic/datasets/lift/mh/weita_5hz_dataset.hdf5"
# # hdf5_file_path = "real_data/dataset.hdf5"

# def print_hdf5_structure(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
#     elif isinstance(obj, h5py.Group):
#         print(f"Group: {name}")

# with h5py.File(hdf5_file_path, "r") as f:
#     f.visititems(print_hdf5_structure)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import h5py
# import numpy as np

# # 文件路径
# hdf5_file_path = "data/robomimic/datasets/lift/mh/weita_5hz_dataset.hdf5"

# # 指定你要打印的 demo 名称
# demo_name = "demo_30"

# def print_dataset(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"\n📊 Dataset: {name}")
#         print(f"   shape: {obj.shape}")
#         print(f"   dtype: {obj.dtype}")

#         data = obj[()]  # 读取全部数据

#         # 如果是图像数据，避免爆屏
#         if data.ndim >= 3:
#             print("   ⚠ 图像数据过大，仅显示第一帧:")
#             print(data[0])
#         elif data.size > 100:
#             print("   ⚠ 数据过大，仅显示前 10 行:")
#             print(data[:10])
#         else:
#             print("   数据内容:")
#             print(data)


# if __name__ == "__main__":
#     with h5py.File(hdf5_file_path, "r") as f:

#         demo_path = f"data/{demo_name}"

#         if demo_path not in f:
#             print(f"❌ {demo_name} 不存在")
#         else:
#             print(f"\n========== 打印 {demo_name} 全部内容 ==========")

#             group = f[demo_path]

#             # 遍历这个 demo 下所有内容
#             group.visititems(print_dataset)

#             print("\n========== 打印完成 ==========")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np

# 文件路径
hdf5_file_path = "init_rectangle_pink_weita\\init_rectangle_pink_weita_dataset.hdf5"

# 指定 demo
demo_name = "demo_0"

if __name__ == "__main__":
    with h5py.File(hdf5_file_path, "r") as f:

        demo_path = f"data/{demo_name}"

        if demo_path not in f:
            print(f"❌ {demo_name} 不存在")
            exit()

        print(f"\n========== 打印 {demo_name} ==========")

        demo_group = f[demo_path]

        # ========= 1️⃣ 打印 actions =========
        actions = demo_group["actions"][()]
        print("\n===== ACTIONS =====")
        print("shape:", actions.shape)
        print(actions)   # 完整打印，不省略

        # ========= 2️⃣ 打印 robot0_qpos =========
        qpos = demo_group["obs"]["robot0_qpos"][()]
        print("\n===== robot0_qpos =====")
        print("shape:", qpos.shape)
        print(qpos)      # 完整打印

        # ========= 3️⃣ 打印 robot0_gripper_qpos =========
        gripper = demo_group["obs"]["robot0_gripper_qpos"][()]
        print("\n===== robot0_gripper_qpos =====")
        print("shape:", gripper.shape)
        print(gripper)   # 完整打印

        print("\n========== 打印完成 ==========")