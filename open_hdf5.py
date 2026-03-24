import h5py
import numpy as np
import cv2
import os
import random

# 文件路径
hdf5_file_path = "pink_weita_dataset.hdf5"

SAVE_DIR = "save_img"

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[DATASET] {name} | shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"\n[GROUP] {name}")


if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    with h5py.File(hdf5_file_path, "r") as f:

        # ========= 2️⃣ 随机选 demo =========
        demo_keys = list(f["data"].keys())
        demo_name = random.choice(demo_keys)

        print(f"\n🎯 随机选择 demo: {demo_name}")

        demo_group = f[f"data/{demo_name}"]
        print("\n========== HDF5 文件结构 ==========\n")
        demo_group.visititems(print_hdf5_structure)

        # ========= 3️⃣ 打印 actions =========
        actions = demo_group["actions"][()]
        print("\n===== ACTIONS =====")
        print("shape:", actions.shape)
        print(actions)
        # ========= 2️⃣ 打印 robot0_qpos =========
        qpos = demo_group["obs"]["robot0_qpos"][()]
        print("\n===== robot0_qpos =====")
        print("shape:", qpos.shape)
        print(qpos)      # 完整打印

        # ========= 4️⃣ 打印 gripper =========
        gripper = demo_group["obs"]["robot0_gripper_qpos"][()]
        print("\n===== robot0_gripper_qpos =====")
        print("shape:", gripper.shape)
        print(gripper)

        # ========= 5️⃣ 保存图像 =========
        obs_group = demo_group["obs"]

        print("\n========== 开始保存图像 ==========")

        for key in obs_group.keys():

            data = obs_group[key]

            # 只处理图像 (T,H,W,3) 或 (T,3,H,W)
            if not isinstance(data, h5py.Dataset):
                continue
            if len(data.shape) != 4:
                continue
            if not (data.shape[-1] == 3 or data.shape[1] == 3):
                continue

            print(f"\n🟢 处理图像: {key} | shape={data.shape}")

            img_data = data[()]

            save_path = os.path.join(SAVE_DIR, demo_name, key)
            os.makedirs(save_path, exist_ok=True)

            for i in range(len(img_data)):
                img = img_data[i]

                # (3,H,W) → (H,W,3)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))

                # float → uint8
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)

                # RGB → BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(save_path, f"{i:05d}.png"), img)

            print(f"✅ 已保存到: {save_path}")

        print("\n🎉 完成！")