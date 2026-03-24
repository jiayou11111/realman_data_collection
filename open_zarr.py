import zarr
import numpy as np
import random
import os
import cv2

zarr_path = "pink_weita_delete0data_initdofchange_camerachange_BGR_cloud.zarr"
SAVE_DIR = "save_cloud"


# ========= 打印结构 =========
def print_structure(group, prefix=""):
    for k, v in group.items():
        if isinstance(v, zarr.Group):
            print(f"\n[GROUP] {prefix}{k}/")
            print_structure(v, prefix + k + "/")
        else:
            print(f"[DATASET] {prefix}{k}  shape={v.shape} dtype={v.dtype}")


# ========= 计算全局范围（固定视角核心） =========
def compute_global_range(clouds):
    all_pts = clouds.reshape(-1, 3)
    center = all_pts.mean(0)
    scale = np.max(np.abs(all_pts - center)) + 1e-6
    return center, scale


# ========= 点云渲染（固定视角版） =========
def render_pc(pc, center, scale, size=512):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    pc = (pc - center) / scale   # 👉 固定视角关键

    pts = ((pc[:, :2] + 1) / 2 * (size - 1)).astype(np.int32)

    for x, y in pts:
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = (255, 255, 255)

    return img


# ========= 图像 =========
def save_and_play_images(imgs, name):
    path = os.path.join(SAVE_DIR, f"{name}_img")
    os.makedirs(path, exist_ok=True)

    print(f"\n🎬 播放 {name} 图像 (q退出)")

    for i, img in enumerate(imgs):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(path, f"{i:05d}.png"), img_bgr)
        cv2.imshow(name, img_bgr)

        if cv2.waitKey(30) == ord('q'):
            break

    cv2.destroyWindow(name)
    print(f"✅ {name} 图像保存: {path}")


# ========= 点云 =========
def save_and_play_cloud(clouds, name):
    path = os.path.join(SAVE_DIR, f"{name}_cloud")
    os.makedirs(path, exist_ok=True)

    video_path = os.path.join(SAVE_DIR, f"{name}_cloud.mp4")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (512, 512)
    )

    # ✅ 计算全局范围（关键）
    center, scale = compute_global_range(clouds)

    print(f"\n🎬 播放 {name} 点云 (固定视角, q退出)")

    for pc in clouds:
        img = render_pc(pc, center, scale)

        writer.write(img)
        cv2.imshow(name, img)

        if cv2.waitKey(30) == ord('q'):
            break

    writer.release()
    cv2.destroyWindow(name)

    print(f"🎥 视频: {video_path}")


# ========= 主程序 =========
if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    root = zarr.open(zarr_path, mode='r')

    print("\n========== 📦 STRUCTURE ==========")
    print_structure(root)

    data = root["data"]
    episode_ends = root["meta"]["episode_ends"][:]

    # ========= 随机 episode =========
    ep = random.randint(0, len(episode_ends) - 1)
    s = 0 if ep == 0 else episode_ends[ep - 1]
    e = episode_ends[ep]

    print(f"\n🎲 episode={ep}  range=[{s},{e})")

    sample = {k: data[k][s:e] for k in data.keys()}

    print("\n========== 📊 SHAPE ==========")
    for k, v in sample.items():
        print(k, v.shape)

    # ========= action / state =========
    print("\n===== ACTION =====")
    print(sample["action"])

    print("\n===== STATE =====")
    print(sample["state"])

    # ========= 图像 =========
    if "cam_head_rgb" in sample:
        save_and_play_images(sample["cam_head_rgb"], "head")

    if "cam_right_rgb" in sample:
        save_and_play_images(sample["cam_right_rgb"], "right")

    # ========= 点云 =========
    if "cloud_cam_head" in sample:
        save_and_play_cloud(sample["cloud_cam_head"], "head")

    if "cloud_cam_right" in sample:
        save_and_play_cloud(sample["cloud_cam_right"], "right")

    print("\n🎉 DONE")