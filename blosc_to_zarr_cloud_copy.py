# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import zarr
from tqdm import tqdm
import cv2
from termcolor import cprint
import blosc
import numpy as np
import open3d as o3d
import torch
import pytorch_kinematics as pk
import trimesh as tm
import os
from scipy.spatial.transform import Rotation as R

ARM_AND_GRIPPER_DOF = 8
FINGER_OPEN_ANGLE = 0.91

#外参待修改
# 相机相对于末端坐标系
RIGHT_CAM_EXTRINSIC = np.array([
    [-0.04168939, 0.99735439, -0.05955013, -0.09113213],
    [-0.9989049, -0.04033902, -0.02370156, 0.03494395],
    [-0.02123666, -0.06047302, -0.9979439, 0.01531916],
    [0, 0, 0, 1]
], dtype=np.float32)

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
    action, offset = unpack_one(raw, offset)
    timestamps, offset = unpack_one(raw, offset)

    return obs, action, timestamps


# Gripper Model

class GripperModel:

    def __init__(self, urdf_path, mesh_root, gripper_links,
                 device="cuda", sample_points=64):

        self.device = device
        self.gripper_links = gripper_links
        self.sample_points = sample_points

        with open(urdf_path, 'rb') as f:
            urdf_bytes = f.read()

        self.robot = pk.build_chain_from_urdf(urdf_bytes).to(
            dtype=torch.float32,
            device=device
        )

        print("Robot joints:", self.robot.n_joints)
        self.joint_names = self.robot.get_joint_parameter_names()

        self.link_points = {}

        for link in gripper_links:

            stl = os.path.join(mesh_root, f"{link}.STL")

            mesh = tm.load(stl, force="mesh")

            pts = mesh.sample(sample_points)

            self.link_points[link] = torch.tensor(
                pts, dtype=torch.float32, device=device
            )

    def _transform_points(self, points, transform):

        ones = torch.ones(len(points), 1, device=self.device)

        points_h = torch.cat([points, ones], dim=1)

        return (points_h @ transform.T)[:, :3]


    def compose_joint_vector(self, q_data):

        q_data = np.asarray(q_data, dtype=np.float32).reshape(-1)

        if q_data.shape[0] < ARM_AND_GRIPPER_DOF:
            raise ValueError(
                f"q_data must contain at least {ARM_AND_GRIPPER_DOF} values, got {q_data.shape[0]}"
            )

        q_data = q_data[:ARM_AND_GRIPPER_DOF]

        full_q = np.zeros(self.robot.n_joints, dtype=np.float32)

        # -------- 机械臂7关节 --------
        arm_joint_map = {
            "joint_right_1": q_data[0],
            "joint_right_2": q_data[1],
            "joint_right_3": q_data[2],
            "joint_right_4": q_data[3],
            "joint_right_5": q_data[4],
            "joint_right_6": q_data[5],
            "joint_right_7": q_data[6],
        }

        # -------- gripper --------
        gripper_state = float(np.clip(q_data[7], 0.0, 1.0))
        gripper_state = 1.0 - gripper_state  # 0: open, 1: close
        finger_angle = -FINGER_OPEN_ANGLE * gripper_state

        finger_joint_map = {
            "Left_1_Joint2": finger_angle,
            "Left_Support_Joint2": -finger_angle,
            "Left_2_Joint2": finger_angle,
            "Right_2_Joint2": -finger_angle,
            "Right_1_Joint2": -finger_angle,
            "Right_Support_Joint2": -finger_angle,
        }

        # -------- 合并所有joint --------
        joint_map = {**arm_joint_map, **finger_joint_map}

        # -------- 填充full_q --------
        for index, joint_name in enumerate(self.joint_names):
            if joint_name in joint_map:
                full_q[index] = joint_map[joint_name]

        return torch.tensor(full_q, dtype=torch.float32, device=self.device)


    def forward_kinematics(self, q):

        if q.dim() == 1:
            q = q.unsqueeze(0)

        q = q.to(self.device)

        fk = self.robot.forward_kinematics(q)

        # print("\n正在查询 link_right_7 的坐标变换...")
        # # 获取 link_right_7 的变换
        self.transform_right_7 = fk["link_right_7"]
        
        # print("link_right_7 变换矩阵", self.transform_right_7)

        all_points = []

        for link_name, pts in self.link_points.items():

            T = fk[link_name].get_matrix()[0]

            pts_world = self._transform_points(pts, T)

            all_points.append(pts_world)
            # print("link:", link_name)

        return torch.cat(all_points, dim=0)


class VisualizeHeadCloud:

    def __init__(self,
                 camera_name="cam_head_depth",
                 num_points=1600):
        
        self.camera_name = camera_name
        self.num_points = num_points
        self.cam_params = {

            "cam_head_depth": {
                "fx": 393.23,
                "fy": 393.23,
                "cx": 317.85,
                "cy": 243.23,
                "scale": 0.001
            },

            "cam_right_depth": {
                "fx": 389.05,
                "fy": 389.05,
                "cx": 314.73,
                "cy": 236.93,
                "scale": 0.001
            }
        }

        cam = self.cam_params[camera_name]

        self.fx = cam["fx"]
        self.fy = cam["fy"]
        self.cx = cam["cx"]
        self.cy = cam["cy"]
        self.scale = cam["scale"]

    def depth_to_cloud(self, depth):

        depth = depth.astype(np.float32) * self.scale

        H, W = depth.shape

        u = np.arange(W)
        v = np.arange(H)

        u, v = np.meshgrid(u, v)

        Z = depth

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        cloud = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        mask = np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > 0)

        return cloud[mask]

    def farthest_point_sampling(self, points, num_points=1600, use_cuda=True):

        """
        points: numpy array (N, 3)
        return:
            sampled_points: (num_points, 3)
            indices: (num_points,)
        """

        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        pts = torch.from_numpy(points).float().to(device)   # (N, 3)
        N = pts.shape[0]

        # 存储采样索引
        indices = torch.zeros(num_points, dtype=torch.long, device=device)

        # 存储每个点到当前已选点集的最小距离
        distances = torch.ones(N, device=device) * 1e10

        # 随机选择第一个点
        farthest = torch.randint(0, N, (1,), device=device)

        for i in range(num_points):
            indices[i] = farthest
            centroid = pts[farthest, :].view(1, 3)

            # 计算到新加入点的距离
            dist = torch.sum((pts - centroid) ** 2, dim=1)

            # 更新最小距离
            mask = dist < distances
            distances[mask] = dist[mask]

            # 选择最远点
            farthest = torch.argmax(distances)

        sampled_points = pts[indices]

        return sampled_points.cpu().numpy(), indices.cpu().numpy()


    def preprocess_point_cloud(self, points, num_points=1600, use_cuda=True):

        WORK_SPACE = [
            [-0.6, 1.3],   # X
            [-0.3, 0.5],   # Y
            [0.2, 1.8]     # Z
        ]

        # 只保留工作空间内的点
        mask = (
            (points[:, 0] > WORK_SPACE[0][0]) & (points[:, 0] < WORK_SPACE[0][1]) &
            (points[:, 1] > WORK_SPACE[1][0]) & (points[:, 1] < WORK_SPACE[1][1]) &
            (points[:, 2] > WORK_SPACE[2][0]) & (points[:, 2] < WORK_SPACE[2][1])
        )
        points = points[mask]

        if len(points) == 0:
            return np.zeros((num_points, 3), dtype=np.float32)

        points_xyz = points[:, :3]

        points_xyz, _ = self.farthest_point_sampling(points_xyz, num_points=num_points, use_cuda=use_cuda)

        return points_xyz


    def process_head_cloud(self, sample):

        # ===== 相机点云 =====
        depth = np.squeeze(sample[self.camera_name])
        cloud_cam_head = self.preprocess_point_cloud(
            self.depth_to_cloud(depth),
            num_points=self.num_points
        )

        return cloud_cam_head



class VisualizeRightCloud:

    def __init__(self,
                 gripper_model,
                 camera_name="cam_right_depth",
                 num_points=1600):
        
        self.camera_name = camera_name
        self.num_points = num_points
        self.gripper = gripper_model

        self.cam_params = {

            "cam_head_depth": {
                "fx": 393.23,
                "fy": 393.23,
                "cx": 317.85,
                "cy": 243.23,
                "scale": 0.001
            },

            "cam_right_depth": {
                "fx": 389.05,
                "fy": 389.05,
                "cx": 314.73,
                "cy": 236.93,
                "scale": 0.001
            }
        }

        cam = self.cam_params[camera_name]

        self.fx = cam["fx"]
        self.fy = cam["fy"]
        self.cx = cam["cx"]
        self.cy = cam["cy"]
        self.scale = cam["scale"]

        
    def transform_cloud(self, cloud, T):

        ones = np.ones((cloud.shape[0], 1))

        cloud_h = np.concatenate([cloud, ones], axis=1)

        cloud_trans = (T @ cloud_h.T).T[:, :3]

        return cloud_trans

    def pose_to_matrix(self, pose):
        """
        pose: [x, y, z, rx, ry, rz]
        rx ry rz: rad
        """
        x, y, z, rx, ry, rz = pose

        cx, cy, cz = np.cos([rx, ry, rz])
        sx, sy, sz = np.sin([rx, ry, rz])

        # Rz Ry Rx
        Rz = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])

        R = Rz @ Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def depth_to_cloud(self, depth):

        depth = depth.astype(np.float32) * self.scale

        H, W = depth.shape

        u = np.arange(W)
        v = np.arange(H)

        u, v = np.meshgrid(u, v)

        Z = depth

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        cloud = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        mask = np.isfinite(cloud).all(axis=1) & (cloud[:, 2] > 0)

        return cloud[mask]

    def farthest_point_sampling(self, points, num_points=1600, use_cuda=True):

        """
        points: numpy array (N, 3)
        return:
            sampled_points: (num_points, 3)
            indices: (num_points,)
        """

        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        pts = torch.from_numpy(points).float().to(device)   # (N, 3)
        N = pts.shape[0]

        # 存储采样索引
        indices = torch.zeros(num_points, dtype=torch.long, device=device)

        # 存储每个点到当前已选点集的最小距离
        distances = torch.ones(N, device=device) * 1e10

        # 随机选择第一个点
        farthest = torch.randint(0, N, (1,), device=device)

        for i in range(num_points):
            indices[i] = farthest
            centroid = pts[farthest, :].view(1, 3)

            # 计算到新加入点的距离
            dist = torch.sum((pts - centroid) ** 2, dim=1)

            # 更新最小距离
            mask = dist < distances
            distances[mask] = dist[mask]

            # 选择最远点
            farthest = torch.argmax(distances)

        sampled_points = pts[indices]

        return sampled_points.cpu().numpy(), indices.cpu().numpy()


    def preprocess_point_cloud(self, points, num_points=1600, use_cuda=True):

        WORK_SPACE = [
            [-0.6, 1.3],   # X
            [-0.3, 0.5],   # Y
            [0.2, 1.8]     # Z
        ]

        # 只保留工作空间内的点
        mask = (
            (points[:, 0] > WORK_SPACE[0][0]) & (points[:, 0] < WORK_SPACE[0][1]) &
            (points[:, 1] > WORK_SPACE[1][0]) & (points[:, 1] < WORK_SPACE[1][1]) &
            (points[:, 2] > WORK_SPACE[2][0]) & (points[:, 2] < WORK_SPACE[2][1])
        )
        points = points[mask]

        if len(points) == 0:
            return np.zeros((num_points, 3), dtype=np.float32)

        points_xyz = points[:, :3]

        points_xyz, _ = self.farthest_point_sampling(points_xyz, num_points=num_points, use_cuda=use_cuda)

        return points_xyz


    def process_right_cloud(self, sample):

        # ===== 关节数据 =====
        q_data = sample["q"][:8].astype(np.float32)

        q = self.gripper.compose_joint_vector(q_data)
        cloud_hand = self.gripper.forward_kinematics(q)

        cloud_hand = cloud_hand.detach().cpu().numpy()

        # ===== 相机点云 =====
        depth = np.squeeze(sample[self.camera_name])
        cloud_cam = self.preprocess_point_cloud(
            self.depth_to_cloud(depth),
            num_points=self.num_points
        )


        T_base_ee = self.gripper.transform_right_7.get_matrix().cpu().numpy()[0]

        cloud_ee = self.transform_cloud(cloud_cam, RIGHT_CAM_EXTRINSIC)

        T_rot = np.eye(4)
        T_rot[:3, :3] = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        cloud_ee = self.transform_cloud(cloud_ee, T_rot)

        # ===== ee → base =====
        cloud_base = self.transform_cloud(cloud_ee, T_base_ee)

        # ===== 融合 =====
        merged_cloud = np.concatenate([cloud_base, cloud_hand], axis=0)
        return merged_cloud





URDF_PATH = r"D:\texttwo\realman_data_collection\grasp_point_model\Embodied lifting robot_two wheels_RM75-B-V\urdf\robot_hand.urdf"

MESH_ROOT = r"D:\texttwo\realman_data_collection\grasp_point_model\Embodied lifting robot_two wheels_RM75-B-V\meshes"

GRIPPER_LINKS = [
    "hand_base2",
    "Left_1_Link2",
    "Left_Support_Link2",
    "Left_2_Link2",
    "Right_2_Link2",
    "Right_1_Link2",
    "Right_Support_Link2"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

gripper = GripperModel(
    URDF_PATH,
    MESH_ROOT,
    GRIPPER_LINKS,
    device=device,
    sample_points=64
)

viewer_right = VisualizeRightCloud(
    gripper,
    num_points=1600
)
viewer_head = VisualizeHeadCloud(  
    num_points=2048
)

INPUT_DIR = "init_rectangle_blue_weita"
save_data_path = "init_rectangle_blue_weita.zarr"

# 4️⃣ 数据容器
state_list, pose_list = [], []
action_list, action_pose_list = [], []
rgb_head_list, rgb_right_list = [], []
depth_head_list, depth_right_list = [], []
cloud_head_list, cloud_right_list = [], []
episode_ends = []

frame_count = 0  # 累计帧数
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.blosc")))

# 5️⃣ 读取数据
for file in files:

    print("Processing:", file)
    obs, action, _ = load_blosc_file(file)

    #删除第一帧
    obs = obs[1:]
    action = action[1:]

    if len(obs) == 0:
        print("Skip file with no frames after dropping the first frame:", file)
        continue


    episode_length = len(obs)

    for i, sample in enumerate(tqdm(obs)):

        # ===== state =====
        q = sample["q"].astype(np.float32)
        a = action[i].astype(np.float32)

        # state / pose
        state_list.append(q[:8])
        pose_list.append(q[8:14])

        # action / action_pose
        action_list.append(a[:8])
        action_pose_list.append(a[8:14])

        # RGB
        head_rgb = cv2.resize(np.squeeze(sample["cam_head_rgb"]), (320, 240))
        right_rgb = cv2.resize(np.squeeze(sample["cam_right_rgb"]), (320, 240))
        rgb_head_list.append(head_rgb.astype(np.uint8))
        rgb_right_list.append(right_rgb.astype(np.uint8))

        # Depth
        head_depth = cv2.resize(np.squeeze(sample["cam_head_depth"]), (84, 84), interpolation=cv2.INTER_NEAREST)
        right_depth = cv2.resize(np.squeeze(sample["cam_right_depth"]), (84, 84), interpolation=cv2.INTER_NEAREST)
        depth_head_list.append(head_depth.astype(np.float32))
        depth_right_list.append(right_depth.astype(np.float32))

        # ===== 相机点云 =====
        cloud_head = viewer_head.process_head_cloud(sample)
        cloud_right = viewer_right.process_right_cloud(sample)

        cloud_head_list.append(cloud_head)
        cloud_right_list.append(cloud_right)

    # 记录 episode 结束位置
    frame_count += episode_length
    episode_ends.append(frame_count)

# 6️⃣ 转 numpy

state_arrays = np.stack(state_list, axis=0)
pose_arrays = np.stack(pose_list, axis=0)
action_arrays = np.stack(action_list, axis=0)
action_pose_arrays = np.stack(action_pose_list, axis=0)
rgb_head_arrays = np.stack(rgb_head_list, axis=0)
rgb_right_arrays = np.stack(rgb_right_list, axis=0)
depth_head_arrays = np.stack(depth_head_list, axis=0)
depth_right_arrays = np.stack(depth_right_list, axis=0)
cloud_head_arrays = np.stack(cloud_head_list, axis=0)  # (N,2048,3)
cloud_right_arrays = np.stack(cloud_right_list, axis=0)  # (N,2048,3)
episode_ends_arrays = np.array(episode_ends, dtype=np.int64)

# 图像通道放最后
if rgb_head_arrays.shape[1] == 3:
    rgb_head_arrays = np.transpose(rgb_head_arrays, (0,2,3,1))
if rgb_right_arrays.shape[1] == 3:
    rgb_right_arrays = np.transpose(rgb_right_arrays, (0,2,3,1))

# 7️⃣ 保存到 zarr
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

# ===== chunk size =====
img_head_chunk = (100, rgb_head_arrays.shape[1], rgb_head_arrays.shape[2], rgb_head_arrays.shape[3])
img_right_chunk = (100, rgb_right_arrays.shape[1], rgb_right_arrays.shape[2], rgb_right_arrays.shape[3])
point_cloud_chunk = (100, cloud_head_arrays.shape[1], cloud_head_arrays.shape[2])
depth_head_chunk = (100, depth_head_arrays.shape[1], depth_head_arrays.shape[2])
depth_right_chunk = (100, depth_right_arrays.shape[1], depth_right_arrays.shape[2])
action_chunk = (100, action_arrays.shape[1])
state_chunk = (100, state_arrays.shape[1])


# ===== 保存 data =====
# zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)

zarr_data.create_dataset('cam_head_rgb', data=rgb_head_arrays, chunks=img_head_chunk, dtype='uint8', overwrite=True,compressor=compressor)
zarr_data.create_dataset('cam_right_rgb', data=rgb_right_arrays, chunks=img_right_chunk, dtype='uint8', overwrite=True,compressor=compressor)

zarr_data.create_dataset('cloud_cam_head', data=cloud_head_arrays, chunks=point_cloud_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('cloud_cam_right', data=cloud_right_arrays, chunks=point_cloud_chunk, dtype='float32', overwrite=True,compressor=compressor)

zarr_data.create_dataset('cam_head_depth', data=depth_head_arrays, chunks=depth_head_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('cam_right_depth', data=depth_right_arrays, chunks=depth_right_chunk, dtype='float32', overwrite=True,compressor=compressor)

zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('pose', data=pose_arrays, chunks=state_chunk, dtype='float32', overwrite=True,compressor=compressor)
zarr_data.create_dataset('action_pose', data=action_pose_arrays, chunks=action_chunk, dtype='float32', overwrite=True,compressor=compressor)

# ===== 保存 meta =====
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True,compressor=compressor)
# ==========================================================
# 8️⃣ 打印信息

cprint(f'cam_head_rgb shape: {rgb_head_arrays.shape}, range: [{np.min(rgb_head_arrays)}, {np.max(rgb_head_arrays)}]', 'green')
cprint(f'cam_right_rgb shape: {rgb_right_arrays.shape}, range: [{np.min(rgb_right_arrays)}, {np.max(rgb_right_arrays)}]', 'green')
cprint(f'cloud_cam_head shape: {cloud_head_arrays.shape}, range: [{np.min(cloud_head_arrays)}, {np.max(cloud_head_arrays)}]', 'green')
cprint(f'cloud_cam_right shape: {cloud_right_arrays.shape}, range: [{np.min(cloud_right_arrays)}, {np.max(cloud_right_arrays)}]', 'green')
cprint(f'cam_head_depth shape: {depth_head_arrays.shape}, range: [{np.min(depth_head_arrays)}, {np.max(depth_head_arrays)}]', 'green')
cprint(f'cam_right_depth shape: {depth_right_arrays.shape}, range: [{np.min(depth_right_arrays)}, {np.max(depth_right_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'pose shape: {pose_arrays.shape}, range: [{np.min(pose_arrays)}, {np.max(pose_arrays)}]', 'green')
cprint(f'action_pose shape: {action_pose_arrays.shape}, range: [{np.min(action_pose_arrays)}, {np.max(action_pose_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

print("\n✅ Conversion Finished Successfully!")
