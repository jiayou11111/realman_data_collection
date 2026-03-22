#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


class VisualizeCloud:

    def __init__(self,
                 blosc_path,
                 gripper_model,
                 camera_name="cam_right_depth",
                 num_points=1600):

        self.blosc_path = blosc_path
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

        self.obs, self.action, self.timestamps = self.load_blosc_file(blosc_path)

        self.q = self.extract_q(self.obs)
        self.ee_pose = self.extract_ee_pose(self.obs)
        # print("ee_pose:", self.ee_pose)

        self.num_frames = len(self.obs)

        print("Total frames:", self.num_frames)

        self.current_frame = 0

    # BL osc loader

    def unpack_one(self, raw, offset):

        nbytes, cbytes, blocksize = blosc.get_cbuffer_sizes(raw[offset:])

        buf = raw[offset: offset + cbytes]

        arr = blosc.unpack_array(buf)

        return arr, offset + cbytes


    def load_blosc_file(self, path):

        with open(path, "rb") as f:
            raw = f.read()

        offset = 0

        obs, offset = self.unpack_one(raw, offset)
        action, offset = self.unpack_one(raw, offset)
        timestamps, offset = self.unpack_one(raw, offset)

        return obs, action, timestamps


    def extract_q(self, obs):

        q_list = []

        for sample in obs:

            if "q" in sample:

                q = np.asarray(sample["q"][:ARM_AND_GRIPPER_DOF], dtype=np.float32)

                q_list.append(q)

        q_array = np.array(q_list)

        # print("q:", q_array)

        return q_array

    def extract_ee_pose(self, obs):

        pose_list = []

        for sample in obs:

            if "q" in sample:

                q = np.asarray(sample["q"], dtype=np.float32)

                ee_pose = q[-6:]   # 取最后6维

                pose_list.append(ee_pose)

        pose_array = np.array(pose_list)

        return pose_array
        
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


    def get_frame_clouds(self, frame_id):

        sample = self.obs[frame_id]

        depth = np.squeeze(sample[self.camera_name])

        cloud_cam = self.preprocess_point_cloud(
            self.depth_to_cloud(depth),
            num_points=self.num_points
        )

        q_data = self.q[frame_id]
        # print("q_data:", q_data)

        # head camera

        if self.camera_name == "cam_head_depth":

            return cloud_cam, q_data
        # right camera

        q = self.gripper.compose_joint_vector(q_data)

        cloud_hand = self.gripper.forward_kinematics(q)

        cloud_hand = cloud_hand.detach().cpu().numpy()


        # 正确的误解（待解决）
        # # 1 读取末端位姿
        # ee_pose = self.ee_pose[frame_id] # x,y,z,rx,ry,rz

        # print("末端位姿:", ee_pose)

        # # # 构造变换矩阵 
        # T_ee_cam = RIGHT_CAM_EXTRINSIC

        # cloud_cam = self.transform_cloud(
        #     cloud_cam,
        #     T_ee_cam
        # )
        # # 构造绕 EE z 轴旋转 180°
        # T_rot = np.eye(4)
        # T_rot[:3, :3] = np.array([
        #     [1,  0, 0],
        #     [ 0, 1, 0],
        #     [ 0, 0, -1]
        # ])

        # # 再做一次变换
        # cloud_cam = self.transform_cloud(
        #     cloud_cam,
        #     T_rot
        # )

        # T_world_ee = self.pose_to_matrix(ee_pose)

        # T_base_world = [[ 0.69367913 ,0, -0.71986322,  0],
        # [0,  1, 0 ,0],
        # [ 0.7192272  , 0 , 0.69410756 , 0],
        # [ 0.       ,   0.        ,  0.         , 1.        ]]

        # T_base_cam = T_base_world @T_world_ee

        # cloud_cam = self.transform_cloud(
        #     cloud_cam,
        #     T_base_cam
        # )
        # merged_cloud = np.concatenate([cloud_cam, cloud_hand], axis=0)



        # 存疑answer
        tf = self.gripper.transform_right_7

        # # ✅ 正确获取4x4矩阵
        T = tf.get_matrix().cpu().numpy()[0]

        T_base_ee = T

        # 2 hand-eye
        T_ee_cam = RIGHT_CAM_EXTRINSIC

        cloud_cam = self.transform_cloud(
            cloud_cam,
            T_ee_cam
        )
        # 疑惑
        T_rot = np.eye(4)
        T_rot[:3, :3] = np.array([
            [1,  0, 0],
            [ 0, 1, 0],
            [ 0, 0, -1]
        ])

        # 再做一次变换
        cloud_ee_rot = self.transform_cloud(
            cloud_cam,
            T_rot
        )

        cloud_ee_rot = self.transform_cloud(
            cloud_ee_rot,
            T_base_ee
        )

        merged_cloud = np.concatenate([cloud_ee_rot, cloud_hand], axis=0)
        return merged_cloud, q_data

    def update_frame(self, frame_id):

        frame_id = max(0, min(frame_id, self.num_frames - 1))

        self.current_frame = frame_id

        cloud, q_data = self.get_frame_clouds(frame_id)

        self.pcd.points = o3d.utility.Vector3dVector(
            cloud.astype(np.float64)
        )

        self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()


    def visualize(self):

        cloud, _ = self.get_frame_clouds(0)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.vis.create_window("PointCloud Viewer", 1200, 900)

        self.pcd = o3d.geometry.PointCloud()

        self.pcd.points = o3d.utility.Vector3dVector(
            cloud.astype(np.float64)
        )

        self.pcd.paint_uniform_color([1, 0, 0])

        self.vis.add_geometry(self.pcd)

        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0,0,0])

        self.vis.register_key_callback(ord("N"), self.next_frame)
        self.vis.register_key_callback(ord("B"), self.prev_frame)

        self.update_frame(0)

        self.vis.run()

        self.vis.destroy_window()

    def next_frame(self, vis):

        self.update_frame(self.current_frame + 1)

        return False


    def prev_frame(self, vis):

        self.update_frame(self.current_frame - 1)

        return False


    def quit_vis(self, vis):

        vis.close()

        return False



BLOSC_PATH = "dataset/demo0004.blosc"

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

viewer = VisualizeCloud(
    BLOSC_PATH,
    gripper,
    camera_name="cam_right_depth" ,  #cam_right_depth,cam_head_depth
    num_points=1600
)

viewer.visualize()