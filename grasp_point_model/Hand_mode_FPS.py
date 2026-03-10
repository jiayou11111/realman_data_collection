import os
import torch
import numpy as np
import pytorch_kinematics as pk
import trimesh as tm
import open3d as o3d
import time

class GripperModel:
    def __init__(
            self,
            urdf_path,
            mesh_root,
            gripper_links,
            device="cuda",
            scale=100.0,
            sample_points=512
    ):
        self.device = device
        self.scale = scale
        self.sample_points = sample_points
        self.gripper_links = gripper_links

        # -------- 构建运动学链 --------
        with open(urdf_path, 'rb') as f:
            urdf_bytes = f.read()

        self.robot = pk.build_chain_from_urdf(urdf_bytes).to(
            dtype=torch.float32,
            device=device
        )

        print("Number of movable joints:", self.robot.n_joints)

        # -------- 采样每个 link 的 mesh --------
        self.link_points = {}

        for link in gripper_links:
            stl_path = self._find_stl_path(mesh_root, link)

            if not os.path.exists(stl_path):
                raise FileNotFoundError(f"STL file not found: {stl_path}")

            mesh = tm.load(stl_path, force="mesh")
            pts = mesh.sample(sample_points)

            self.link_points[link] = torch.tensor(
                pts,
                dtype=torch.float32,
                device=device
            )

    def _find_stl_path(self, mesh_root, link_name):
        return os.path.join(mesh_root, f"{link_name}.STL")

    # -------- GPU FPS --------
    def farthest_point_sampling(self, points, num_points=128):
        pts = points.to(self.device).float()
        N = pts.shape[0]

        if N == 0:
            return pts

        indices = torch.zeros(num_points, dtype=torch.long, device=self.device)
        distances = torch.ones(N, device=self.device) * 1e10
        farthest = torch.randint(0, N, (1,), device=self.device)

        for i in range(num_points):
            indices[i] = farthest
            centroid = pts[farthest]
            dist = torch.sum((pts - centroid) ** 2, dim=1)
            distances = torch.minimum(distances, dist)
            farthest = torch.argmax(distances)

        return pts[indices]

    # -------- Forward Kinematics --------
    def forward_kinematics(self, q):
        if q.dim() == 1:
            q = q.unsqueeze(0)

        q = q.to(self.device)

        fk = self.robot.forward_kinematics(q)
        all_points = []

        for link_name, pts in self.link_points.items():
            T = fk[link_name].get_matrix()[0]

            n = pts.shape[0]
            ones = torch.ones(n, 1, device=self.device) 

            pts_h = torch.cat([pts, ones], dim=1) #相对于该link下的坐标系
            pts_world = (pts_h @ T.T)[:, :3]

            all_points.append(pts_world)

        all_points = torch.cat(all_points, dim=0)

        return all_points * self.scale


# ================= 用户配置 =================
a=time.time()
urdf_path = r"D:\texttwo\grasp_point_model\Embodied lifting robot_two wheels_RM75-B-V\urdf\robot_hand.urdf"
mesh_root = r"D:\texttwo\grasp_point_model\Embodied lifting robot_two wheels_RM75-B-V\meshes"

gripper_links = [
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
    urdf_path,
    mesh_root,
    gripper_links,
    device=device
)

# -------- 设置关节角 --------
q = torch.zeros(gripper.robot.n_joints, dtype=torch.float32, device=device)
q[6] = -0.8# 手指张开
points = gripper.forward_kinematics(q)

# -------- FPS --------
fps_points = gripper.farthest_point_sampling(points)

b=time.time()
print("总耗时: {:.4f} 秒".format(b - a))
print("FPS 点云 shape:", fps_points.shape)


# ================= Open3D 可视化 =================

cloud_np = fps_points.detach().cpu().numpy().astype(np.float64)

if cloud_np.shape[0] == 0:
    raise ValueError("点云为空，请检查 FK 或 STL 采样")

print("可视化点数:", cloud_np.shape[0])

# 创建点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud_np)
pcd.paint_uniform_color([1.0, 0.0, 0.0])

# 创建坐标系
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)

# 可视化
o3d.visualization.draw_geometries(
    [pcd, coord],
    window_name="Gripper Point Cloud",
    width=1200,
    height=900
)