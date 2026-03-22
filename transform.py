# import numpy as np

# def quat_to_euler_xyz(q):
#     """
#     四元数 -> 欧拉角 (XYZ顺序)
    
#     参数
#     ----
#     q : array-like [w, x, y, z]

#     返回
#     ----
#     euler : np.array [roll_x, pitch_y, yaw_z]  (单位: rad)
#     """

#     w, x, y, z = q

#     # roll (x-axis rotation)
#     t0 = 2.0 * (w * x + y * z)
#     t1 = 1.0 - 2.0 * (x * x + y * y)
#     roll_x = np.arctan2(t0, t1)

#     # pitch (y-axis rotation)
#     t2 = 2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     pitch_y = np.arcsin(t2)

#     # yaw (z-axis rotation)
#     t3 = 2.0 * (w * z + x * y)
#     t4 = 1.0 - 2.0 * (y * y + z * z)
#     yaw_z = np.arctan2(t3, t4)

#     return np.array([roll_x, pitch_y, yaw_z])


# q1 =[0.0245, 0.3042, 0.9133, 0.2697]

# euler1 = quat_to_euler_xyz(q1)



# print("roll, pitch, yaw (rad):", euler1)
# print("roll, pitch, yaw (deg):", np.degrees(euler1))



import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pose):
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

def matrix_to_pose(T):
    x, y, z = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    rx, ry, rz = rot.as_euler('zyx', degrees=False)

    return x, y, z, rx, ry, rz

# ================== 已知数据 ==================

# 世界坐标系下的末端
world_ee = [0.531361, -0.056726, 0.508789, -1.5708, 1.131, -1.425]
# base坐标系下的末端
base_ee = [0.01595, -0.056735, 0.735497, -0.469, 0.639, -0.551]

# ================== 计算 ==================

T_world_ee = pose_to_matrix(world_ee)
T_base_ee  = pose_to_matrix(base_ee)

# 求逆
T_ee_base = np.linalg.inv(T_base_ee)

# 核心计算
T_world_base = T_world_ee @ T_ee_base

print("T_world_base:")
print(T_world_base)

T_base_world = np.linalg.inv(T_world_base)

print("T_base_world:")
print(T_base_world)
# # 转回位姿
# base_pose = matrix_to_pose(T_world_base)

# print("Base 在世界坐标系下：")
# print("x, y, z, rx, ry, rz =")
# print(base_pose)