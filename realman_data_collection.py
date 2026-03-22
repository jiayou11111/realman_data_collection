#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-camera + robot temporal dataset collection (RealSense D435c)

- Each camera timestamp drives alignment
- Robot states aligned to each camera frame (<= timestamp)
- Multi-step sequence support (n_obs_steps)
- Obs: {q, cam0, cam1, cam2}
- Action: previous q
- Robot initialized by IP
"""

import time
import threading
from collections import deque
from typing import Dict
import numpy as np

import os
from datetime import datetime
import sys
import blosc
import msvcrt
from scipy.spatial.transform import Rotation as R


from Robotic_Arm.rm_robot_interface import *

from camera_socket import CameraSocket
from camera_socket import RealSenseCameraBuffer

PORT = 5000

def enable_nonblocking_keyboard():
    if sys.platform == "win32":
        # Windows: nothing to do
        return None


def get_key_nonblock():
    if sys.platform == "win32":
        if msvcrt.kbhit():
            try:
                return msvcrt.getch().decode("utf-8").lower()
            except:
                return None
        return None



class RobotArm:
    def __init__(self):

        self.l_robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.r_robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        self.l_handle = self.l_robot.rm_create_robot_arm("169.254.128.18", 8080)
        self.r_handle = self.r_robot.rm_create_robot_arm("169.254.128.19", 8080)

        # self.set_r_joints_angles( [-162.3698, -119.6334,  155.1919,  -88.6483,   39.4240,  108.4789, -97.4292],10,0,0,1) #初始位置，前提是要将臂先抬起来


    def get_l_gripper_state(self):
        state = self.l_robot.rm_get_rm_plus_state_info()
        return state[1]['pos'][0]

    def get_r_gripper_state(self):
        state = self.r_robot.rm_get_rm_plus_state_info()
        return state[1]['pos'][0]


    def get_l_robot_joints(self):
        state = self.l_robot.rm_get_current_arm_state()
        return state[1]['joint']

    def get_r_robot_joints(self):
        state = self.r_robot.rm_get_current_arm_state()
        return state[1]['joint']

    def get_l_robot_pose(self):
        state = self.l_robot.rm_get_current_arm_state()
        return state[1]['pose']
    
    def get_r_robot_pose(self):
        state = self.r_robot.rm_get_current_arm_state()
        return state[1]['pose']
    
    def set_r_joints_angles(self, r_angles):
        self.r_robot.rm_movej_canfd(r_angles, True, 0, 1, 50)

    def set_l_joints_angles(self, l_angles):
        self.l_robot.rm_movej_canfd(l_angles, True, 0, 1, 50) 

    def rm_set_l_gripper_position(self, position):
        self.l_robot.rm_set_gripper_position(position, True, 10)     

    def rm_set_r_gripper_position(self, position):
        self.r_robot.rm_set_gripper_position(position, True, 10)      
        

class TimeStampedBuffer:
    def __init__(self, maxlen=3000):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, timestamp: float, data):
        with self.lock:
            self.buffer.append((timestamp, data))


class ArmStateRecorder(threading.Thread):
    def __init__(self, arm, buffer, freq=50):
        super().__init__(daemon=True)
        self.arm = arm
        self.buffer = buffer
        self.dt = 1.0 / freq
        self.running = True


    def run(self):
        # 初始化上一帧缓存
        self.prev_pos = None
        self.prev_rot = None
        self.prev_joint = None
        while self.running:
            ts = time.time()
            joint = self.arm.get_r_robot_joints()
            for i in range(len(joint)):
                joint[i] = joint[i] / 180.0 * np.pi  # deg -> rad
            

            # 将列表转换为 numpy 数组
            joint = np.array(joint, dtype=np.float32)
            
            pos = np.array(self.arm.get_r_robot_pose()[:3], dtype=np.float32)
            orn = np.array(self.arm.get_r_robot_pose()[3:], dtype=np.float32) 
            gripper = np.array([self.arm.get_r_gripper_state()/1000.0], dtype=np.float32)

            # 计算 Δjoint
            if self.prev_joint is not None:
                djoint = joint - self.prev_joint
            else:
                djoint = np.zeros_like(joint)

            self.prev_joint = joint.copy()

            # 计算 Δposition
            if self.prev_pos is not None:
                dpos = pos - self.prev_pos
            else:
                dpos = np.zeros(3, dtype=np.float32)

            self.prev_pos = pos.copy()

            # 计算 Δorientation
            rot_curr = R.from_euler('xyz', orn)

            if self.prev_rot is not None:
                rot_delta = self.prev_rot.inv() * rot_curr
                drot = rot_delta.as_rotvec()
            else:
                drot = np.zeros(3, dtype=np.float32)

            self.prev_rot = rot_curr
     
            q = np.concatenate([joint, gripper, pos, orn], dtype=np.float32)
            dq = np.concatenate([djoint, dpos, drot]).astype(np.float32)
            self.buffer.push(ts, {'q': q, 'dq': dq, 'robot_receive_timestamp': ts})

            time.sleep(self.dt)

    def stop(self):
        self.running = False


class SharedSequenceBuffer:
    def __init__(self, capacity=5000, n_obs_steps=5):
        self.capacity = capacity
        self.obs = [None] * capacity
        self.action = [None] * capacity
        self.timestamp = np.zeros(capacity, dtype=np.float64)
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()
        self.n_obs_steps = n_obs_steps

    def add(self, obs: Dict, action: np.ndarray, timestamp: float):
        with self.lock:
            self.obs[self.ptr] = obs
            self.action[self.ptr] = action
            self.timestamp[self.ptr] = timestamp

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def clear(self):
        """
        Clear all stored data and reset buffer state.
        """
        with self.lock:
            self.obs = [None] * self.capacity
            self.action = [None] * self.capacity
            self.timestamp = np.zeros(self.capacity, dtype=np.float64)
            self.ptr = 0
            self.size = 0

    def __len__(self):
        return self.size


class MultiCameraDatasetBuilder:
    def __init__(self, arm_buffer: TimeStampedBuffer, shared_buffer: SharedSequenceBuffer, cam_buffers: Dict[str, RealSenseCameraBuffer], n_obs_steps=5, frequency=10.0):
        self.arm_buffer = arm_buffer
        self.shared_buffer = shared_buffer
        self.cam_buffers = cam_buffers
        self.n_obs_steps = n_obs_steps
        self.frequency = frequency
        self.prev_q = None
        self.prev_obs = None

    def step(self):

        # 摄像头最新时间戳
        last_timestamps = []
        for buf in self.cam_buffers.values():
            with buf.lock:
                if len(buf.buffer) == 0:
                    return
                last_timestamps.append(buf.buffer[-1][0])
        last_timestamp = max(last_timestamps)

        # 对齐序列时间戳
        dt = 1.0 / self.frequency
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        # 获取机器人数据
        all_robot_data = list(self.arm_buffer.buffer)
        if not all_robot_data:
            return
        robot_timestamps = np.array([x[0] for x in all_robot_data])
        this_idxs = []
        for t in obs_align_timestamps:
            idxs = np.nonzero(robot_timestamps <= t)[0]
            this_idx = idxs[-1] if len(idxs) > 0 else 0
            this_idxs.append(this_idx)


        q = all_robot_data[this_idxs[-1]][1]['q'] 


        # 摄像头对齐帧
        cam_data = {}
        for cid, buf in self.cam_buffers.items():
            imgs = []
            depths = []
            for t in obs_align_timestamps:
                img,depth = buf.get_closest_before(t)
                if img is None or depth is None:
                    return
                imgs.append(img)
                depths.append(depth)
            cam_data[f"{cid}_rgb"] = np.array(imgs)    # (T, H, W, 3)
            cam_data[f"{cid}_depth"] = np.array(depths) # (T, H, W)

        current_obs = {'q': q, **cam_data}

         # 清空所有数据
        # if self.shared_buffer.ptr == 0:
        #     self.prev_obs = None

        if self.prev_obs is None:
            self.prev_obs = current_obs
            return
        
        # ===== 存入 (obs_{t-1}, action_t) =====
        obs_to_store = self.prev_obs
        action_to_store = q

        self.shared_buffer.add(obs_to_store, action_to_store, last_timestamp)

        # 更新缓存
        self.prev_obs = current_obs


t = 0
def save_and_clear(shared_buffer, builder, save_dir="init_rectangle_blue_weita"):#视角/形状/颜色/物体名称

    global t
    os.makedirs(save_dir, exist_ok=True)

    with shared_buffer.lock:
        size = shared_buffer.size
        # 1. 把图像从 dtype=object 转成可压缩的 4-D uint8 数组
        obs = np.asarray(shared_buffer.obs[:size], dtype=object)
        action  = np.asarray(shared_buffer.action[:size])
        time_st = np.asarray(shared_buffer.timestamp[:size])

    # 2. 用 blosc 高速压缩写盘（zstd 平衡速度与压缩率）
    path = os.path.join(save_dir, f"demo{t:04d}.blosc")
    with open(path, 'wb') as f:
        f.write(blosc.pack_array(obs,  cname='zstd', clevel=3))
        f.write(blosc.pack_array(action, cname='zstd', clevel=3))
        f.write(blosc.pack_array(time_st, cname='zstd', clevel=3))

    print(f"[✓] Saved {size} samples -> {path}  ({os.path.getsize(path)>>20} MB)")
    t += 1
    shared_buffer.clear()
    # 清空 builder 的历史状态
    builder.prev_obs = None


def main():

    BUFFER_SIZE = 1500
    N_OBS_STEPS = 1

    arm = RobotArm()

    arm_buffer = TimeStampedBuffer(maxlen=150)
    arm_recorder = ArmStateRecorder(arm, arm_buffer, freq=100)
    arm_recorder.start()


    CAMERAS = [
        {"name": "cam_head",  "port": 5001, "c_ip": "169.254.128.20"},
        {"name": "cam_right", "port": 5002, "c_ip": "169.254.128.20"},
        # {"name": "cam_left",  "port": 5003, "c_ip": "169.254.128.20"},
    ]
    
    cam_buffers = {}
    for cam in CAMERAS:

        cam_buf =  RealSenseCameraBuffer(maxlen=200)

        client = CameraSocket(cam,cam_buf)
    
        cam_buffers[cam["name"]] = cam_buf

        client.start()


    shared_buffer = SharedSequenceBuffer(capacity=BUFFER_SIZE, n_obs_steps=N_OBS_STEPS)
    builder = MultiCameraDatasetBuilder(arm_buffer, shared_buffer, cam_buffers, n_obs_steps=N_OBS_STEPS, frequency=10.0)

    collecting = False

    print("Press 's' to START, 'c' to STOP & SAVE")
    enable_nonblocking_keyboard()  

    try:
        while True:
            key = get_key_nonblock()
            if key == 's':
                collecting = True
                time.sleep(0.5)
                print("[●] Collecting...")
                time.sleep(0.1)
                
            if key == 'c':
                collecting = False
                save_and_clear(shared_buffer, builder)
                print("[■] Stopped")
                time.sleep(0.5)

            if collecting:
                builder.step()
            time.sleep(0.1) #总采集数据频率

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()


