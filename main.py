import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2

from send_vel import follow_navigation_path 
from navigation import get_navigation_path

# 1. Load môi trường
XML_PATH = "env_mujoco_2/mujoco/scene.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="head_camera")

# --- PHẦN 1: TÌNH TOÁN ĐƯỜNG ĐI (NAVIGATION) TRƯỚC KHI CHẠY ---
START_WORLD = (0.93, 0.52) 
GOAL_WORLD = (2.06, 3.55) 
print("=========================================")
path_world = get_navigation_path(START_WORLD, GOAL_WORLD)

if path_world is None:
    print(" Dừng chương trình vì không có quỹ đạo bay!")
    exit()
else:
    print(f"Đã nhận được quỹ đạo gồm {len(path_world)} điểm waypoints.")
print("=========================================")

# --- PHẦN 2: TÌM ID ĐỘNG CƠ CÁNH TAY ---
def get_arm_actuators(model):
    arm_ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if "arm" in name or "torso" in name or "head" in name:
            arm_ids.append(i)
    return arm_ids

arm_actuator_ids = get_arm_actuators(model)

# --- PHẦN 3: THIẾT LẬP TƯ THẾ GẬP TAY ---
arm_pose = {
    'arm_left_1_joint': 1.5,
    'arm_right_1_joint': 0.0,
    'arm_left_2_joint': 1.5,
    'arm_right_2_joint': 1.5,
    'arm_left_4_joint': 2.2,
    'arm_right_4_joint': 0.8,
    'torso_lift_joint': 0.15
}

mujoco.mj_forward(model, data)

print("⏳ Robot đang khởi tạo và chờ ổn định vật lý (5 giây)...")
speed_factor = 5 
nav_started = False 

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        for _ in range(speed_factor):
            
            # 1. KHÓA CHẾT TƯ THẾ TAY LIÊN TỤC (Chống giật)
            for joint_name, value in arm_pose.items():
                try:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    qpos_adr = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_adr] = value # Ép góc liên tục
                except: pass

            # 2. LOGIC ĐỢI 5 GIÂY ỔN ĐỊNH
            if data.time < 1.5:
                # Phanh cứng 4 bánh xe để xe không bị trôi đi
                id_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_right_joint_velocity")
                id_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_left_joint_velocity")
                id_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_right_joint_velocity")
                id_rl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_left_joint_velocity")
                
                if id_fr != -1: data.ctrl[id_fr] = 0.0
                if id_fl != -1: data.ctrl[id_fl] = 0.0
                if id_rr != -1: data.ctrl[id_rr] = 0.0
                if id_rl != -1: data.ctrl[id_rl] = 0.0
            else:
                # Sau 5 giây, bắt đầu gọi hàm Navigation
                if not nav_started:
                    print("🚀 Đã ổn định xong! Bắt đầu bám đường (Path Tracking)...")
                    nav_started = True
                    
                # Gọi hàm bám đường đã viết ở send_vel.py
                follow_navigation_path(model, data, path_world, base_speed=1.5)

            # Bước nhảy vật lý
            mujoco.mj_step(model, data)
        
        # Đồng bộ giao diện 3D
        viewer.sync()
            
        # Cân bằng thời gian thực
        time_until_next_step = model.opt.timestep - (time.time() - step_start) / speed_factor
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

cv2.destroyAllWindows()