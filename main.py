import mujoco
import mujoco.viewer
import time
import numpy as np
from send_vel import apply_constant_move 
import cv2
# 1. Load môi trường
XML_PATH = "env_mujoco_2/mujoco/scene.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="head_camera")


# --- PHẦN 1: TÌM ID ĐỘNG CƠ CÁNH TAY ---
# Mình sẽ dùng hàm này để lấy tất cả ID của cánh tay cho gọn
def get_arm_actuators(model):
    arm_ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if "arm" in name or "torso" in name or "head" in name:
            arm_ids.append(i)
    return arm_ids

arm_actuator_ids = get_arm_actuators(model)

# --- PHẦN 2: THIẾT LẬP TƯ THẾ GẬP TAY (Góc tính bằng Radian) ---
# Thắng có thể chỉnh các con số này cho đến khi ưng ý
arm_pose = {
    # Khớp 1: Xoay vai vào phía trong (ngực)
    'arm_left_1_joint': 1.5,    # Xoay vai trái vào trong một chút
    'arm_right_1_joint': 0.0      ,   # Xoay vai phải vào trong một chút

    # Khớp 2: Ép bả vai xuống sát sườn hơn
    'arm_left_2_joint': 1.5,    # Tăng từ 1.2 lên 1.5 để khép nách
    'arm_right_2_joint': 1.5,   # Tăng từ 1.2 lên 1.5 để khép nách

    # Khớp 4: Gập khuỷu tay sâu hơn
    'arm_left_4_joint': 2.2,    # Gập khuỷu tay sâu hơn (tối đa khoảng 2.3)
    'arm_right_4_joint': 0.8,

    'torso_lift_joint': 0.15
}

# Áp dụng tư thế ngay lập tức trước khi chạy (Tránh bị T-pose lúc đầu)

mujoco.mj_forward(model, data)

# --- PHẦN 3: VÒNG LẶP MÔ PHỎNG ---
print("Robot đã gập tay. Đang chạy mô phỏng...")
speed_factor = 5 

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        for joint_name, value in arm_pose.items():
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_adr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_adr] = value
            except: pass

            
        for _ in range(speed_factor):
            # 1. Điều khiển bánh xe (từ file control.py)
            apply_constant_move(model, data, speed=15.0)

            # 2. Điều khiển tay (Khóa tay ở tư thế gập)
            # Nếu actuator của bạn là 'position', dòng này sẽ giữ tay cứng ngắc
            # Nếu là 'velocity', dòng này sẽ gán vận tốc = 0 (phanh tay)
            for act_id in arm_actuator_ids:
                # Tìm xem actuator này điều khiển khớp nào để gán đúng giá trị trong arm_pose
                act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
                if "velocity" in act_name:
                    data.ctrl[act_id] = 0.0

        mujoco.mj_step(model, data)
        
        viewer.sync()
        renderer.update_scene(data, camera="head_camera")
        pixels = renderer.render()
        
        # Chuyển đổi màu từ RGB sang BGR để OpenCV hiển thị đúng
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("Robot Eye", bgr_pixels)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # Nhấn q để đóng cửa sổ cam
            break
        time_until_next_step = model.opt.timestep - (time.time() - step_start) / speed_factor
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)