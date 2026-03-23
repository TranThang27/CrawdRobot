import mujoco
import mujoco.viewer
import time
import numpy as np
# import cv2

# Giữ nguyên các module custom của bạn
import send_vel
from send_vel import follow_navigation_path 
from navigation import get_navigation_path

# Cấu hình đường dẫn và khởi tạo
XML_PATH = "env_mujoco_2/mujoco/scene.xml" 
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="head_camera")

# Tọa độ di chuyển
START_WORLD = (0.93, 0.52) 
GOAL_WORLD = (2.0, 1.31) 

path_world = get_navigation_path(START_WORLD, GOAL_WORLD)

# Lấy ID của mối dính (weld) đã khai báo trong XML
weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'grab_book')

# Tư thế tay robot khi di chuyển
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

speed_factor = 5 
nav_started = False 
is_grabbed = False # Cờ đánh dấu đã dính vật hay chưa


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # 1. Xử lý Camera
        renderer.update_scene(data, camera="head_camera")
        pixels = renderer.render() 
        # bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Robot Eye", bgr_pixels) 
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # 2. Vòng lặp mô phỏng vật lý
        for _ in range(speed_factor):
            # Cập nhật tư thế các khớp tay (Servo-like control)
            for joint_name, value in arm_pose.items():
                try:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    data.qpos[model.jnt_qposadr[joint_id]] = value 
                except: pass

            # Logic di chuyển
            if data.time < 1.5:
                # Đứng yên đợi ổn định
                pass 
            else:
                if not nav_started:
                    nav_started = True
                
                # Gọi hàm navigation của bạn
                follow_navigation_path(model, data, path_world, base_speed=2.5)

                # 3. KIỂM TRA ĐIỀU KIỆN DÍNH VẬT
                # Lấy vị trí hiện tại của robot (thường là base_link hoặc base_footprint)
                # Nếu không biết tên body, bạn có thể dùng data.qpos[0:2] nếu base là freejoint
                try:
                    current_pos = data.body('base_link').xpos[:2]
                except:
                    current_pos = data.qpos[0:2] 

                dist_to_goal = np.linalg.norm(current_pos - np.array(GOAL_WORLD))

                if send_vel.current_wp_index == -1 and not is_grabbed:
                    if weld_id != -1:
                        # Dịch chuyển sách vào tay trước khi dính để tránh lực kéo khổng lồ (bay loạn)
                        try:
                            book_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "living_room_paperback_book_0_s4_0")
                            grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_left_finger_link")
                            
                            b_jnt = model.body_jntadr[book_id]
                            if b_jnt != -1:
                                q_adr = model.jnt_qposadr[b_jnt]
                                v_adr = model.jnt_dofadr[b_jnt]
                                
                                # Teleport tọa độ
                                target = data.xpos[grip_id].copy()
                                target[2] -= 0.02 # Lùi Z xuống xíu cho khớp tay
                                data.qpos[q_adr:q_adr+3] = target
                                
                                # Hủy vận tốc cũ
                                data.qvel[v_adr:v_adr+6] = 0.0
                                mujoco.mj_forward(model, data)
                        except Exception as e:
                            print("Lỗi teleport:", e)

                        data.eq_active[weld_id] = 1 # KÍCH HOẠT DÍNH!
                        is_grabbed = True
                        print(f"--- Đã đến đích! Đã 'dính' vật thể tại thời điểm: {data.time:.2f}s ---")

            mujoco.mj_step(model, data)
        
        viewer.sync()
            
        # Điều tiết FPS
        time_until_next_step = model.opt.timestep - (time.time() - step_start) / speed_factor
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# cv2.destroyAllWindows()