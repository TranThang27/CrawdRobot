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
START_WORLD = (-0.88,0.88) 
GOAL_WORLD = (2.0, 1.62) 

path_world = get_navigation_path(START_WORLD, GOAL_WORLD)

# Lấy ID của mối dính (weld) đã khai báo trong XML
weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'grab_book')

# Tư thế tay robot mục tiêu khi di chuyển
target_arm_pose = {
    'arm_left_1_joint': 1.5,
    'arm_right_1_joint': 0.0,
    'arm_left_2_joint': 1.5,
    'arm_right_2_joint': 1.5,
    'arm_left_3_joint': 0.0,
    'arm_left_4_joint': 2.2,
    'arm_right_4_joint': 0.8,
    'torso_lift_joint': 0.15
}
current_arm_pose = target_arm_pose.copy()

mujoco.mj_forward(model, data)

speed_factor = 5 
nav_started = False 
is_grabbed = False # Cờ đánh dấu đã dính vật hay chưa
stop_time = -1.0
arm_raised_time = -1.0


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
            # Cập nhật tư thế các khớp tay (Servo-like control) từ từ về phía mục tiêu
            for joint_name, target_val in target_arm_pose.items():
                curr_val = current_arm_pose.get(joint_name, target_val)
                # Hệ số nội suy 0.01 để tay di chuyển từ từ (Exponential Smoothing)
                curr_val += (target_val - curr_val) * 0.01
                current_arm_pose[joint_name] = curr_val
                try:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    data.qpos[model.jnt_qposadr[joint_id]] = curr_val 
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
                    if stop_time < 0:
                        stop_time = data.time
                        
                    if data.time - stop_time > 1.2:
                        if arm_raised_time < 0:
                            # Thay đổi tư thế khớp tay để vươn thẳng về phía trước
                            target_arm_pose['arm_left_1_joint'] = 1.5 # Chỉnh vai
                            target_arm_pose['arm_left_2_joint'] = 0.2 # Nâng cánh tay lên thẳng
                            target_arm_pose['arm_left_3_joint'] = 0.0 # Xoay cánh tay lại
                            target_arm_pose['arm_left_4_joint'] = 0.0 # Duỗi thẳng khuỷu tay
                            arm_raised_time = data.time
                            print("Đang giơ tay thẳng ra trước...")
                            
                        elif data.time - arm_raised_time > 1.5:
                            # Đã đợi tay vươn ra đủ thời gian, thực hiện nhặt
                            if weld_id != -1:
                                # Dịch chuyển sách vào tay trước khi dính để tránh lực kéo khổng lồ (bay loạn)
                                try:
                                    book_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "living_room_paperback_book_0_s4_0")
                                    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left_left_finger_link")
                                    
                                    b_jnt = model.body_jntadr[book_id]
                                    if b_jnt != -1:
                                        q_adr = model.jnt_qposadr[b_jnt]
                                        v_adr = model.jnt_dofadr[b_jnt]
                                        
                                        # 1. Lấy vị trí và hướng hiện tại của tay
                                        grip_xpos = data.xpos[grip_id]
                                        grip_xmat = data.xmat[grip_id].reshape(3, 3)
                                        grip_quat = data.xquat[grip_id]
                                        
                                        # 2. Tính offset trong không gian world (trùng với relpose="0 0 -0.12" trong XML)
                                        # Hướng mũi nhọn của ngón tay (tiago_dual) thực chất là trục Z ÂM.
                                        local_offset = np.array([0.0, 0.0, -0.15])
                                        world_offset = grip_xmat @ local_offset
                                        
                                        # 3. Áp dụng toạ độ và góc xoay cho sách
                                        data.qpos[q_adr:q_adr+3] = grip_xpos + world_offset
                                        data.qpos[q_adr+3:q_adr+7] = grip_quat.copy()
                                        
                                        # 4. Tắt ranh giới va chạm để sách không bị lực vật lý đẩy văng khỏi ngón tay
                                        book_geom_adr = model.body_geomadr[book_id]
                                        book_geom_num = model.body_geomnum[book_id]
                                        for g in range(book_geom_adr, book_geom_adr + book_geom_num):
                                            model.geom_contype[g] = 0
                                            model.geom_conaffinity[g] = 0
                                        
                                        # Hủy vận tốc cũ
                                        data.qvel[v_adr:v_adr+6] = 0.0
                                        mujoco.mj_forward(model, data)
                                except Exception as e:
                                    print("Lỗi teleport:", e)
        
                                data.eq_active[weld_id] = 1 # KÍCH HOẠT DÍNH!
                                is_grabbed = True
                                print(f"--- Đã đến đích! Đã 'dính' vật thể tại thời điểm: {data.time:.2f}s ---")
                                
                                # Đưa tay về lại vị trí cất ban đầu
                                target_arm_pose['arm_left_1_joint'] = 1.5
                                target_arm_pose['arm_left_2_joint'] = 1.5
                                target_arm_pose['arm_left_3_joint'] = 0.2
                                target_arm_pose['arm_left_4_joint'] = 2.2
                                
                                # Tính đường đi mới và tiếp tục di chuyển
                                print("Đang lập bản đồ đến vị trí tiếp theo...")
                                pos_now = data.body('base_link').xpos[:2]
                                # Tôi tạo tọa độ quay về chỗ xuất phát (0.93, 0.52), bác có thể đổi thành (2.0, 1.31) nếu muốn
                                new_path = get_navigation_path((pos_now[0], pos_now[1]), (5.69,3.14)) 
                                if new_path:
                                    path_world = new_path
                                    send_vel.current_wp_index = 0

            mujoco.mj_step(model, data)
        
        viewer.sync()
            
        # Điều tiết FPS
        time_until_next_step = model.opt.timestep - (time.time() - step_start) / speed_factor
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# cv2.destroyAllWindows()