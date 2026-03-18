




import mujoco
import numpy as np
import math

# Biến toàn cục để lưu trạng thái
current_v = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
current_wp_index = 0  
last_print_time =0.0
def get_robot_pose(model, data):
    """Hàm lấy tọa độ (x, y) và góc hướng (yaw) thực tế của robot trong MuJoCo"""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rx = data.xpos[base_id][0]
    ry = data.xpos[base_id][1]
    # Lấy góc Yaw từ ma trận xoay
    ryaw = math.atan2(data.xmat[base_id][3], data.xmat[base_id][0])
    return rx, ry, ryaw

def limit_accel(current, target, max_accel):
    """Giới hạn gia tốc để động cơ tăng/giảm từ từ (Chống bốc đầu/ngã ngửa)"""
    if target > current:
        return min(target, current + max_accel)
    elif target < current:
        return max(target, current - max_accel)
    return target

def follow_navigation_path(model, data, path_world, base_speed=1.5):
    """
    Hàm bám quỹ đạo hoàn chỉnh (Bản Mượt mà - Continuous Control):
    - Dùng hàm Cosine để nhả ga tự động khi vào cua.
    - Không dùng if/else phanh gấp.
    - Rà phanh chậm dần khi cách đích < 0.8m.
    """
    global current_v, current_wp_index

    # Lấy ID của 4 động cơ bánh xe
    id_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_right_joint_velocity")
    id_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_left_joint_velocity")
    id_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_right_joint_velocity")
    id_rl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_left_joint_velocity")

    if -1 in [id_fr, id_fl, id_rr, id_rl]:
        return

    # KIỂM TRA ĐÍCH ĐẾN (Chống đi lùi)
    if path_world is None or current_wp_index >= len(path_world) or current_wp_index == -1:
        target_l, target_r = 0.0, 0.0
        if current_wp_index != -1:
            print("🚀 Đã đến đích an toàn!")
            current_wp_index = -1 
    else:
        # 1. Lấy vị trí hiện tại và điểm đến
        rx, ry, ryaw = get_robot_pose(model, data)
        tx, ty = path_world[current_wp_index]
        
        dx = tx - rx
        dy = ty - ry
        distance = math.hypot(dx, dy)
        
        # 2. Chuyển waypoint nếu đã đến đủ gần
        if distance < 0.2:
            current_wp_index += 1
            return 
            
        # 3. KHOẢNG CÁCH ĐẾN ĐÍCH CUỐI CÙNG (RÀ PHANH CHỐNG TRƯỢT KHI DỪNG)
        final_tx, final_ty = path_world[-1]
        dist_to_final = math.hypot(final_tx - rx, final_ty - ry)
        
        current_base_speed = base_speed
        
        # Bắt đầu giảm tốc độ khi cách đích dưới 0.8m
        if dist_to_final < 0.2:
            current_base_speed = base_speed * (dist_to_final / 0.2)
            # Giữ tốc độ tối thiểu là 0.2 để xe không dừng hẳn khi chưa tới nơi
            current_base_speed = max(current_base_speed, 1.0)
        
        # 4. TÍNH SAI SỐ GÓC
        target_yaw = math.atan2(dy, dx)
        angle_error = target_yaw - ryaw
        
        # Chuẩn hóa góc về đoạn [-pi, pi]
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        # ========================================================
        # 5. ĐIỀU KHIỂN VẬN TỐC LIÊN TỤC BẰNG HÀM COSINE (TUYỆT CHIÊU)
        # ========================================================
  
        
        # Nếu góc lệch > 90 độ (xe đang quay lưng lại với mục tiêu), xe dừng tiến và chỉ xoay tại chỗ
        angle_multiplier = math.cos(angle_error)
        
        if angle_multiplier < 0:
            v_forward = 0.0
        else:
            # FIX Ở ĐÂY: Dùng max() bao bọc cả cụm tính toán v_forward!
            # Ép xe luôn lướt tới với tốc độ ít nhất là 0.8, mặc kệ góc lệch có bị khuếch đại thế nào.
            v_forward = max(0.8, current_base_speed * (angle_multiplier ** 2))
            
        # Vận tốc xoay (Đánh lái mượt)
        Kp_turn = 1.5 
        v_turn = Kp_turn * angle_error
            
        # Tính vận tốc 2 bánh
        target_l = v_forward - v_turn
        target_r = v_forward + v_turn

        # Giới hạn trần vận tốc bánh xe để an toàn
        max_wheel_speed = 5.0 
        target_l = np.clip(target_l, -max_wheel_speed, max_wheel_speed)
        target_r = np.clip(target_r, -max_wheel_speed, max_wheel_speed)

    # ========================================================
    # 6. GIA TỐC SIÊU MƯỢT MÀ (Linear Acceleration Limiter)
    # ========================================================
    if current_wp_index == -1:
        MAX_ACCEL = 0.075 # Phanh gấp khi chạm đích chống đâm tường bật ngược
    else:
        MAX_ACCEL = 0.001  
    
    current_v['fl'] = limit_accel(current_v['fl'], target_l, MAX_ACCEL)
    current_v['rl'] = limit_accel(current_v['rl'], target_l, MAX_ACCEL)
    current_v['fr'] = limit_accel(current_v['fr'], target_r, MAX_ACCEL)
    current_v['rr'] = limit_accel(current_v['rr'], target_r, MAX_ACCEL)
    
    # Truyền lệnh điều khiển vào MuJoCo
    data.ctrl[id_fl] = current_v['fl']
    data.ctrl[id_rl] = current_v['rl']
    data.ctrl[id_fr] = current_v['fr']
    data.ctrl[id_rr] = current_v['rr']