import mujoco
import numpy as np
import math

current_v = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}
current_wp_index = 0  
last_print_time = 0.0

def get_robot_pose(model, data):
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rx = data.xpos[base_id][0]
    ry = data.xpos[base_id][1]
    ryaw = math.atan2(data.xmat[base_id][3], data.xmat[base_id][0])
    return rx, ry, ryaw

def limit_accel(current, target, max_accel):
    if target > current:
        return min(target, current + max_accel)
    elif target < current:
        return max(target, current - max_accel)
    return target

def follow_navigation_path(model, data, path_world, base_speed=1.5):
    global current_v, current_wp_index

    id_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_right_joint_velocity")
    id_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_left_joint_velocity")
    id_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_right_joint_velocity")
    id_rl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_left_joint_velocity")

    if -1 in [id_fr, id_fl, id_rr, id_rl]:
        return

    if path_world is None or current_wp_index >= len(path_world) or current_wp_index == -1:
        target_l, target_r = 0.0, 0.0
        if current_wp_index != -1:
            print("Đã đến đích an toàn!")
            current_wp_index = -1 
    else:
        rx, ry, ryaw = get_robot_pose(model, data)
        tx, ty = path_world[current_wp_index]
        
        dx = tx - rx
        dy = ty - ry
        distance = math.hypot(dx, dy)
        
        if distance < 0.2:
            current_wp_index += 1
            return 
            
        final_tx, final_ty = path_world[-1]
        dist_to_final = math.hypot(final_tx - rx, final_ty - ry)
        
        current_base_speed = base_speed
        
        if dist_to_final < 0.2:
            current_base_speed = base_speed * (dist_to_final / 0.2)
            current_base_speed = max(current_base_speed, 1.0)
        
        target_yaw = math.atan2(dy, dx)
        angle_error = target_yaw - ryaw
        
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        angle_multiplier = math.cos(angle_error)
        
        if angle_multiplier < 0:
            v_forward = 0.0
        else:
            v_forward = max(0.8, current_base_speed * (angle_multiplier ** 2))
            
        Kp_turn = 1.5 
        v_turn = Kp_turn * angle_error
            
        target_l = v_forward - v_turn
        target_r = v_forward + v_turn

        max_wheel_speed = 5.0 
        target_l = np.clip(target_l, -max_wheel_speed, max_wheel_speed)
        target_r = np.clip(target_r, -max_wheel_speed, max_wheel_speed)

    if current_wp_index == -1:
        MAX_ACCEL = 0.055
    else:
        MAX_ACCEL = 0.001  
    
    current_v['fl'] = limit_accel(current_v['fl'], target_l, MAX_ACCEL)
    current_v['rl'] = limit_accel(current_v['rl'], target_l, MAX_ACCEL)
    current_v['fr'] = limit_accel(current_v['fr'], target_r, MAX_ACCEL)
    current_v['rr'] = limit_accel(current_v['rr'], target_r, MAX_ACCEL)
    
    data.ctrl[id_fl] = current_v['fl']
    data.ctrl[id_rl] = current_v['rl']
    data.ctrl[id_fr] = current_v['fr']
    data.ctrl[id_rr] = current_v['rr']