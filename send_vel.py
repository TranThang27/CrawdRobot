import mujoco

# 1. Biến lưu vận tốc hiện tại để làm mượt (Smoothing)
# Khởi tạo vận tốc ban đầu của 4 bánh bằng 0
current_v = {
    'fl': 0.0, 'fr': 0.0, 
    'rl': 0.0, 'rr': 0.0
}

def apply_constant_move(model, data, speed=10.0):
    """
    Hàm điều khiển robot tự động đi thẳng với vận tốc 10.0 rad/s
    """
    global current_v

    # Lấy ID động cơ từ model
    id_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_right_joint_velocity")
    id_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_left_joint_velocity")
    id_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_right_joint_velocity")
    id_rl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_left_joint_velocity")

    if -1 in [id_fr, id_fl, id_rr, id_rl]:
        return

    # BƯỚC 1: Xác định vận tốc MỤC TIÊU cố định (Đi thẳng)
    # Vì bạn muốn đi thẳng, tất cả bánh xe đều nhận giá trị speed dương
    target_fl = speed
    target_fr = speed
    target_rl = speed
    target_rr = speed

    # BƯỚC 2: Bộ lọc làm mượt (Smoothing Filter)
    # Giữ alpha để robot khởi động êm, không bị giật (bốc đầu) dù vận tốc nhỏ
    alpha = 0.05 

    current_v['fl'] += (target_fl - current_v['fl']) * alpha
    current_v['fr'] += (target_fr - current_v['fr']) * alpha
    current_v['rl'] += (target_rl - current_v['rl']) * alpha
    current_v['rr'] += (target_rr - current_v['rr']) * alpha

    # BƯỚC 3: Truyền lệnh vào controller
    data.ctrl[id_fl] = current_v['fl']
    data.ctrl[id_fr] = current_v['fr']
    data.ctrl[id_rl] = current_v['rl']
    data.ctrl[id_rr] = current_v['rr']