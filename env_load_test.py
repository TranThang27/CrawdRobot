import mujoco
import mujoco.viewer
import time

# 1. Đường dẫn tới file XML của bạn
xml_path = '/home/acer/Clawdrobot/pal_tiago_dual/scene_motor.xml'

try:
    # 2. Load model từ file XML
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # 3. Tạo đối tượng data chứa trạng thái mô phỏng (vị trí, vận tốc, lực...)
    data = mujoco.MjData(model)

    # 4. Mở cửa sổ hiển thị (Viewer)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Đang chạy mô phỏng. Nhấn Ctrl+C để dừng.")
        
        while viewer.is_running():
            step_start = time.time()

            # Thực hiện một bước tính toán vật lý
            mujoco.mj_step(model, data)

            # Cập nhật hình ảnh lên viewer
            viewer.sync()

            # Đảm bảo tốc độ mô phỏng khớp với thời gian thực
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print(f"Lỗi khi load file XML: {e}")