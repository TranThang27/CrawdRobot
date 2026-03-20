import mujoco
import mujoco.viewer
import time

xml_path = '/home/acer/Clawdrobot/pal_tiago_dual/scene_motor.xml'

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Đang chạy mô phỏng. Nhấn Ctrl+C để dừng.")
        
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except Exception as e:
    print(f"Lỗi khi load file XML: {e}")