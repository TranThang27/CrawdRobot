import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2

from send_vel import follow_navigation_path 
from navigation import get_navigation_path

XML_PATH = "env_mujoco_2/mujoco/scene.xml" 
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="head_camera")

START_WORLD = (0.93, 0.52) 
GOAL_WORLD = (6.3 , 2.88) 


path_world = get_navigation_path(START_WORLD, GOAL_WORLD)



def get_arm_actuators(model):
    arm_ids = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if "arm" in name or "torso" in name or "head" in name:
            arm_ids.append(i)
    return arm_ids

arm_actuator_ids = get_arm_actuators(model)

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

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        renderer.update_scene(data, camera="head_camera")
        pixels = renderer.render() 
        
        bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("Robot Eye", bgr_pixels) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        for _ in range(speed_factor):
            
            for joint_name, value in arm_pose.items():
                try:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    qpos_adr = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_adr] = value 
                except: pass

            if data.time < 1.5:
                id_fr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_right_joint_velocity")
                id_fl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_front_left_joint_velocity")
                id_rr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_right_joint_velocity")
                id_rl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_rear_left_joint_velocity")
                
                if id_fr != -1: data.ctrl[id_fr] = 0.0
                if id_fl != -1: data.ctrl[id_fl] = 0.0
                if id_rr != -1: data.ctrl[id_rr] = 0.0
                if id_rl != -1: data.ctrl[id_rl] = 0.0
            else:
                if not nav_started:
                    nav_started = True
                    
                follow_navigation_path(model, data, path_world, base_speed=2.5)

            mujoco.mj_step(model, data)
        
        viewer.sync()
            
        time_until_next_step = model.opt.timestep - (time.time() - step_start) / speed_factor
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

cv2.destroyAllWindows()