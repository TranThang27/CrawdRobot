import mujoco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  

XML_PATH = "env_mujoco_2/mujoco/scene.xml" 
RESOLUTION = 0.02  
SCAN_X = (-3, 9)   
SCAN_Y = (-1, 5)   

def scan_static_map(model, data, x_range, y_range, resolution):
    width = int((x_range[1] - x_range[0]) / resolution)
    height = int((y_range[1] - y_range[0]) / resolution)
    grid_map = np.zeros((width, height))

    direction = np.array([0, 0, -1], dtype=np.float64)
    print(f"Bắt đầu quét map: {width}x{height} ô...")

    geomid = np.zeros(1, dtype=np.int32) 

    for i in range(width):
        for j in range(height):
            x = x_range[0] + i * resolution
            y = y_range[0] + j * resolution
            
            p_start = np.array([x, y, 5.0], dtype=np.float64)
            direction = np.array([0, 0, -1], dtype=np.float64)
            
            dist = mujoco.mj_ray(model, data, p_start, direction, None, 1, -1, geomid)
            
            if dist > 0:
                hit_z = 5.0 - dist
                if hit_z > 0.05:
                    grid_map[i, j] = 1
        
        if i % 20 == 0: print(f"Tiến độ: {int(i/width*100)}%")

    return grid_map

try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)

    map_data = scan_static_map(model, data, SCAN_X, SCAN_Y, RESOLUTION)

    png_data = ((1 - map_data.T) * 255).astype(np.uint8)
    
    img = Image.fromarray(png_data)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    img.save("static_map.png")
    print("--- THÀNH CÔNG ---")
    print("Đã lưu bản đồ vào file static_map.png")

    plt.figure(figsize=(10, 10))
    plt.imshow(map_data.T, cmap='Greys', origin='lower', 
               extent=[SCAN_X[0], SCAN_X[1], SCAN_Y[0], SCAN_Y[1]])
    plt.colorbar(label='0: Đường đi, 1: Vật cản')
    plt.title("Bản đồ tĩnh quét từ MuJoCo (Occupancy Grid Map)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    
    np.save("static_map.npy", map_data)
    print("Đã lưu bản đồ vào file static_map.npy")
    
    plt.show()

except Exception as e:
    print(f"Lỗi khi load môi trường: {e}")