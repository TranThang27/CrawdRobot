import mujoco
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Thêm thư viện để lưu ảnh

# 1. Cấu hình đường dẫn và thông số
XML_PATH = "env_mujoco_2/mujoco/scene.xml" # Đường dẫn file của Thắng
RESOLUTION = 0.02  # 2cm mỗi ô
SCAN_X = (-3, 9)   # Quét từ -5m đến 5m trục X
SCAN_Y = (-1, 5)   # Quét từ -5m đến 5m trục Y

def scan_static_map(model, data, x_range, y_range, resolution):
    # Tính toán kích thước ma trận
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
            
            # Truyền geomid đã khởi tạo đúng kiểu vào đây
            dist = mujoco.mj_ray(model, data, p_start, direction, None, 1, -1, geomid)
            
            if dist > 0:
                hit_z = 5.0 - dist
                if hit_z > 0.05:
                    grid_map[i, j] = 1
        
        if i % 20 == 0: print(f"Tiến độ: {int(i/width*100)}%")

    return grid_map

# 2. Thực thi
try:
    # Load môi trường
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    # Cập nhật vị trí các vật thể trước khi quét
    mujoco.mj_forward(model, data)

    # Quét
    map_data = scan_static_map(model, data, SCAN_X, SCAN_Y, RESOLUTION)

    # --- PHẦN LƯU FILE PNG (GIỮ NGUYÊN TỈ LỆ) ---
    # Trong mô phỏng: 1 là vật cản, 0 là đường đi
    # Trong ảnh PNG: 0 là đen (vật cản), 255 là trắng (đường đi)
    # Công thức: (1 - giá trị) * 255
    png_data = ((1 - map_data.T) * 255).astype(np.uint8)
    
    # Tạo ảnh từ ma trận
    img = Image.fromarray(png_data)
    # Lật ảnh để khớp với hệ tọa độ hiển thị (origin='lower')
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Lưu file
    img.save("static_map.png")
    print("--- THÀNH CÔNG ---")
    print("Đã lưu bản đồ vào file static_map.png")

    # 3. Hiển thị (Giữ nguyên phần plot của bạn để đối chiếu)
    plt.figure(figsize=(10, 10))
    plt.imshow(map_data.T, cmap='Greys', origin='lower', 
               extent=[SCAN_X[0], SCAN_X[1], SCAN_Y[0], SCAN_Y[1]])
    plt.colorbar(label='0: Đường đi, 1: Vật cản')
    plt.title("Bản đồ tĩnh quét từ MuJoCo (Occupancy Grid Map)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    
    # Lưu file npy
    np.save("static_map.npy", map_data)
    print("Đã lưu bản đồ vào file static_map.npy")
    
    plt.show()

except Exception as e:
    print(f"Lỗi khi load môi trường: {e}")