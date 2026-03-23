import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
import scipy.interpolate as si  

MAP_FILE = 'map.png' 
RESOLUTION = 0.02  
SCAN_X = (-3, 9)   
SCAN_Y = (-1, 5)   

ROBOT_RADIUS = 0.144

def world_to_grid(x, y):
    map_width_m = SCAN_X[1] - SCAN_X[0]
    map_height_m = SCAN_Y[1] - SCAN_Y[0]
    grid_width = int(map_width_m / RESOLUTION)
    grid_height = int(map_height_m / RESOLUTION)

    px = int((x - SCAN_X[0]) / RESOLUTION)
    py = int((SCAN_Y[1] - y) / RESOLUTION)
    
    px = np.clip(px, 0, grid_width - 1)
    py = np.clip(py, 0, grid_height - 1)
    return (px, py)

def grid_to_world(px, py):
    x = px * RESOLUTION + SCAN_X[0]
    y = SCAN_Y[1] - (py * RESOLUTION)
    return (x, y)

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(grid, start, goal):
    move_directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_heap = []
    
    heapq.heappush(open_heap, (f_score[start], start))
    height, width = grid.shape

    while open_heap:
        current = heapq.heappop(open_heap)[1]
        if current == goal:
            path_pixels = []
            while current in came_from:
                path_pixels.append(current)
                current = came_from[current]
            path_pixels.append(start)
            return path_pixels[::-1]
            
        close_set.add(current)
        for dx, dy in move_directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height): continue
            if grid[neighbor[1]][neighbor[0]] == 1: continue
                
            move_cost = 1.0 if dx == 0 or dy == 0 else 1.414
            tentative_g_score = g_score[current] + move_cost
            
            if neighbor in close_set and tentative_g_score >= g_score.get(neighbor, float('inf')): continue
                
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    return None

def smooth_path(path_world, smooth_factor=0.5, num_points=100):
    unique_path = [path_world[0]]
    for p in path_world[1:]:
        if np.linalg.norm(np.array(p) - np.array(unique_path[-1])) > 0.05: 
            unique_path.append(p)
            
    if len(unique_path) < 4:
        return path_world

    x = [p[0] for p in unique_path]
    y = [p[1] for p in unique_path]

    tck, u = si.splprep([x, y], s=smooth_factor)
    
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = si.splev(u_new, tck)

    smoothed_path = list(zip(x_new, y_new))
    return smoothed_path

def get_navigation_path(start_world, goal_world):
    print("Đang đọc bản đồ và tính toán đường đi A*...")
    img_raw = cv2.imread(MAP_FILE, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        print(f"Lỗi: Không tìm thấy file '{MAP_FILE}'!")
        return None

    binary_map = (img_raw < 128).astype(np.uint8)
    inflation_pixels = int(ROBOT_RADIUS / RESOLUTION)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_pixels * 2, inflation_pixels * 2))
    inflated_grid = cv2.dilate(binary_map, kernel, iterations=1)

    start_pixel = world_to_grid(*start_world)
    goal_pixel = world_to_grid(*goal_world)

    if inflated_grid[start_pixel[1]][start_pixel[0]] == 1 or inflated_grid[goal_pixel[1]][goal_pixel[0]] == 1:
        print("Lỗi: Start hoặc Goal nằm trong vùng vật cản!")
        return None

    path_pixels = astar(inflated_grid, start_pixel, goal_pixel)
    
    if path_pixels is not None:
        path_world = [grid_to_world(p[0], p[1]) for p in path_pixels]
        
        total_dist = 0
        for i in range(len(path_world)-1):
            p1 = path_world[i]
            p2 = path_world[i+1]
            total_dist += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
        desired_spacing = 0.1 
        dynamic_num_points = max(10, int(total_dist / desired_spacing))
        
        print(f"Tổng chiều dài đường đi: {total_dist:.2f}m. Sinh ra {dynamic_num_points} waypoints.")

        # Giảm smooth_factor (từ 0.3 xuống 0.05) để B-Spline bám sát các điểm A* gốc hơn, tránh tình trạng "cắt góc" lẹm vào vật cản
        smoothed_path = smooth_path(path_world, smooth_factor=0.05, num_points=dynamic_num_points)
        return smoothed_path
    
    return None

if __name__ == "__main__":
    try:
        img_raw = cv2.imread(MAP_FILE, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise ValueError(f"Lỗi: Không tìm thấy file '{MAP_FILE}'!")

        binary_map = (img_raw < 128).astype(np.uint8)

        inflation_pixels = int(ROBOT_RADIUS / RESOLUTION)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_pixels * 2, inflation_pixels * 2))
        inflated_grid = cv2.dilate(binary_map, kernel, iterations=1)

        START_WORLD = (0.93, 0.52) 
        GOAL_WORLD = (2.0, 1.31) 

        start_pixel = world_to_grid(*START_WORLD)
        goal_pixel = world_to_grid(*GOAL_WORLD)

        path_pixels = astar(inflated_grid, start_pixel, goal_pixel)

        plt.figure(figsize=(12, 6))
        
        display_img = np.zeros((img_raw.shape[0], img_raw.shape[1], 3), dtype=np.uint8)
        display_img[binary_map == 0] = [255, 255, 255]
        display_img[inflated_grid == 1] = [180, 180, 180]
        display_img[binary_map == 1] = [0, 0, 0]

        plt.imshow(display_img, origin='upper', extent=[SCAN_X[0], SCAN_X[1], SCAN_Y[0], SCAN_Y[1]])

        if path_pixels is not None:
            path_world = [grid_to_world(p[0], p[1]) for p in path_pixels]
            
            smoothed_path_world = smooth_path(path_world, smooth_factor=0.05, num_points=50)
            
            pw_arr = np.array(path_world)
            spw_arr = np.array(smoothed_path_world)

            plt.plot(pw_arr[:, 0], pw_arr[:, 1], color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='A* Gốc (Zigzag)')
            
            plt.plot(spw_arr[:, 0], spw_arr[:, 1], color='cyan', linewidth=2.5, label='Đường Nội Suy (B-Spline)')
            
            print(f"--- THÀNH CÔNG ---")
            print(f"Đường A* gốc: {len(path_world)} điểm -> Giảm còn {len(smoothed_path_world)} waypoints nội suy.")
        else:
            print(f"Thuật toán A* báo: Không tìm thấy đường đi khả thi!")

        plt.scatter(*START_WORLD, color='green', marker='x', s=150, linewidth=3, label='Start (X Xanh)', zorder=5)
        plt.scatter(*GOAL_WORLD, color='red', marker='x', s=150, linewidth=3, label='Goal (X Đỏ)', zorder=5)

        plt.title(f"A* Navigation with B-Spline Smoothing")
        plt.xlabel("X (Meters)")
        plt.ylabel("Y (Meters)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axis('equal')
        plt.show()

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")