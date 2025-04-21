import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt





def in_bounds(x, y, grid):
    """Check if a coordinate is within the grid."""
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]


def find_frontiers(grid, explored):
    """遍历寻找frontier点"""
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 四个移动方向
    frontiers = []
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if explored[r][c]:
                for dx, dy in DIRS:
                    new_x, new_y = r + dx, c + dy
                    if in_bounds(new_x, new_y, grid) and not explored[new_x][new_y] and grid[new_x][new_y] == 0:
                        frontiers.append([new_x, new_y])
    return np.array(frontiers)


def frontier_exploration_with_kmeans(grid, start_pos, num_clusters=3):
    """使用KMeans算法进行前沿探索"""
    rows, cols = grid.shape
    explored = np.zeros((rows, cols), dtype=bool)
    current_x, current_y = start_pos[0], start_pos[1]
    explored[current_x][current_y] = True

    plt.imshow(grid, cmap='gray', vmin=0, vmax=1)  # 可视化地图，0为白色（可通行），1为黑色（障碍物），初始时全部为未探索（灰色）
    plt.pause(0.1)
    plt.title("Exploration Process")
    while True:
        frontiers = find_frontiers(grid, explored)
        if len(frontiers) == 0:
            break
        # if len(frontiers) < num_clusters:
        else:
            # 前沿点数量小于聚类数量，直接选择离当前位置最近的前沿点
            closest_frontier = None
            min_distance = float('inf')
            for frontier in frontiers:
                distance = np.sqrt((current_x - frontier[0]) ** 2 + (current_y - frontier[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_frontier = frontier
            print(f"Next step: Moving towards frontier at {closest_frontier}")
            current_x, current_y = closest_frontier
        # else:
        #     # 使用KMeans进行聚类
        #     kmeans = KMeans(n_clusters=num_clusters).fit(frontiers)
        #     centers = kmeans.cluster_centers_.astype(int)
        #     # 选择距离当前位置最近的聚类中心作为下一个探索点
        #     closest_center = None
        #     min_distance = float('inf')
        #     for center in centers:
        #         distance = np.sqrt((current_x - center[0]) ** 2 + (current_y - center[1]) ** 2)
        #         if distance < min_distance:
        #             min_distance = distance
        #             closest_center = center
        #     print(f"Next step: Moving towards cluster center at {closest_center}")
        #     current_x, current_y = closest_center

        explored[current_x][current_y] = True
        plt.plot(current_y, current_x, 'ro')  # 在地图上标记已探索位置，红色圆点
        plt.pause(0.1)
    print("Exploration completed!")
    plt.show()
    return explored


def grid_map(size, num_obstacles):
    """
    生成指定大小和障碍物数量的栅格地图
    :param size: 地图的边长（正方形地图）
    :param num_obstacles: 要设置的障碍物数量
    :return: 生成的栅格地图（二维numpy数组，0表示可通行，1表示障碍物）
    """
    map_width = size
    map_height = size
    grid_map = np.zeros((map_height, map_width), dtype=int)

    # 随机设置障碍，确保不重复设置
    obstacle_count = 0
    obstacle_positions = set()
    while obstacle_count < num_obstacles:
        x = np.random.randint(0, map_width)
        y = np.random.randint(0, map_height)
        if (x, y) not in obstacle_positions:
            grid_map[y, x] = 1
            obstacle_positions.add((x, y) )
            obstacle_count += 1

    return grid_map


if __name__ == "__main__":
    start_pos = (0, 0)
    size = 10
    num_obstacles = 20  # 可根据需要调整障碍物数量
    grid_map_data = grid_map(size, num_obstacles)
    explored_result = frontier_exploration_with_kmeans(grid_map_data, start_pos)