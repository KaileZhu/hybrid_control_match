"""
Input: OccupancyMap  Current Position
Output: Frontier Points
"""


import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PointStamped

from visualization_msgs.msg import Marker, MarkerArray


# 只是为了测试
import math
import random
from tf.transformations import quaternion_from_euler


class FrontierExplorer:
    """Frontier Explorer Node"""
    def __init__(self):
        rospy.init_node("frontier_explorer")
        self.map = OccupancyGrid()
        self.current_position = None
        self.distance_obs = 0.5
    
        # 订阅地图
        rospy.wait_for_message("/map", OccupancyGrid)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        # 订阅当前位置
        rospy.wait_for_message("/tf", TFMessage)
        self.listener = tf.TransformListener()

        self.centroid_markers_pub = rospy.Publisher('/centroid_markers', MarkerArray, queue_size=10)
        # 定时器
        rospy.Timer(rospy.Duration(1), self.timer_callback)

        # 发布前沿点
        self.frontier_pub = rospy.Publisher("frontiers", PoseArray, queue_size=10)
        self.resolution = self.map.info.resolution
        print("分辨率", self.resolution)
        self.origin = self.map.info.origin
        print("坐标原点", self.origin)

        rospy.spin()


    def get_pose(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))  # (x, y, z), (x, y, z, w)
            euler = tf.transformations.euler_from_quaternion(rot)
            return trans, euler  # RPY 
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None


    def map_callback(self, map):
        self.map = map

    
    def timer_callback(self, event):
        # Input: Current_Position
        cur_position, cur_euler = self.get_pose()
        x = cur_position[0]
        y = cur_position[1]
        yaw = cur_euler[2]
        print("------------------------------------------------------------")
        print(f"-----Current Position: ({x}, {y}, {yaw})")
        # Input: OccupancyMap
        map = self.map
        """
        OccupancyMap:
        std_msgs/Header header
            uint32 seq
            time stamp
            string frame_id # 地图坐标系
        nav_msgs/MapMetaData info
            time map_load_time
            float32 resolution # 地图分辨率
            uint32 width # 地图宽度
            uint32 height   # 地图高度
            geometry_msgs/Pose origin # 地图原点
                geometry_msgs/Point position
                float64 x
                float64 y
                float64 z
                geometry_msgs/Quaternion orientation
                float64 x
                float64 y
                float64 z
                float64 w
        int8[] data
        """
        self.set_map_size_info(map)
        self.generate_frontiers()        

        self.publish_frontiers()
        print('here')

        # Output: Frontier Points
        # self.frontier_exploration_with_kmeans(np.array(map.data).reshape(map.info.height, map.info.width), (x, y))

        # # # 只是为了测试
        # frontier_points = [(1.01, 2.02), (3.03, 4.04), (5.05, 6.06)]  # K 个前沿点
        # pose_array = PoseArray()
        # pose_array.header.frame_id = 'map'
        # pose_array.header.stamp = rospy.Time.now()
        
        # for i in range(3):
        #     pose = Pose()
        #     pose.position.x = random.uniform(-3, 3)
        #     pose.position.y = random.uniform(-2.5, 3.5)

        # 这里的x和y是世界坐标系下的坐标
        frontier_points = self.frontier_exploration_with_kmeans(np.array(map.data), (x, y))
        print("前沿点的数量",len(frontier_points))
        print("前沿点为：",frontier_points)        
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = rospy.Time.now()

        for point in frontier_points:
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = 0

            # 生成随机朝向
            yaw = random.uniform(-math.pi, math.pi)
            quaternion = quaternion_from_euler(0, 0, yaw)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            pose_array.poses.append(pose)
        
        self.frontier_pub.publish(pose_array)
    

    def set_map_size_info(self, map_msg):
        """
        Get size of occupancy-grid map according to the input map and original axises.
        """
        self.res = map_msg.info.resolution
        self.occupancy_map = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        
        self.grid_max_raw = map_msg.info.height
        self.grid_max_col = map_msg.info.width
        self.map_origin_x = map_msg.info.origin.position.x
        self.map_origin_y = map_msg.info.origin.position.y

        self.grid_ori_col = int((0-self.map_origin_x)/self.res)
        self.grid_ori_raw = int((0-self.map_origin_y)/self.res)

        self.grid_cols = range(0, self.grid_max_col)
        self.grid_raws = range(0, self.grid_max_raw)
        self.safe_grid_dis = int(self.distance_obs/self.res)

    def generate_frontiers(self, method='mean_shift', expand=10, frontier_search=False):
        """
        Generate frontiers according to the occupancy map,
        ----------
        Parameters:
            method: mean_shift or group_mid
        """
        frontiers = set()
        if self.occupancy_map is None:
            rospy.LogWarning('No map info!')
            return None, None
        ##detect frontiers in two ways: first search and incremental search
        rospy.loginfo('Start detect the frontiers!')

        rospy.loginfo('Initial detection.')
        ##extract frontier points from the whole occupancy map
        for raw in self.grid_raws:
            for col in self.grid_cols:
                if self.occupancy_map[raw][col] == 0:  # Unoccupied cell
                    # Check if the cell has unexplored neighbors
                    if self._has_unexplored_neighbors(raw, col):
                        point_x = round(col*self.res+self.map_origin_x, 2)
                        point_y = round(raw*self.res+self.map_origin_y, 2)
                        frontiers.add((point_x, point_y))
        if frontier_search:
            rospy.loginfo('Expand frontiers to continue detection.')
            ##expand frontiers from the past frontiers
            for cluster in self.frontier_clusters.values():
                col_min = int((np.min(cluster['frontiers'][:, 0])-self.map_origin_x)/self.res)
                col_max = int((np.max(cluster['frontiers'][:, 0])-self.map_origin_x)/self.res)
                raw_min = int((np.min(cluster['frontiers'][:, 1])-self.map_origin_y)/self.res)
                raw_max = int((np.max(cluster['frontiers'][:, 1])-self.map_origin_y)/self.res)
                for raw in range(raw_min-expand, raw_max+expand+1):
                    for col in range(col_min-expand, col_max+expand+1):
                        if self.occupancy_map[raw][col] == 0:  # Unoccupied cell
                            # Check if the cell has unexplored neighbors
                            if self._has_unexplored_neighbors(raw, col):
                                point_x = round(col*self.res+self.map_origin_x, 2)
                                point_y = round(raw*self.res+self.map_origin_y, 2)
                                frontiers.add((point_x, point_y))
        ##cluster the set of frontiers
        if len(frontiers) > 0:
            rospy.loginfo(f'Get frontiers, total number {len(frontiers)}!')
            if method == 'mean_shift':
                self.frontier_clusters = self._mean_shift(frontiers)
                return self.frontier_clusters
            else:
                pass
        else:
            rospy.logwarn('There are no frontiers!')
            self.frontier_clusters = dict()
            return None

    def _has_unexplored_neighbors(self, row, col, grids=4):
        # Check for unexplored neighbors
        if grids == 4:
            neighbor_shift = [(0,1), (1,0), (-1,0), (0,-1)]
        else:
            neighbor_shift = [(1, 1), (0,1), (-1,1), (1,0), (-1,0), (1,-1), (0,-1), (-1,-1)]
        for c, r in neighbor_shift:
            if (row+r in self.grid_raws and col+c in self.grid_cols and
                self.occupancy_map[row+r][col+c] == -1):
                return True
        return False

    def _mean_shift(self, frontiers, bandwidth=2.0, update=True):
        """
        Use Mean Shift clustering to filter and group frontier points.
        """
        from sklearn.cluster import MeanShift
        frontier_clusters = dict()
        ##transfer list to numpy array
        frontiers = np.array(list(frontiers))
        ##initialize Mean Shift and execute clustering
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit_predict(frontiers)
        labels = ms.labels_
        ##process each cluster
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_indices = np.where(labels == label)[0]
            cluster_frontiers = frontiers[cluster_indices]
            ##calculate the centroid for the cluster of frontiers
            centroid = np.mean(cluster_frontiers, axis=0)
            if self._close_to_obstacles(centroid):
                centroid = min(cluster_frontiers, key=lambda p:math.dist(p, centroid))
            ##add new centroid and frontiers to frontier_clusters
            frontier_clusters[label] = {
                'centroid': centroid,
                'frontiers': cluster_frontiers,
            }
            
        return frontier_clusters

    def _close_to_obstacles(self, centroid):
        col = int(centroid[0]/self.res)
        row = int(centroid[1]/self.res)
        safe_grid_range = np.arange(-self.safe_grid_dis, self.safe_grid_dis+1)
        for r in safe_grid_range:
            for c in safe_grid_range:
                if row+r in self.grid_raws and col+c in self.grid_cols:
                    if self.occupancy_map[row+r][col+c] == 1:
                        return True
        return False


    def publish_frontiers(self):
        centroid_markers_msg = MarkerArray()
        ## prepare the message type
        for id, cluster in self.frontier_clusters.items():
            frontiers = cluster['frontiers']
            centroid = cluster['centroid']
            # Set the frontier markers
            centroid_markers_msg.markers.append(self._set_frontier_marker_msg(id, centroid))

        # self.frontiers_pub.publish(frontiers_msg)
        self.centroid_markers_pub.publish(centroid_markers_msg)

        return centroid_markers_msg

    def _set_frontier_marker_msg(self, id, centroid):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "my_namespace"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = centroid[0]
        marker.pose.position.y = centroid[1]
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration(10.0)
        return marker


    def in_bounds(self, x, y, grid):
        """Check if a coordinate is within the grid."""
        # return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
        return 0 <= x < int(math.sqrt(len(grid))) and 0 <= y < int(math.sqrt(len(grid)))


    def find_frontiers(self, grid):
        """遍历寻找frontier点"""
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 四个移动方向
        frontiers = []  # 前沿点重设为空列表
        # rows = grid.shape[0]
        # cols = grid.shape[1]
        # for r in range(rows):
        #     for c in range(cols):
        #         if grid[r][c] == 0:  # 如果该点是已经被探索过的点

        for r in range(int(math.sqrt(len(grid)))):
            for c in range(int(math.sqrt(len(grid)))):
                if grid[r * c] == 0:  # 如果该点是已经被探索过的点
                    for dx, dy in DIRS:
                        new_x, new_y = r + dx, c + dy  # 判断周围四个点位的情况
                        # 如果周围的点在地图内，且未被探索过，且这个点不是墙壁
                        if self.in_bounds(new_x, new_y, grid) and grid[new_x * new_y] == -1 and grid[new_x * new_y] != 100:
                            new_x, new_y = self.map_to_world(new_x, new_y)
                            frontiers.append([new_x, new_y])
        return np.array(frontiers)
   

    def frontier_exploration_with_kmeans(self, grid, start_pos, min_num_clusters=3, max_num_clusters=10):
        """使用KMeans算法进行前沿探索"""
        # rows, cols = grid.shape
        # explored = np.zeros((rows, cols), dtype=bool)
        current_world_x, current_world_y = start_pos[0], start_pos[1]
        current_map_x, current_map_y = self.world_to_map(current_world_x, current_world_y)
        # explored[current_x][current_y] = True
        
        # 实时绘图模块
        # plt.imshow(grid, cmap='gray', vmin=0, vmax=1)
        # plt.pause(0.1)
        # plt.title("Exploration Process")

        frontiers_points = self.find_frontiers(grid)  # 得到的是世界坐标系下的前沿点
        print("搜索到的前言点数量",len(frontiers_points))
        print("搜索到的前言点展示", frontiers_points)

        if len(frontiers_points) == 0:
            print("-----No frontiers found!")
            
        else:
        # if len(frontiers_points) < min_num_clusters:
            # frotier点选择模块
            # frontier点数量太少，选择离当前位置最近的frontier点
            closest_frontier = None
            min_distance = float('inf')
            for frontier in frontiers_points:
                distance = np.sqrt((current_world_x - frontier[0]) ** 2 + (current_world_y - frontier[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_frontier = frontier
            frontiers_points = [closest_frontier]
            print(f"-----Next step: Moving towards frontier at {closest_frontier}")
            print(frontiers_points)
            print("Too few frontiers found!")
            
            
        # if len(frontiers_points) >= min_num_clusters:
        #     # 使用K-Means进行聚类
        #     best_error = np.inf  # 初始化最小误差为无穷大
        #     best_kmeans = None
        #     for i in range(min_num_clusters, max_num_clusters):
        #         kmeans = KMeans(n_clusters=i).fit(frontiers_points)
        #         inertia = kmeans.inertia_  # 获取当前聚类的惯性（簇内平方和，可作为误差衡量）
        #         if inertia < best_error:
        #             best_error = inertia
        #             best_kmeans = kmeans

        #     centers = best_kmeans.cluster_centers_.astype(int)
        #     print(centers)

        #     # 选择距离当前位置最近的聚类中心作为下一个探索点
        #     closest_center = None
        #     min_distance = float('inf')
        #     for center in centers:
        #         distance = np.sqrt((current_world_x - center[0]) ** 2 + (current_world_y - center[1]) ** 2)
        #         if distance < min_distance:
        #             min_distance = distance
        #             closest_center = center
        #     frontiers_points = [closest_center]
        #     print(f"-----Next step: Moving towards cluster center at {closest_center}")


        # plt.plot(current_y, current_x, 'ro')  # 在地图上标记已探索位置，红色圆点
        # plt.pause(0.1)    
        # plt.show()

        print("------An exploration has been completed!")    
        return frontiers_points


    def world_to_map(self, x, y):
        # 将世界坐标转换为栅格地图坐标
        map_x = int((x - self.origin.position.x) / self.resolution)
        map_y = int((y - self.origin.position.y) / self.resolution)
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        # 将栅格地图坐标转换为世界坐标
        x = map_x * self.resolution * 0.1 + self.origin.position.x
        y = map_y * self.resolution * 0.1 + self.origin.position.y
        return x, y 



if __name__ == "__main__":
    rospy.init_node("frontier_explorer")
    explorer = FrontierExplorer()
    # while not rospy.is_shutdown():
    #     trans, euler = explorer.get_pose()
    #     # [-0.03651428222656249, 0.05396270751953125, -8.673617379884035e-19] (-2.073244747820623e-20, 1.921909275271493e-21, 0.02196865731557298)yaw偏航角 rad

    #     print(trans, euler)
    #     rospy.sleep(1)
