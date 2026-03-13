# Standard libraries
import os

# Third-party libraries
import osmnx as ox

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import NavSatFix

class MapGeneratorNode(Node):
    def __init__(self):
        super().__init__('map_generator')
        self.subscription_gps = self.create_subscription(NavSatFix, 'gps/fix', self.gps_callback, 10)
        self.gps_coords = None

        share_dir = get_package_share_directory('bd_nav')
        self.assets_dir = os.path.join(share_dir, "assets")

        self.remove_graph_files()

    def remove_graph_files(self):
        target_ext = ".graphml"
        file_names_list = [f for f in os.listdir(self.assets_dir) if f.endswith(target_ext)]
        for fn in file_names_list:
            file_path = os.path.join(self.assets_dir, fn)
            if os.path.exists(file_path):
                os.remove(file_path)

    def gps_callback(self, msg):
        lat, lon = msg.latitude, msg.longitude
        if self.gps_coords is None:

            self.remove_graph_files()

            self.gps_coords = (lat, lon)
            map_file_name = f'map_{lat}_{lon}.graphml'
            local_graph_path = os.path.join(self.assets_dir, map_file_name)
            self.graph_path = local_graph_path

            os.makedirs(os.path.dirname(local_graph_path), exist_ok=True)

            self.get_logger().info(f"{map_file_name} File X, Creating ...")
            G = ox.graph_from_point(self.gps_coords , dist=200, network_type="walk", simplify=False)
            for idx, node in enumerate(G.nodes(), start=1):
                G.nodes[node]['index'] = idx
            ox.save_graphml(G, filepath=self.graph_path)
            self.get_logger().info(f"{map_file_name} save path: {self.graph_path}")

            self.get_logger().info(f"{map_file_name} File load")
            self.G = ox.load_graphml(self.graph_path)

def main(args=None):
    rclpy.init(args=args)
    node = MapGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Map Generator Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
