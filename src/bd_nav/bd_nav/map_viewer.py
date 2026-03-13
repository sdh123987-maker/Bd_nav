# Standard libraries
import os
import math
import json
import threading
import base64
from io import BytesIO

# Third-party libraries
import osmnx as ox
import matplotlib.pyplot as plt
import requests
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

class MapViewer(Node):
    def __init__(self):
        super().__init__('map_viewer')

        self.create_subscription(String, 'route_data', self.route_callback, 10)

        self.map_image_publisher = self.create_publisher(String, '/map_image', 10)
        self.map_image_wo_labels_publisher = self.create_publisher(String, '/map_image_wo_labels', 10)

        # 1. Settings
        self.zoom = 18
        self.tile_size = 256
        self.graph_dist_m = 300

        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.map_loaded = False

    def timer_callback(self):
        if not self.map_loaded:
            try:
                # 2. Load graph
                share_dir = get_package_share_directory('bd_nav')
                assets_dir = os.path.join(share_dir, "assets")
                target_ext = ".graphml"
                file_names_list = [f for f in os.listdir(assets_dir) if f.endswith(target_ext)]
                self.local_graph_path = os.path.join(assets_dir, file_names_list[0])
                self.G = ox.load_graphml(self.local_graph_path)
                self.assign_node_indices()

                # 3. Compute center
                self.center_lon, self.center_lat = self.calculate_center()
                self.center_x, self.center_y = self.deg2num(self.center_lat, self.center_lon, self.zoom)
                self.tile_range = self.calculate_tile_range()

                # 4. Fetch tile images
                self.map_image = self.fetch_osm_tiles()

                # 5. Visualization
                self.visualize()

                # Start file watcher after visualization
                threading.Thread(target=self.start_file_watcher, daemon=True).start()
                
                self.map_loaded = True
            except Exception as e:
                self.get_logger().error(f"map_viewer error: {e}") 

    def route_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.visualize(route_data=data)
        except Exception as e:
            self.get_logger().error(f"[route_callback] Parsing error: {e}")

    def start_file_watcher(self):
        class Handler(FileSystemEventHandler):
            def __init__(self, node):
                self.node = node

            def on_modified(self, event):
                if event.src_path.endswith('.graphml'):
                    try:
                        self.node.G = ox.load_graphml(self.node.local_graph_path)
                        self.node.assign_node_indices()
                        self.node.get_logger().info("map change detected ... Revisualizing")
                        self.node.visualize()
                    except Exception as e:
                        self.node.get_logger().error(f"map reload error: {e}")

        observer = Observer()
        handler = Handler(self)
        observer.schedule(handler, path=os.path.dirname(self.local_graph_path), recursive=False)
        observer.start()

    def assign_node_indices(self):
        for idx, node in enumerate(self.G.nodes, start=1):
            self.G.nodes[node]['index'] = idx

    def calculate_center(self):
        xs = [data['x'] for _, data in self.G.nodes(data=True)]
        ys = [data['y'] for _, data in self.G.nodes(data=True)]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def deg2num(self, lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    def num2deg(self, xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def meters_per_tile(self, lat, zoom):
        return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

    def calculate_tile_range(self):
        tile_m = self.meters_per_tile(self.center_lat, self.zoom)
        return min(math.ceil(self.graph_dist_m / tile_m), 3)

    def fetch_osm_tiles(self):
        headers = {'User-Agent': 'MyResearchBot/1.0 (user@yourdomain.com)'}
        size = (2 * self.tile_range + 1) * self.tile_size
        map_image = Image.new('RGB', (size, size))

        for dx in range(-self.tile_range, self.tile_range + 1):
            for dy in range(-self.tile_range, self.tile_range + 1):
                x = self.center_x + dx
                y = self.center_y + dy
                url = f"https://tile.openstreetmap.org/{self.zoom}/{x}/{y}.png"
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    tile = Image.open(BytesIO(response.content))
                    map_image.paste(tile, ((dx + self.tile_range) * self.tile_size, (dy + self.tile_range) * self.tile_size))
                except Exception as e:
                    self.get_logger().warn(f"Tile {self.zoom}/{x}/{y} Load Fail: {e}")
        return map_image
    

    def publish_axes_as_png(self, publisher, ax):
        fig = ax.figure

        fig.canvas.draw()

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        try:
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches=extent, pad_inches=0)
            buf.seek(0)
            img_bytes = buf.read()
            base64_img = base64.b64encode(img_bytes).decode('utf-8')

            msg = String()
            msg.data = base64_img
            publisher.publish(msg)
            self.get_logger().info("[visualize] Axes-only image transfer complete")
        except Exception as e:
            self.get_logger().error(f"[visualize] Axes-only image transfer error: {e}")

    def visualize(self, route_data=None):
        # Compute geo extent of tiles
        top_left_lat, top_left_lon = self.num2deg(self.center_x - self.tile_range, self.center_y - self.tile_range, self.zoom)
        bottom_right_lat, bottom_right_lon = self.num2deg(self.center_x + self.tile_range + 1, self.center_y + self.tile_range + 1, self.zoom)

        fig1, ax1 = plt.subplots(figsize=(10, 10))
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax1.imshow(self.map_image, 
                  extent=[top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat],
                  alpha=0.7)
        ax2.imshow(self.map_image, 
                  extent=[top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat],
                  alpha=0.7)

        # Draw graph (draw nodes, but not edges)
        ox.plot_graph(
            self.G,
            ax=ax1,
            show=False,
            close=False,
            node_size=0.1,
            edge_color='None',
            edge_linewidth=0.5,
            bgcolor=None
        )
        ox.plot_graph(
            self.G,
            ax=ax2,
            show=False,
            close=False,
            node_size=0.1,
            edge_color='None',
            edge_linewidth=0.5,
            bgcolor=None
        )

        # If route_data provided, visualize routes, waypoints, and related node indices
        if route_data:
            try:
                default_route = route_data.get("default_route", [])
                custom_route = route_data.get("route", [])
                waypoints = route_data.get("waypoints", [])
                # node_indices = route_data.get("route_node_indices", {})  # currently unused

                # Separate labeling for route nodes (darkgreen) and adjacent nodes (purple)
                route_nodes = {nid for nid in custom_route if nid in self.G.nodes}

                # Identify start (first) node of the route and hide its label
                start_node = None
                if custom_route:
                    first = custom_route[0]
                    if first in route_nodes:
                        start_node = first

                neighbor_nodes = set()
                for nid in route_nodes:
                    neighbor_nodes.update(self.G.neighbors(nid))
                neighbor_nodes -= route_nodes  # Exclude route nodes

                def _label_nodes(nodes, color, ax):
                    for node_id in nodes:
                        if node_id in self.G.nodes:
                            node_info = self.G.nodes[node_id]
                            x, y = node_info['x'], node_info['y']
                            label = node_info.get('index', node_id)
                            ax.text(
                                x, y, str(label),
                                fontsize=7, color=color, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1),
                                zorder=50  # Keep labels visible
                            )

                # Label route nodes except the start node
                route_nodes_to_label = route_nodes - {start_node} if start_node is not None else route_nodes
                _label_nodes(route_nodes_to_label, 'darkgreen', ax1)
                _label_nodes(neighbor_nodes, 'purple', ax1)

                # Draw blue default route
                if default_route:
                    path_xs = [self.G.nodes[n]["x"] for n in default_route if n in self.G.nodes]
                    path_ys = [self.G.nodes[n]["y"] for n in default_route if n in self.G.nodes]
                    ax1.plot(path_xs, path_ys, color='blue', linewidth=8, alpha=0.8) #, label='Default Route'
                    ax2.plot(path_xs, path_ys, color='blue', linewidth=8, alpha=0.8) #, label='Default Route'

                # Draw red custom route
                if custom_route:
                    path_xs = [self.G.nodes[n]["x"] for n in custom_route if n in self.G.nodes]
                    path_ys = [self.G.nodes[n]["y"] for n in custom_route if n in self.G.nodes]
                    ax1.plot(path_xs, path_ys, color='red', linewidth=8, alpha=0.8) #, label='Custom Route'
                    ax2.plot(path_xs, path_ys, color='red', linewidth=8, alpha=0.8) #, label='Custom Route'

                    # Mark start point
                    start_node = custom_route[0]
                    if start_node in self.G.nodes:
                        sx = self.G.nodes[start_node]['x']
                        sy = self.G.nodes[start_node]['y']
                        ax1.plot(
                                sx, sy, linestyle="none", marker="o",
                                markersize=25,
                                markerfacecolor="#E5E4E2",
                                markeredgecolor="black",
                                markeredgewidth=1
                                )
                        ax2.plot(
                                sx, sy, linestyle="none", marker="o",
                                markersize=25,
                                markerfacecolor="#E5E4E2",
                                markeredgecolor="black",
                                markeredgewidth=1
                                )
                    # Mark destination point
                    dest_node = custom_route[-1]
                    if dest_node in self.G.nodes:
                        sx = self.G.nodes[dest_node]['x']
                        sy = self.G.nodes[dest_node]['y']
                        # ax1.plot(sx, sy, linestyle="none", marker="*",
                        #         markersize=35,
                        #         markerfacecolor="#E5E4E2",
                        #         markeredgecolor="black",
                        #         markeredgewidth=1
                        #         )
                        ax2.plot(sx, sy, linestyle="none", marker="*",
                                markersize=35,
                                markerfacecolor="#E5E4E2",
                                markeredgecolor="black",
                                markeredgewidth=1
                                )
                # Draw waypoints
                for i, node_id in enumerate(waypoints):
                    if node_id in self.G.nodes:
                        x = self.G.nodes[node_id]['x']
                        y = self.G.nodes[node_id]['y']
                        label = 'Waypoint' if i == 0 else None  # Show in legend only once
                        # ax1.plot(x, y, marker='o', markersize=10, color='red', alpha=0.9) # label=label
                        # ax2.plot(x, y, marker='o', markersize=10, color='red', alpha=0.9) # label=label
                        # ax1.plot(x, y, linestyle="none", marker="s",
                        #         markersize=25,
                        #         markerfacecolor="#E5E4E2",
                        #         markeredgecolor="black",
                        #         markeredgewidth=1
                        #         )
                        ax2.plot(x, y, linestyle="none", marker="s",
                                markersize=25,
                                markerfacecolor="#E5E4E2",
                                markeredgecolor="black",
                                markeredgewidth=1
                                )

                ax1.legend()
                ax2.legend()

            except Exception as e:
                self.get_logger().error(f"[visualize] Visualizing route data error: {e}")
        ax1.axis('off')
        fig1.tight_layout()
        ax2.axis('off')
        fig2.tight_layout()

        self.publish_axes_as_png(self.map_image_publisher, ax1)
        self.publish_axes_as_png(self.map_image_wo_labels_publisher, ax2)

        plt.close(fig1)
        plt.close(fig2)


def main(args=None):
    rclpy.init(args=args)
    node = MapViewer()
    try:
        rclpy.spin(node)  # Keep node alive
    except KeyboardInterrupt:
        node.get_logger().info("Shut Down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
