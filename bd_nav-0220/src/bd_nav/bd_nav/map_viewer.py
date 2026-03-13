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
import matplotlib.patheffects as pe
import requests
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from bd_nav.config import RENDER_DIST_M

class MapViewer(Node):
    def __init__(self):
        super().__init__('map_viewer')

        self.create_subscription(String, 'route_data', self.route_callback, 10)
        self.create_subscription(NavSatFix, 'gps/fix', self.gps_callback, 10)

        self.map_image_publisher = self.create_publisher(String, '/map_image', 10)
        self.map_image_wo_labels_publisher = self.create_publisher(String, '/map_image_wo_labels', 10)

        # 1. Settings
        self.zoom = 18
        self.tile_size = 256
        self.graph_dist_m = RENDER_DIST_M  # visualization only (tile extent)

        self.gps_coords = None  # (lat, lon); render center when available
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.map_loaded = False
        self.visible_node_ids = None  # 필요한 경우 필터링용으로 사용

    def gps_callback(self, msg):
        self.gps_coords = (msg.latitude, msg.longitude)
        if not getattr(self, "map_loaded", False):
            return
        # Recenter map on GPS and redraw
        self._update_center_and_redraw()

    def _update_center_and_redraw(self):
        if self.gps_coords is None:
            return
        self.center_lat, self.center_lon = self.gps_coords[0], self.gps_coords[1]
        self.center_x, self.center_y = self.deg2num(self.center_lat, self.center_lon, self.zoom)
        self.tile_range = self.calculate_tile_range()
        self.map_image = self.fetch_osm_tiles()
        self.visualize()

    def get_render_center(self):
        """Return (lon, lat) for tile/plot center: use GPS if available, else graph centroid."""
        if self.gps_coords is not None:
            return (self.gps_coords[1], self.gps_coords[0])  # (lon, lat)
        return self.calculate_center()

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

                # 3. Compute center (GPS if available, else graph centroid)
                self.center_lon, self.center_lat = self.get_render_center()
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
        """Number of tiles from center to edge; extent ≈ (2*tile_range+1) * tile_m meters."""
        tile_m = self.meters_per_tile(self.center_lat, self.zoom)
        # Threshold so small RENDER_DIST_M → tile_range=1; larger → ceil(graph_dist_m/tile_m)
        if self.graph_dist_m <= 230:
            tile_range = 1
        else:
            tile_range = math.ceil(self.graph_dist_m / tile_m)
        return min(tile_range, 3)

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
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        try:
            buf = BytesIO()
            # Save full figure (axes already [0,0,1,1]); no tight crop to avoid content-based margins
            fig.savefig(buf, format='png', dpi=125)
            buf.seek(0)
            img_bytes = buf.read()
            base64_img = base64.b64encode(img_bytes).decode('utf-8')

            msg = String()
            msg.data = base64_img
            publisher.publish(msg)
            self.get_logger().info("[visualize] Axes-only image transfer complete")
        except Exception as e:
            self.get_logger().error(f"[visualize] Axes-only image transfer error: {e}")

    # =========================================================================
    # 경로 노드 정보를 받아, 경로 노드면 테두리를 초록색으로 칠하도록 인자 추가됨
    # =========================================================================
    def draw_node_indices(self, ax, route_nodes=None):
        """모든 노드의 index 번호를 크게/고대비로 표시 (10m 이내 중복 제거)."""
        if route_nodes is None:
            route_nodes = set()
        else:
            route_nodes = set(route_nodes)

        visible = getattr(self, "visible_node_ids", None)
        drawn_points = []

        # 10m 이내 겹침 방지에서 경로 노드가 누락되지 않고 우선 표시되도록 먼저 그리도록 정렬
        nodes_list = list(self.G.nodes(data=True))
        nodes_list.sort(key=lambda x: x[0] in route_nodes, reverse=True)

        for node_id, data in nodes_list:
            if visible is not None and node_id not in visible:
                continue

            x, y = data.get('x'), data.get('y')
            idx = data.get('index')
            if x is None or y is None or idx is None:
                continue

            is_route_node = node_id in route_nodes

            # 10m 이내 겹침 방지 거리 계산
            is_too_close = False
            for (dx, dy) in drawn_points:
                dist_y = (y - dy) * 111320.0
                dist_x = (x - dx) * 111320.0 * math.cos(math.radians((y + dy) / 2.0))
                dist_m = math.hypot(dist_x, dist_y)

                if dist_m < 3.0:  
                    is_too_close = True
                    break
            
            if is_too_close:
                continue
                
            drawn_points.append((x, y))

            # 경로 위에 있는 노드면 테두리를 초록색(#00FF00)과 두껍게 적용
            edge_color = "#00FF00" if is_route_node else "white"
            line_width = 2.5 if is_route_node else 1.0
            z_order = 10 if is_route_node else 8  # 경로 마커를 위로

            ax.scatter([x], [y],
                       s=500, c="#111111", edgecolors=edge_color,
                       linewidths=line_width, alpha=0.95, zorder=z_order)

            t = ax.text(x, y, str(idx),
                        fontsize=16, fontweight='bold',
                        color='white', ha='center', va='center', zorder=z_order+1)

            t.set_path_effects([
                pe.Stroke(linewidth=2.5, foreground='black'),
                pe.Normal()
            ])

    def visualize(self, route_data=None):
        # Compute geo extent of tiles
        top_left_lat, top_left_lon = self.num2deg(self.center_x - self.tile_range, self.center_y - self.tile_range, self.zoom)
        bottom_right_lat, bottom_right_lon = self.num2deg(self.center_x + self.tile_range + 1, self.center_y + self.tile_range + 1, self.zoom)

        fig1, ax1 = plt.subplots(figsize=(16, 16))
        fig2, ax2 = plt.subplots(figsize=(16, 16))
        ax1.set_position([0, 0, 1, 1])  # fill canvas, no margins
        ax2.set_position([0, 0, 1, 1])
        ax1.imshow(self.map_image,
                  extent=[top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat],
                  alpha=0.7, aspect='auto')
        ax2.imshow(self.map_image,
                  extent=[top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat],
                  alpha=0.7, aspect='auto')

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

        # 경로 노드 정보를 `draw_node_indices`에 넘겨주기 위해 먼저 파싱합니다.
        route_nodes = []
        if route_data:
            try:
                custom_route = route_data.get("route", [])
                def _in_graph(n):
                    try:
                        return n in self.G.nodes
                    except TypeError:
                        return any(nn == n for nn in self.G.nodes)
                route_nodes = [nid for nid in custom_route if _in_graph(nid)]
            except Exception as e:
                self.get_logger().error(f"[visualize] Extracting route nodes error: {e}")

        # =====================================================================
        # 파싱된 경로 정보를 바탕으로 전체 노드 인덱스 마커 그리기 (경로 노드 초록색 처리)
        # =====================================================================
        self.draw_node_indices(ax1, route_nodes=route_nodes)

        # If route_data provided, visualize routes, waypoints
        if route_data:
            try:
                default_route = route_data.get("default_route", [])
                custom_route = route_data.get("route", [])
                waypoints = route_data.get("waypoints", [])
                
                # 삭제됨: 기존의 neighbor_nodes, _label_nodes (darkgreen/purple) 표시 로직

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
                                markerfacecolor="#E5E4E2",        # 내부 색
                                markeredgecolor="black",        # 테두리 색
                                markeredgewidth=1               # 테두리 두께
                                )
                        ax2.plot(
                                sx, sy, linestyle="none", marker="o",
                                markersize=25,
                                markerfacecolor="#E5E4E2",        # 내부 색
                                markeredgecolor="black",        # 테두리 색
                                markeredgewidth=1               # 테두리 두께
                                )
                    # Mark destination point
                    dest_node = custom_route[-1]
                    if dest_node in self.G.nodes:
                        sx = self.G.nodes[dest_node]['x']
                        sy = self.G.nodes[dest_node]['y']
                        ax1.plot(sx, sy, linestyle="none", marker="*",
                                markersize=35,
                                markerfacecolor="#E5E4E2",        # 내부 색
                                markeredgecolor="black",        # 테두리 색
                                markeredgewidth=1               # 테두리 두께
                                )
                        ax2.plot(sx, sy, linestyle="none", marker="*",
                                markersize=35,
                                markerfacecolor="#E5E4E2",        # 내부 색
                                markeredgecolor="black",        # 테두리 색
                                markeredgewidth=1               # 테두리 두께
                                )
                # Draw waypoints
                for i, node_id in enumerate(waypoints):
                    if node_id in self.G.nodes:
                        x = self.G.nodes[node_id]['x']
                        y = self.G.nodes[node_id]['y']
                        label = 'Waypoint' if i == 0 else None  # Show in legend only once
                        ax1.plot(x, y, marker='o', markersize=10, color='red', alpha=0.9) # label=label
                        ax2.plot(x, y, marker='o', markersize=10, color='red', alpha=0.9) # label=label

                ax1.legend()
                ax2.legend()

            except Exception as e:
                self.get_logger().error(f"[visualize] Visualizing route data error: {e}")
        ax1.axis('off')
        ax2.axis('off')
        # Plot range = tile extent only (no white margin from graph bbox)
        ax1.set_xlim(top_left_lon, bottom_right_lon)
        ax1.set_ylim(bottom_right_lat, top_left_lat)
        ax1.set_aspect('auto')
        ax1.set_position([0, 0, 1, 1])
        ax2.set_xlim(top_left_lon, bottom_right_lon)
        ax2.set_ylim(bottom_right_lat, top_left_lat)
        ax2.set_aspect('auto')
        ax2.set_position([0, 0, 1, 1])

        self.publish_axes_as_png(self.map_image_publisher, ax1)
        # self.publish_axes_as_png(self.map_image_wo_labels_publisher, ax2)

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
