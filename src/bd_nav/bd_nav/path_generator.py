# Standard libraries
import json
import math
import os
import re

# Third-party libraries
import networkx as nx
import osmnx as ox
import utm
from geopy.distance import distance
from osmnx import features

# ROS 2 libraries
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point as ROSPoint
from rclpy.node import Node
from std_msgs.msg import String

class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__('path_generator')

        self.original_destination_name = None
        self.original_waypoints = []
        self.ranked_destinations = []
        self.destination_name = None      # string/name or node_id string
        self.dest_node = None             # integer node_id (when resolved)
        self.prefer_path_function = None
        self.destination_ready = False
        self.path_function_ready = False
        self.latest_route = []  # store latest route

        self.subscription_output = self.create_subscription(
            String, 'user_output', self.destination_callback, 10)
        self.subscription_function = self.create_subscription(
            String, 'prefer_path_function', self.prefer_path_callback, 10)
        self.subscription_path_reply = self.create_subscription(
            String, 'path_reply', self.path_reply_callback, 10)
        self.subscription_eval_output = self.create_subscription(
            String, 'eval_output', self.eval_output_callback, 10)
        
        self.path_request_pub = self.create_publisher(String, 'path_request', 10)
        self.route_data_pub = self.create_publisher(String, 'route_data', 10)

        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Path Generator Node has been started.')

        self.G = None
        self.start_coords = None
        self.map_loaded = False

        # waypoints/avoidance
        self.waypoints = []        # always keep as a list of node_id(int) (overwritten after resolution in callbacks)
        self.avoidance = []        # original tokens (for logs)
        self.avoid_nodes = set()   # node_id set
        self.virtual_edges = []    # list of (u, v) virtual edges (added to G.copy() during routing)

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _gc_distance_m(lat1, lon1, lat2, lon2):
        """Haversine distance in meters."""
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dl   = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def _clean_token(x: str) -> str:
        return str(x).strip().strip('"').strip("'")

    @staticmethod
    def _geom_center_latlon(geom):
        """If geometry is a Point, return it as-is; otherwise use representative_point (avoid warnings)."""
        if geom.geom_type == 'Point':
            return geom.y, geom.x
        p = geom.representative_point()
        return p.y, p.x

    def calculate_center(self):
        xs = [data['x'] for _, data in self.G.nodes(data=True)]
        ys = [data['y'] for _, data in self.G.nodes(data=True)]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    
    def process_graph(self, map_file_name):
        dist = 200
        share_dir = get_package_share_directory('bd_nav')
        local_graph_path = os.path.join(share_dir, 'assets', map_file_name)
        if not os.path.exists(local_graph_path):
            self.get_logger().error(f"{map_file_name} File X: {local_graph_path}")
            raise FileNotFoundError(f"{map_file_name} not found.")
        self.get_logger().info(f"{map_file_name} loading: {local_graph_path}")
        self.G = ox.load_graphml(local_graph_path)

        tags = {"name": True}
        self.center_lon, self.center_lat = self.calculate_center()
        self.center_point = (self.center_lat, self.center_lon)
        north, south, east, west = ox.utils_geo.bbox_from_point(self.center_point, dist=dist)
        bbox = (north, south, east, west)
        self.named_features = features.features_from_bbox(bbox=bbox, tags=tags)

        # Build reverse mapping index → node_id from G
        self.index_to_node_id = {}
        for node in self.G.nodes:
            idx = self.G.nodes[node].get('index')
            if idx is not None:
                self.index_to_node_id[int(idx)] = int(node)

    # ---------------------------
    # ROS callbacks
    # ---------------------------
    def timer_callback(self):
        if not self.map_loaded:
            try:
                share_dir = get_package_share_directory('bd_nav')
                assets_dir = os.path.join(share_dir, "assets")
                target_ext = ".graphml"
                file_names_list = [f for f in os.listdir(assets_dir) if f.endswith(target_ext)]
                file_name = file_names_list[0]
                self.process_graph(file_name)
                fn_split = file_name.replace(".graphml", "").split("_")
                lat = float(fn_split[1])
                lon = float(fn_split[2])

                self.start_coords = (lat, lon)
                self.start_node = ox.distance.nearest_nodes(
                    self.G, X=self.start_coords[1], Y=self.start_coords[0])

                self.start_utm_x, self.start_utm_y, self.zone_number, self.zone_letter = utm.from_latlon(
                    self.start_coords[0], self.start_coords[1]
                )
                self.map_loaded = True
            except Exception as e:
                self.get_logger().error(f"path_generator error: {e}") 

    def path_reply_callback(self, msg):
        try:
            reply = msg.data.strip().lower()
            if reply == 'yes':
                if self.latest_route:
                    self.get_logger().info("Confirmed route publishing.")
                else:
                    self.get_logger().warn("No saved route to publish.")
            else:
                self.get_logger().info("Route not confirmed. Republishing path request.")
        except Exception as e:
            self.get_logger().error(f"path_reply_callback error: {e}")
    
    def eval_output_callback(self, msg):
        try:
            data = json.loads(msg.data) if msg and msg.data else {}

            # YES handling
            if not data or (str(data.get("confirm", "")).lower() == "yes"):
                if getattr(self, "latest_route", None):
                    self.get_logger().info("Confirmed route publishing.")
                else:
                    self.get_logger().warn("No saved route to publish.")
                return

            # Replace destination
            if "destination" in data:
                dest_raw = data["destination"]
                self.destination_name = str(dest_raw)
                try:
                    self.dest_node = self.to_node_id(self.destination_name)
                    self.destination_name = str(self.dest_node)
                    self.get_logger().info(f"[destination] updated to {self.dest_node}")
                except Exception as e:
                    self.get_logger().warn(f"Could not resolve destination {self.destination_name}: {e}")

            # Add waypoints
            if "waypoints" in data:
                for wp in data["waypoints"]:
                    try:
                        if "-" in wp:
                            u, v = [self.to_node_id(x) for x in wp.split("-")]
                            self.waypoints.extend([u, v])
                            self.virtual_edges.append((u, v))
                        else:
                            self.waypoints.append(self.to_node_id(wp))
                    except Exception as e:
                        self.get_logger().warn(f"Could not resolve waypoint {wp}: {e}")

            # Add avoidance
            if "avoidance" in data:
                for av in data["avoidance"]:
                    try:
                        nid = self.to_node_id(av)
                        if nid not in self.avoid_nodes:
                            self.avoid_nodes.add(nid)
                            self.resolved_avoidance.append({"name": av, "node_id": nid})
                    except Exception as e:
                        self.get_logger().warn(f"Could not resolve avoidance {av}: {e}")

            # Merge conditions
            if "conditions" in data:
                for c in data["conditions"]:
                    if c not in self.conditions:
                        self.conditions.append(c)
            self.destination_ready = bool(getattr(self, "dest_node", None) in getattr(self, "G", {}).nodes)          
            self.path_function_ready = True

            self.get_logger().info(
                f"[pre-try] dest_ready={self.destination_ready}, "
                f"dest_node={getattr(self, 'dest_node', None)}, "
                f"dest_name={getattr(self, 'destination_name', None)}, "
                f"path_fn_ready={getattr(self, 'path_function_ready', None)}"
            )
            # Recalculate route after applying changes
            self.try_generate_path()

        except Exception as e:
            self.get_logger().error(f"eval_output_callback error: {e}")

    # ---------------------------
    # Main input callback
    # ---------------------------
    def to_node_id(self, token: str) -> int:
        s = self._clean_token(token)
        if s.isdigit():
            k = int(s)
            if k in self.index_to_node_id:
                nid = int(self.index_to_node_id[k])
                self.get_logger().info(f"[index→node] {k} → {nid}")
                return nid
            if k in self.G.nodes:
                return k
            if isinstance(self.latest_route, (list, tuple)) and 0 <= k < len(self.latest_route):
                nid = int(self.latest_route[k])
                self.get_logger().info(f"[routeIndex→node] {k} → {nid}")
                return nid
            raise ValueError(f"token '{s}' not resolvable to node_id")
        # place name
        lat, lon = self.get_nearest_named_point(s)
        nid = ox.distance.nearest_nodes(self.G, X=lon, Y=lat)
        if nid not in self.G.nodes:
            raise ValueError(f"resolved node_id {nid} for '{s}' not in graph")
        self.get_logger().info(f"[name→node] '{s}' → {nid}")
        return int(nid)
        
    def destination_callback(self, msg):
        try:
            data = json.loads(msg.data)

            # 0) Normalize inputs
            ranked_destinations = data.get("ranked_destinations", [])
            self.ranked_destinations = [str(d) for d in ranked_destinations if d]

            dest_raw = data.get("destination", None)
            raw_waypoints = data.get("waypoints", []) or data.get("waypoints_list", [])
            raw_avoidance = data.get("avoidance", []) or data.get("avoid", [])
            self.original_destination_name = dest_raw
            self.original_waypoints = raw_waypoints[:]
            if isinstance(dest_raw, int):
                self.destination_name = str(dest_raw)   # keep for logs
                self.get_logger().info(f"Destination received as index/node: {self.destination_name}")
            else:
                self.destination_name = dest_raw

            # Unify strings
            raw_waypoints_str = [self._clean_token(wp if isinstance(wp, str) else str(wp)) for wp in raw_waypoints]
            raw_avoidance_str = [self._clean_token(av if isinstance(av, str) else str(av)) for av in raw_avoidance]
            self.avoidance = raw_avoidance_str  # keep original (for logs)

            self.get_logger().info(f"Destination(raw): {self.destination_name}")
            self.get_logger().info(f"Waypoints(raw): {raw_waypoints_str}")
            self.get_logger().info(f"Avoidance(raw): {raw_avoidance_str}")

            # Reset state
            self.avoid_nodes = set()
            self.resolved_avoidance = []
            self.virtual_edges = []
            resolved_wps = []  # list of node_id

            # Avoidance
            for item in raw_avoidance_str:
                try:
                    node_id = self.to_node_id(item)
                    if node_id not in self.G.nodes:
                        raise ValueError(f"node_id {node_id} not in graph.")
                    self.avoid_nodes.add(node_id)
                    self.resolved_avoidance.append({"name": item, "node_id": node_id})
                    self.get_logger().info(f"Avoid node added: {node_id} (from '{item}')")
                except Exception as e:
                    self.get_logger().warn(f"Could not resolve avoidance item '{item}': {e}")

            # Waypoints
            for item in raw_waypoints_str:
                try:
                    s = self._clean_token(item)
                    if "-" in s:
                        # 'X-Y' waypoint interpretation
                        parts = re.split(r"\s*-\s*", s)
                        if len(parts) == 2:
                            u = self.to_node_id(parts[0])
                            v = self.to_node_id(parts[1])
                            # sequential waypoint
                            resolved_wps.extend([u, v])
                            # virtual edges will be added to G.copy() during routing
                            self.virtual_edges.append((u, v))
                            self.get_logger().info(f"[virtual-edge plan] {u} <-> {v} (from '{s}')")
                            continue
                    # single token
                    nid = self.to_node_id(s)
                    resolved_wps.append(nid)
                    self.get_logger().info(f"[WP] Added waypoint node_id: {nid} (from '{s}')")
                except Exception as e:
                    self.get_logger().warn(f"[WP] Could not resolve waypoint '{item}': {e}")

            # Destination
            self.dest_node = None
            if self.destination_name is not None:
                try:
                    self.dest_node = self.to_node_id(self.destination_name)
                    self.get_logger().info(f"[destination] '{self.destination_name}' → node_id {self.dest_node}")
                except Exception as e:
                    self.get_logger().warn(f"[destination] Could not resolve '{self.destination_name}': {e}")

            # Overwrite waypoints with node_id(int) list
            # Also unify destination_name to node_id string (for downstream compatibility)
            self.waypoints = [int(n) for n in resolved_wps]
            if self.dest_node is not None:
                self.destination_name = str(self.dest_node)

            self.destination_ready = (self.dest_node is not None) or (self.destination_name is not None)
            self.conditions = data.get("conditions", [])

            # Try generating Path
            self.try_generate_path()

        except Exception as e:
            self.get_logger().error(f"Destination callback error: {e}")

    def prefer_path_callback(self, msg):
        try:
            exec_globals = {"avoid_nodes": self.avoid_nodes}
            exec(msg.data, exec_globals)
            self.prefer_path_function = exec_globals.get("prefer_path", None)
            if self.prefer_path_function:
                self.get_logger().info("prefer_path function received.")
                self.path_function_ready = True
                self.try_generate_path()
        except Exception as e:
            self.get_logger().error(f"Prefer path callback error: {e}")

    # ---------------------------
    # Routing
    # ---------------------------
    def try_generate_path(self):
        if not (self.destination_ready and self.path_function_ready):
            return
        elif self.start_coords is None:
            self.get_logger().error("No gps data")
            return
        try:
            G_work = self.G.copy()
            for (u, v) in getattr(self, "virtual_edges", []):
                if (u in G_work.nodes) and (v in G_work.nodes):
                    latu, lonu = G_work.nodes[u]['y'], G_work.nodes[u]['x']
                    latv, lonv = G_work.nodes[v]['y'], G_work.nodes[v]['x']
                    length = float(self._gc_distance_m(latu, lonu, latv, lonv))
                    G_work.add_edge(u, v, length=length, virtual=True, name="virtual_link")
                    G_work.add_edge(v, u, length=length, virtual=True, name="virtual_link")
                    self.get_logger().info(f"[virtual-edge add] {u} <-> {v}, length={length:.2f} m")

            # Destination/waypoint node_id sequence
            if self.dest_node is None:
                # If destination is still a string (name) and not yet resolved
                self.dest_node = self.to_node_id(self.destination_name)

            node_seq = [self.start_node] + [int(w) for w in self.waypoints] + [int(self.dest_node)]

            # Generate path for each segment and merge
            full_route = []
            for i in range(len(node_seq) - 1):
                sub_route = nx.shortest_path(
                    G_work,
                    source=node_seq[i],
                    target=node_seq[i + 1],
                    weight=self.prefer_path_function
                )
                if i > 0:
                    sub_route = sub_route[1:]  # remove overlap
                full_route += sub_route
            
            # Default shortest path (for comparison)
            default_route = nx.shortest_path(
                self.G,
                source=self.start_node,
                target=self.dest_node,
                weight='length'
            )

            # publish
            self.publish_route_data(default_route, full_route)

            self.destination_ready = False
            self.path_function_ready = False

            self.latest_route = full_route
            self.publish_path_request()

        except Exception as e:
            self.get_logger().error(f"Path generation failed: {e}")

    # ---------------------------
    # Place name utilities
    # ---------------------------
    def get_nearest_named_point(self, target_name):
        s = re.sub(r"\s*\(.*?\)\s*$", "", str(target_name)).strip()
        df = getattr(self, "named_features", None)
        if df is None or df.empty:
            raise ValueError("No named features available in bbox.")
        df = df.copy()
        for col in ("name", "name:en"):
            if col not in df.columns:
                df[col] = ""

        name = df["name"].astype(str).fillna("")
        name_en = df["name:en"].astype(str).fillna("")

        mask = (name == s) | (name_en == s)
        candidates = df[mask]

        if candidates.empty:
            sl = s.lower()
            candidates = df[(name.str.lower() == sl) | (name_en.str.lower() == sl)]

        if candidates.empty and s:
            pat = re.escape(s)
            candidates = df[
                name.str.contains(pat, case=False, na=False) |
                name_en.str.contains(pat, case=False, na=False)
            ]

        if candidates.empty:
            raise ValueError(f"'{target_name}' not found in named_features.")

        def _center_dist(geom):
            cy, cx = self._geom_center_latlon(geom)
            return distance(self.center_point, (cy, cx)).meters

        candidates = candidates.copy()
        candidates["center_dist"] = candidates.geometry.apply(_center_dist)
        nearest = candidates.sort_values("center_dist").iloc[0]
        cy, cx = self._geom_center_latlon(nearest.geometry)
        return (cy, cx)


    def get_nearest_place_name(self, lat, lon, threshold_m=30):
        # Route node nearest name searching
        best_name, best_d = None, float("inf")
        for _, row in self.named_features.iterrows():
            name = row.get("name")
            if not isinstance(name, str):
                continue
            cy, cx = self._geom_center_latlon(row.geometry)
            d = distance((lat, lon), (cy, cx)).meters
            if d < best_d:
                best_d, best_name = d, name
        return best_name if best_d <= threshold_m else None

    # ---------------------------
    # Publishers / Visualizers
    # ---------------------------
    def publish_route_data(self, default_route, route):
        # waypoints is already a list of node_id(int)
        resolved_waypoints = [int(wp) for wp in self.waypoints]

        route_node_indices = {str(node): i + 1 for i, node in enumerate(route)}
        route_data = {
            "default_route": default_route,
            "route": route,
            "waypoints": resolved_waypoints,
            "route_node_indices": route_node_indices
        }
        msg = String()
        msg.data = json.dumps(route_data)
        self.route_data_pub.publish(msg)

    def create_point(self, x, y):
        pt = ROSPoint()
        pt.x = x
        pt.y = y
        pt.z = 0.0
        return pt

    def publish_path_request(self):
        try:
            request_msg = {
                "destination": self.original_destination_name,    # node_id string if possible
                "destination_candidates": self.ranked_destinations,
                "waypoints": self.original_waypoints,              # list of node_id(int)
                "avoid": self.resolved_avoidance,           # list of dicts: {name, node_id}
                "conditions": self.conditions
            }
            msg = String()
            msg.data = json.dumps(request_msg, ensure_ascii=False)
            self.path_request_pub.publish(msg)
            self.get_logger().info("Path request has been published to /path_request.")
        except Exception as e:
            self.get_logger().error(f"Failed to publish /path_request: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PathGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Path Generator Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()