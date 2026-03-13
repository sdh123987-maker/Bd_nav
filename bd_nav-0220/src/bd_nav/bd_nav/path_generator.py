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

from bd_nav.config import PROCESS_DIST_M

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
        self.prefer_path_source = None    # 파싱된 Weight 함수 코드 원본 저장용
        self.destination_ready = False
        self.path_function_ready = False
        self.latest_route = []  # store latest route

        self.subscription_output = self.create_subscription(
            String, 'user_output', self.destination_callback, 10)
        self.subscription_function = self.create_subscription(
            String, 'prefer_path_function', self.prefer_path_callback, 10)
        
        self.subscription_eval_output = self.create_subscription(
            String, 'eval_output', self.eval_output_callback, 10)
        
        self.path_request_pub = self.create_publisher(String, 'path_request', 10)
        self.route_data_pub = self.create_publisher(String, 'route_data', 10)
        self.summary_pub = self.create_publisher(String, '/generated_path_summary', 10)

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

    def _node_in_graph(self, n, G=None):
        G = G if G is not None else self.G
        try:
            return n in G.nodes
        except TypeError:
            return any(nn == n for nn in G.nodes)

    def process_graph(self, map_file_name):
        dist = PROCESS_DIST_M
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
                self.index_to_node_id[int(idx)] = node

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

    def eval_output_callback(self, msg):
        try:
            data = json.loads(msg.data) if msg and msg.data else {}

            if not data or (str(data.get("confirm", "")).lower() == "yes"):
                if getattr(self, "latest_route", None):
                    self.get_logger().info("Confirmed route publishing.")
                else:
                    self.get_logger().warn("No saved route to publish.")
                return

            if "destination" in data:
                dest_raw = data["destination"]
                self.destination_name = str(dest_raw)
                try:
                    self.dest_node = self.to_node_id(self.destination_name)
                    self.destination_name = str(self.dest_node)
                    self.get_logger().info(f"[destination] updated to {self.dest_node}")
                except Exception as e:
                    self.get_logger().warn(f"Could not resolve destination {self.destination_name}: {e}")

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

            self.avoid_nodes_list = getattr(self, "avoid_nodes_list", [])
            if "avoidance" in data:
                for av in data["avoidance"]:
                    try:
                        nid = self.to_node_id(av)
                        if nid not in self.avoid_nodes_list:
                            self.avoid_nodes_list.append(nid)
                            self.resolved_avoidance.append({"name": av, "node_id": nid})
                    except Exception as e:
                        self.get_logger().warn(f"Could not resolve avoidance {av}: {e}")

            if "conditions" in data:
                for c in data["conditions"]:
                    if c not in self.conditions:
                        self.conditions.append(c)
            dest = getattr(self, "dest_node", None)
            self.destination_ready = dest is not None and self._node_in_graph(dest)          
            self.path_function_ready = True

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
                nid = self.index_to_node_id[k]
                return nid
            if self._node_in_graph(k):
                return k
            if isinstance(self.latest_route, (list, tuple)) and 0 <= k < len(self.latest_route):
                nid = self.latest_route[k]
                return nid
            raise ValueError(f"token '{s}' not resolvable to node_id")
        
        lat, lon = self.get_nearest_named_point(s)
        nid = ox.distance.nearest_nodes(self.G, X=lon, Y=lat)
        if not self._node_in_graph(nid):
            raise ValueError(f"resolved node_id {nid} for '{s}' not in graph")
        return nid
        
    def destination_callback(self, msg):
        try:
            data = json.loads(msg.data)

            ranked_destinations = data.get("ranked_destinations", [])
            self.ranked_destinations = [str(d) for d in ranked_destinations if d]

            dest_raw = data.get("destination", None)
            raw_waypoints = data.get("waypoints", []) or data.get("waypoints_list", [])
            raw_avoidance = data.get("avoidance", []) or data.get("avoid", [])
            self.original_destination_name = dest_raw
            self.original_waypoints = raw_waypoints[:]
            if isinstance(dest_raw, int):
                self.destination_name = str(dest_raw)
            else:
                self.destination_name = str(dest_raw)

            raw_waypoints_str = [self._clean_token(wp if isinstance(wp, str) else str(wp)) for wp in raw_waypoints]
            raw_avoidance_str = [self._clean_token(av if isinstance(av, str) else str(av)) for av in raw_avoidance]
            self.avoidance = raw_avoidance_str 

            self.prefer_path_function = None
            self.prefer_path_source = None
            self.avoid_nodes = set()
            self.avoid_nodes_list = []
            self.resolved_avoidance = []
            self.virtual_edges = []
            resolved_wps = []  

            for item in raw_avoidance_str:
                try:
                    node_id = self.to_node_id(item)
                    if not self._node_in_graph(node_id):
                        raise ValueError(f"node_id {node_id} not in graph.")
                    self.avoid_nodes_list.append(node_id)
                    self.resolved_avoidance.append({"name": item, "node_id": node_id})
                except Exception as e:
                    self.get_logger().warn(f"Could not resolve avoidance item '{item}': {e}")

            for item in raw_waypoints_str:
                try:
                    s = self._clean_token(item)
                    if "-" in s:
                        parts = re.split(r"\s*-\s*", s)
                        if len(parts) == 2:
                            u = self.to_node_id(parts[0])
                            v = self.to_node_id(parts[1])
                            resolved_wps.extend([u, v])
                            self.virtual_edges.append((u, v))
                            continue
                    nid = self.to_node_id(s)
                    resolved_wps.append(nid)
                except Exception as e:
                    self.get_logger().warn(f"[WP] Could not resolve waypoint '{item}': {e}")

            self.dest_node = None
            if self.destination_name is not None:
                try:
                    self.dest_node = self.to_node_id(self.destination_name)
                except Exception as e:
                    self.get_logger().warn(f"[destination] Could not resolve '{self.destination_name}': {e}")

            self.waypoints = list(resolved_wps)
            if self.dest_node is not None:
                self.destination_name = str(self.dest_node)

            self.destination_ready = (self.dest_node is not None) or (self.destination_name is not None)
            self.conditions = data.get("conditions", [])

            self.try_generate_path()

        except Exception as e:
            self.get_logger().error(f"Destination callback error: {e}")

    def prefer_path_callback(self, msg):
        try:
            code = msg.data.strip()
            if "def prefer_path(" in code and "isinstance(u," not in code:
                code = re.sub(r"(def prefer_path\s*\([^)]+\)\s*:)", r"\1\n    u = int(u) if isinstance(u, (int, float)) else -1\n    v = int(v) if isinstance(v, (int, float)) else -1", code, count=1)
            
            self.prefer_path_source = code

            exec_globals = {"avoid_nodes": self.avoid_nodes}
            exec(code, exec_globals)
            raw_prefer = exec_globals.get("prefer_path", None)
            if raw_prefer is not None:
                self.prefer_path_function = raw_prefer
                self.get_logger().info("prefer_path function received and successfully parsed.")
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
            self.get_logger().warn("No GPS yet; path generation skipped until /gps/fix is received.")
            return
        try:
            G_work = self.G.copy()
            for (u, v) in getattr(self, "virtual_edges", []):
                if self._node_in_graph(u, G_work) and self._node_in_graph(v, G_work):
                    latu, lonu = G_work.nodes[u]['y'], G_work.nodes[u]['x']
                    latv, lonv = G_work.nodes[v]['y'], G_work.nodes[v]['x']
                    length = float(self._gc_distance_m(latu, lonu, latv, lonv))
                    G_work.add_edge(u, v, length=length, virtual=True, name="virtual_link")
                    G_work.add_edge(v, u, length=length, virtual=True, name="virtual_link")

            nodes_list = list(G_work.nodes())

            def node_to_idx(node):
                for i, nn in enumerate(nodes_list):
                    if nn == node:
                        return i
                return None

            if G_work.is_directed():
                H = nx.MultiDiGraph() if G_work.is_multigraph() else nx.DiGraph()
            else:
                H = nx.MultiGraph() if G_work.is_multigraph() else nx.Graph()
            for i, n in enumerate(nodes_list):
                H.add_node(i, **dict(G_work.nodes[n]))

            def _normalize_edge_data(data):
                out = {}
                for k, v in (data or {}).items():
                    if isinstance(v, list):
                        out[k] = v[0] if len(v) > 0 else None
                    else:
                        out[k] = v
                return out

            for e in G_work.edges(keys=True, data=True):
                u, v = e[0], e[1]
                if G_work.is_multigraph():
                    key, data = e[2], e[3]
                else:
                    key, data = None, e[2]
                ii, jj = node_to_idx(u), node_to_idx(v)
                if ii is None or jj is None:
                    continue
                data = _normalize_edge_data(data)
                if key is not None:
                    try:
                        H.add_edge(ii, jj, key=key, **data)
                    except TypeError:
                        H.add_edge(ii, jj, key=0, **data)
                else:
                    H.add_edge(ii, jj, **data)

            avoid_indices = set()
            for n in getattr(self, "avoid_nodes_list", []):
                idx = node_to_idx(n)
                if idx is not None:
                    avoid_indices.add(idx)

            if self.dest_node is None:
                self.dest_node = self.to_node_id(self.destination_name)
            node_seq = [self.start_node] + list(self.waypoints) + [self.dest_node]
            node_seq_idx = []
            for n in node_seq:
                idx = node_to_idx(n)
                if idx is None:
                    self.get_logger().error("Path generation failed: start/dest/waypoint not in graph.")
                    return
                node_seq_idx.append(idx)

            # ==========================================================
            # MultiGraph의 병렬 에지 가중치 계산 분리
            # ==========================================================
            def _calc_single_weight(u, v, edge_attrs):
                """단일 에지의 속성을 받아서 Weight를 계산하는 함수"""
                try:
                    u_int = int(u)
                except (TypeError, ValueError):
                    u_int = node_to_idx(u) if node_to_idx(u) is not None else -1
                try:
                    v_int = int(v)
                except (TypeError, ValueError):
                    v_int = node_to_idx(v) if node_to_idx(v) is not None else -1
                
                norm_d = [_normalize_edge_data(edge_attrs)]
                return self.prefer_path_function(u_int, v_int, norm_d)

            def _dijkstra_weight(u, v, d):
                """NetworkX 최단 경로 탐색 시 사용되는 래퍼 함수 (MultiGraph 대응)"""
                if H.is_multigraph():
                    # d는 딕셔너리의 딕셔너리 {0: {...}, 1: {...}} 형태
                    return min((_calc_single_weight(u, v, attrs) for attrs in d.values()), default=float('inf'))
                else:
                    return _calc_single_weight(u, v, d)

            # Update avoid nodes for global access
            self.avoid_nodes.clear()
            self.avoid_nodes.update(avoid_indices)

            # ----------------------------------------------------------
            # 1. Predicted Route (Custom Weight 반영)
            # ----------------------------------------------------------
            full_route_idx = []
            for i in range(len(node_seq_idx) - 1):
                sub_route = nx.shortest_path(
                    H,
                    source=node_seq_idx[i],
                    target=node_seq_idx[i + 1],
                    weight=_dijkstra_weight 
                )
                if i > 0:
                    sub_route = sub_route[1:]
                full_route_idx += sub_route

            full_route = [nodes_list[k] for k in full_route_idx]

            # ----------------------------------------------------------
            # 2. Default Route (거리 우선) - [롤백됨] 출발지와 도착지만 고려
            # ----------------------------------------------------------
            try:
                start_idx = node_seq_idx[0]
                dest_idx = node_seq_idx[-1]
                default_route_idx = nx.shortest_path(H, source=start_idx, target=dest_idx, weight='length')
                default_route = [nodes_list[k] for k in default_route_idx]
            except nx.NetworkXNoPath:
                self.get_logger().error("No path found between start and destination.")
                default_route_idx = []
                default_route = []

            # ----------------------------------------------------------
            # 검증 및 Cost 연산
            # ----------------------------------------------------------
            total_length = 0.0
            total_weight = 0.0

            # Predicted Route 연산
            for i in range(len(full_route_idx) - 1):
                u_idx = full_route_idx[i]
                v_idx = full_route_idx[i + 1]
                
                if H.is_multigraph():
                    min_w = float('inf')
                    min_l = 0.0
                    for key, edge_data in H[u_idx][v_idx].items():
                        w = _calc_single_weight(u_idx, v_idx, edge_data)
                        if w < min_w:
                            min_w = w
                            l = edge_data.get('length', 0.0)
                            min_l = float(l[0] if isinstance(l, list) else l)
                    total_weight += min_w
                    total_length += min_l
                else:
                    edge_data = H[u_idx][v_idx]
                    total_weight += _calc_single_weight(u_idx, v_idx, edge_data)
                    l = edge_data.get('length', 0.0)
                    total_length += float(l[0] if isinstance(l, list) else l)

            route_str = " -> ".join(str(n) for n in full_route)

            # Default Route 연산
            default_total_length = 0.0
            default_total_weight = 0.0
            
            for i in range(len(default_route_idx) - 1):
                u_idx = default_route_idx[i]
                v_idx = default_route_idx[i + 1]
                
                if H.is_multigraph():
                    min_l = float('inf')
                    best_edge_data = None
                    for key, edge_data in H[u_idx][v_idx].items():
                        l = edge_data.get('length', 0.0)
                        l_val = float(l[0] if isinstance(l, list) else l)
                        if l_val < min_l:
                            min_l = l_val
                            best_edge_data = edge_data
                    
                    default_total_length += min_l
                    default_total_weight += _calc_single_weight(u_idx, v_idx, best_edge_data)
                else:
                    edge_data = H[u_idx][v_idx]
                    l = edge_data.get('length', 0.0)
                    default_total_length += float(l[0] if isinstance(l, list) else l)
                    default_total_weight += _calc_single_weight(u_idx, v_idx, edge_data)

            default_route_str = " -> ".join(str(n) for n in default_route)

            func_code = self.prefer_path_source if self.prefer_path_source else "Default Cost (Length)"
            indented_func_code = "\n".join(f"      {line}" for line in func_code.split("\n"))
            
            # ==========================================================
            # 요약 로그 출력
            # ==========================================================
            summary_msg = (
                f"\n{'='*70}\n"
                f"🛣️  [GENERATED PATH SUMMARY]\n\n"
                f"🔴 [Predicted Route (Custom Weight)]\n"
                f"   🔹 Total Distance : {total_length:.2f} m\n"
                f"   🔹 Total Cost     : {total_weight:.4f}\n"
                f"   🔹 Node Sequence ({len(full_route)} nodes):\n"
                f"      {route_str}\n\n"
                f"🔵 [Default Route (Shortest Length)]\n"
                f"   🔹 Total Distance : {default_total_length:.2f} m\n"
                f"   🔹 Total Cost     : {default_total_weight:.4f}\n"
                f"   🔹 Node Sequence ({len(default_route)} nodes):\n"
                f"      {default_route_str}\n\n"
                f"📋 [Parsed Weight Function]\n"
                f"{indented_func_code}\n"
                f"{'='*70}"
            )
            
            msg = String()
            msg.data = summary_msg
            self.summary_pub.publish(msg)
            self.get_logger().info("Generated path summary published to /generated_path_summary")

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
        resolved_waypoints = list(self.waypoints)

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
                "destination": self.original_destination_name,
                "destination_candidates": self.ranked_destinations,
                "waypoints": self.original_waypoints,
                "avoid": self.resolved_avoidance,
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