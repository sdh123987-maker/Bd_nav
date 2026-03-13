# Standard libraries
import base64
import datetime
import json
import os
import threading
from io import BytesIO
from typing import List, Optional, Tuple
import math

# Third-party libraries
import osmnx as ox
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# ROS 2 libraries
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

from bd_nav.config import PROCESS_DIST_M

class StreamlitInputNode(Node):
    def __init__(self):
        super().__init__('streamlit_user_input')
        self.user_input_publisher = self.create_publisher(String, 'user_input', 10)
        self.path_features_publisher = self.create_publisher(String, 'path_features', 10)

        self.subscription = self.create_subscription(String, 'path_request', self.path_request_callback, 10)
        self.subscription_image = self.create_subscription(String, 'map_image', self.map_image_callback, 10)
        self.subscription_gps = self.create_subscription(NavSatFix, 'gps/fix', self.gps_callback, 10)

        self.image_data = None
        self.gps_coords = None
        self.route_summary = ""
        share_dir = get_package_share_directory('bd_nav')
        self.log_path = os.path.join(share_dir, 'assets', 'user_input_log.jsonl')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def gps_callback(self, msg):
        lat, lon = msg.latitude, msg.longitude
        self.gps_coords = (lat, lon)

    def map_image_callback(self, msg):
        self.image_data = msg.data

    def save_input_log(self, mode, user_input):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "mode": mode,
            "user_input": user_input,
        }
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    def get_places_with_features(
        self,
        gps: Optional[tuple] = None,
        dist: int = PROCESS_DIST_M,
        keys: Optional[List[str]] = None,
        keep_unnamed: bool = False,
        limit: Optional[int] = None
    ) -> List[str]:
        if gps is None:
            gps = self.gps_coords
        if gps is None:
            self.get_logger().error("get_places_with_features: No GPS available.")
            return []

        if keys is None:
            keys = [
                "amenity", "shop", "leisure", "tourism", "building", "highway", "landuse",
                "office", "public_transport", "railway", "place"
            ]

        lat, lon = gps
        tags = {k: True for k in keys}
        bbox = ox.utils_geo.bbox_from_point((lat, lon), dist=dist)

        try:
            osm = ox.features_from_bbox(bbox=bbox, tags=tags)
        except Exception as e:
            self.get_logger().error(f"Features load failed: {e}")
            return []

        if osm.empty:
            return []

        osm = osm.copy()
        present_keys = [k for k in keys if k in osm.columns]
        if not present_keys:
            return []

        osm["type"] = osm[present_keys].apply(lambda row: row.first_valid_index(), axis=1)
        subset_cols = [c for c in ["osmid", "geometry"] if c in osm.columns]
        if subset_cols:
            osm.drop_duplicates(subset=subset_cols, keep="first", inplace=True)
        else:
            osm.drop_duplicates(keep="first", inplace=True)

        def merge_names(row):
            name_ko = row.get("name")
            name_en = row.get("name:en")
            if pd.notnull(name_ko) and pd.notnull(name_en):
                return f"{name_ko} ({name_en})"
            elif pd.notnull(name_en):
                return str(name_en)
            elif pd.notnull(name_ko):
                return str(name_ko)
            return None

        osm["merged_name"] = osm.apply(merge_names, axis=1)

        if not keep_unnamed:
            osm.dropna(subset=["merged_name"], inplace=True)

        if limit:
            osm = osm.head(limit)

        osm["merged_name"] = osm["merged_name"].fillna("").astype(str)
        osm["type"] = osm["type"].fillna("").astype(str)

        result = []
        for _, row in osm.iterrows():
            name = row["merged_name"].replace("/", "")
            type_val = row["type"].replace("/", "")
            if name and type_val:
                result.append(f'{{{name}, "{type_val}"}}')
            elif name:
                result.append(f'{{{name}}}')
        return result


    def get_path_features(
        self,
        gps: Optional[tuple] = None,
        dist: int = PROCESS_DIST_M,
        drop_keys: Optional[List[str]] = None,
        only_keys: Optional[List[str]] = None,
    ) -> str:
        if gps is None:
            gps = self.gps_coords
        if gps is None:
            self.get_logger().error("get_path_features: No GPS available.")
            return "{}"

        extra_tags = [
            "surface", "incline", "name", "maxspeed", "lanes", "oneway",
            "bridge", "tunnel", "width", "access", "bicycle", "cycleway",
            "sidewalk", "smoothness", "lit", "service"
        ]
        ox.settings.useful_tags_way = sorted(set(list(ox.settings.useful_tags_way) + extra_tags))

        def _safe_str(x) -> str:
            if x is None:
                return ""
            if isinstance(x, (list, tuple, set)):
                x = ",".join(map(str, x))
            return str(x).strip().replace("/", "")

        def _is_nan(x) -> bool:
            try:
                return isinstance(x, float) and math.isnan(x)
            except Exception:
                return False

        if drop_keys is None:
            drop_keys = [
                "geometry", "osmid", "osmid_original",
                "length", "oneway", "reversed", "incline"
            ]
        drop_keys_set = set(drop_keys)
        only_keys_set = set(only_keys) if only_keys else None

        lat, lon = gps

        try:
            G = ox.graph_from_point((lat, lon), dist=dist, network_type="all", simplify=False)
        except Exception as e:
            self.get_logger().error(f"Graph load failed: {e}")
            return "{}"

        feature_set = set()

        def _add_feature(k: str, v_any):
            if isinstance(v_any, bool):
                v = "true" if v_any else "false"
                feature_set.add(f'{k}={v}')
                return
            if isinstance(v_any, (list, tuple, set)):
                for vi in v_any:
                    si = _safe_str(vi)
                    if si != "":
                        feature_set.add(f'{k}={si}')
                return
            if v_any is None or _is_nan(v_any):
                return
            sv = _safe_str(v_any)
            if sv != "":
                feature_set.add(f'{k}={sv}')

        for _, _, _, data in G.edges(keys=True, data=True):
            for k, v in data.items():
                if k in drop_keys_set:
                    continue
                if only_keys_set is not None and k not in only_keys_set:
                    continue
                if v is None or _is_nan(v):
                    continue
                _add_feature(k, v)

        if not feature_set:
            return "{}"
        features_sorted = sorted(feature_set)
        return "{" + ", ".join(f'"{fv}"' for fv in features_sorted) + "}"


    def path_request_callback(self, msg):
        try:
            data = json.loads(msg.data)

            def to_str(x):
                if isinstance(x, dict):
                    if x.get("name"):
                        return str(x["name"])
                    if "node_id" in x:
                        return str(x["node_id"])
                    return json.dumps(x, ensure_ascii=False)
                return str(x)

            destination_candidates_raw = data.get("destination_candidates", [])
            destination_candidates = [to_str(d) for d in destination_candidates_raw if d not in (None, "")]
            destination = to_str(data.get("destination", "Unknown"))

            waypoints_raw = data.get("waypoints", [])
            waypoints = [to_str(w) for w in waypoints_raw]

            avoid_raw = data.get("avoid", [])
            avoid_names = [to_str(a) for a in avoid_raw] 
            conditions_raw = data.get("conditions", [])
            conditions = [to_str(c) for c in conditions_raw]

            summary = "\n[Route Request Summary]\n"
            summary += f"- Destination (1st priority): {destination}\n"

            if len(destination_candidates) > 1:
                other_ranks = destination_candidates[1:4]
                summary += f"- Other priorities: {', '.join(other_ranks)}\n"
            else:
                summary += "- Other priorities: None\n"

            summary += f"- Waypoints: {', '.join(waypoints) if waypoints else 'None'}\n"
            summary += f"- Places to avoid: {', '.join(avoid_names) if avoid_names else 'None'}\n"
            summary += "- Route preferences:\n"
            summary += "".join([f"  • {c}\n" for c in conditions]) if conditions else "  • None\n"
            summary += "\nType your new instruction below:"

            self.route_summary = summary

        except Exception as e:
            self.get_logger().error(f"Error in path_request_callback: {e}")

    # ==========================================================
    # [수정됨] path_reply 로직을 제거하고 항상 새로운 지시문으로 인식
    # ==========================================================
    def process_user_input(self, user_input):
        if not user_input.strip():
            return

        if self.gps_coords is not None:
            place_with_features = self.get_places_with_features(gps=self.gps_coords)
            path_features = self.get_path_features(gps=self.gps_coords)
            json_data = json.dumps({
                "user_instruction": user_input,
                "place_with_features": place_with_features
            }, ensure_ascii=False)
            path_data = json.dumps({
                "path_features" : path_features
            }, ensure_ascii=False)
            msg = String()
            msg.data = json_data
            path = String()
            path.data = path_data
            self.user_input_publisher.publish(msg)
            self.path_features_publisher.publish(path)
            self.get_logger().info("[New Input] Published to /user_input.")
            self.save_input_log("normal", user_input)
        else:
            self.get_logger().warn("No GPS yet. Waiting for /gps/fix (start the nav stack and ensure position fix).")
            self.route_summary = "No GPS data yet. Start the navigation stack and wait for position fix, then try again."


# ---------- Streamlit UI ----------
if "rclpy_initialized" not in st.session_state:
    try:
        rclpy.init()
    except RuntimeError as e:
        if "must only be called once" not in str(e):
            raise
    st.session_state["rclpy_initialized"] = True

if "ros_node" not in st.session_state:
    node = StreamlitInputNode()
    st.session_state["ros_node"] = node
    st.session_state["last_summary"] = ""
    st.session_state["route_summary"] = ""
else:
    node = st.session_state["ros_node"]

rclpy.spin_once(node, timeout_sec=0)

st_autorefresh(interval=3000, key="status_refresh")

st.title("BD-Nav User Instruction Interface")

if node.route_summary != st.session_state["last_summary"]:
    st.session_state["last_summary"] = node.route_summary
    st.rerun()

st.subheader("Status:")
st.text(st.session_state["last_summary"])

with st.form(key='instruction_form', clear_on_submit=True):
    user_input = st.text_input("Enter instruction:")
    submit_button = st.form_submit_button(label='Send')

    if submit_button:
        if user_input.strip():
            node.process_user_input(user_input)
            st.success("Instruction sent!")
        else:
            st.warning("Please enter an instruction.")

if node.image_data:
    try:
        image_bytes = base64.b64decode(node.image_data)
        img = Image.open(BytesIO(image_bytes))
        st.image(img, caption="Route Map", width='stretch')
    except Exception as e:
        st.error(f"image error: {e}")