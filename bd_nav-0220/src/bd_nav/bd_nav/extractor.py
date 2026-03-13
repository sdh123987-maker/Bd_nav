#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Dict, List, Optional, Set, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import NavSatFix
import osmnx as ox

from bd_nav.config import PROCESS_DIST_M

DEFAULT_GPS = (37.3513, -121.9603)   # (latitude, longitude)
DEFAULT_DIST = PROCESS_DIST_M


class PathFeatureDumper(Node):
    """
    역할:
    - /gps/fix (NavSatFix) 1회 수신 시 해당 좌표로 주변 그래프를 로드
    - edge attribute에서 path_feature 후보를 전부 수집
    - 중복 제거 후 ROS 로그로 전부 출력
    - 출력 후 종료
    """

    def __init__(self):
        super().__init__("path_feature_dumper")

        # Parameters
        self.declare_parameter("dist", DEFAULT_DIST)
        self.declare_parameter(
            "drop_keys",
            # [수정됨] "name"을 drop_keys 기본 리스트에 추가하여 출력에서 배제
            ["geometry", "osmid", "osmid_original", "length", "oneway", "reversed", "incline", "name"],
        )
        self.declare_parameter("only_keys", [])
        self.declare_parameter("fallback_latitude", DEFAULT_GPS[0])
        self.declare_parameter("fallback_longitude", DEFAULT_GPS[1])

        # 핵심: 기본값 False (fallback 사용 안 함)
        self.declare_parameter("use_fallback_if_no_gps", False)

        # (선택) fallback 대기 시간
        self.declare_parameter("fallback_wait_sec", 2.0)

        self.dist = int(self.get_parameter("dist").value)
        self.drop_keys = list(self.get_parameter("drop_keys").value)
        self.only_keys = list(self.get_parameter("only_keys").value) or None

        self.fallback_gps: Tuple[float, float] = (
            float(self.get_parameter("fallback_latitude").value),
            float(self.get_parameter("fallback_longitude").value),
        )
        self.use_fallback = bool(self.get_parameter("use_fallback_if_no_gps").value)
        self.fallback_wait_sec = float(self.get_parameter("fallback_wait_sec").value)

        self.gps: Optional[Tuple[float, float]] = None
        self._done = False

        # Subscribe GPS
        self.create_subscription(
            NavSatFix,
            "/gps/fix",
            self.gps_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Waiting for /gps/fix ... (dist={self.dist}m). "
            f"Fallback enabled={self.use_fallback}"
        )

        # fallback이 켜진 경우에만 타이머 생성
        self._fallback_timer = None
        if self.use_fallback:
            self._fallback_timer = self.create_timer(self.fallback_wait_sec, self._fallback_tick)

    def _fallback_tick(self):
        if self._done:
            return
        if self.gps is not None:
            return
        if not self.use_fallback:
            return

        self.get_logger().warn("No /gps/fix received yet. Using fallback GPS once.")
        self.gps = self.fallback_gps
        self.run_once_and_shutdown()

    def gps_callback(self, msg: NavSatFix):
        if self._done:
            return

        lat, lon = float(msg.latitude), float(msg.longitude)

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            self.get_logger().error(f"Invalid GPS received: lat={lat}, lon={lon}")
            return

        self.gps = (lat, lon)
        self.get_logger().info(f"[GPS] received lat={lat}, lon={lon}")
        self.run_once_and_shutdown()

    @staticmethod
    def _safe_str(x) -> str:
        if x is None:
            return ""
        if isinstance(x, (list, tuple, set)):
            x = ",".join(map(str, x))
        return str(x).strip().replace("/", "")

    @staticmethod
    def _is_nan(x) -> bool:
        try:
            return isinstance(x, float) and math.isnan(x)
        except Exception:
            return False

    def get_path_features(self) -> Dict[str, List[str]]:
        """
        [수정됨] 수집된 특징들을 Key별로 그룹화하여 딕셔너리로 반환합니다.
        예: {"highway": ["residential", "primary"], "surface": ["asphalt"]}
        """
        if self.gps is None:
            return {}

        extra_tags = [
            "surface",
            "incline",
            "bicycle",
            "cycleway",
            "sidewalk",
            "smoothness",
            "lit",
        ]
        ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way + extra_tags))

        drop_keys_set = set(self.drop_keys)
        only_keys_set = set(self.only_keys) if self.only_keys else None

        try:
            G = ox.graph_from_point(
                self.gps,
                dist=self.dist,
                network_type="all",
                simplify=False,
            )
        except Exception as e:
            self.get_logger().error(f"Graph load failed at GPS={self.gps}: {e}")
            return {}

        feature_dict: Dict[str, Set[str]] = {}

        for _, _, _, data in G.edges(keys=True, data=True):
            for k, v in data.items():
                if k in drop_keys_set:
                    continue
                if only_keys_set is not None and k not in only_keys_set:
                    continue
                if v is None or self._is_nan(v):
                    continue
                
                if k not in feature_dict:
                    feature_dict[k] = set()

                if isinstance(v, bool):
                    feature_dict[k].add('true' if v else 'false')
                    continue
                if isinstance(v, (list, tuple, set)):
                    for vi in v:
                        s = self._safe_str(vi)
                        if s:
                            feature_dict[k].add(s)
                    continue

                sv = self._safe_str(v)
                if sv:
                    feature_dict[k].add(sv)

        # 각 Key별 Value Set을 정렬된 List로 변환
        sorted_feature_dict = {k: sorted(list(v)) for k, v in feature_dict.items()}
        return sorted_feature_dict

    def log_all_features(self, feature_dict: Dict[str, List[str]]) -> None:
        """
        [수정됨] 수집된 딕셔너리를 가독성 좋게 포맷팅하여 단일 로그 문자열로 출력합니다.
        """
        if not feature_dict:
            self.get_logger().warn("No path features found.")
            return

        total_keys = len(feature_dict)
        
        # 출력 메시지 작성
        lines = []
        lines.append(f"\n{'=' * 70}")
        lines.append("🛣️  [EXTRACTED PATH FEATURES SUMMARY]")
        lines.append(f"   🔹 GPS Center : {self.gps}")
        lines.append(f"   🔹 Search Dist: {self.dist}m")
        lines.append(f"   🔹 Total Keys : {total_keys}")
        lines.append("-" * 70)

        # Key별로 정렬하여 깔끔하게 출력
        for key in sorted(feature_dict.keys()):
            values = feature_dict[key]
            values_str = ", ".join(values)
            # key 길이를 15글자에 맞춰 정렬
            lines.append(f"   🔸 {key:<15} : {values_str}")
        
        lines.append("=" * 70)
        
        # 한 번에 로거로 출력
        self.get_logger().info("\n".join(lines))

    def run_once_and_shutdown(self):
        if self._done:
            return
        self._done = True

        features = self.get_path_features()
        self.log_all_features(features)

        # timer stop (fallback 사용 시에만 존재)
        try:
            if self._fallback_timer is not None:
                self._fallback_timer.cancel()
        except Exception:
            pass

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PathFeatureDumper()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()