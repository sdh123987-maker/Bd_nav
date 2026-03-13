# Standard libraries
import os
import sys
import threading
import traceback
import math
import json
import textwrap  # <-- 들여쓰기 자동 보정을 위해 추가됨

# Third-party libraries
import networkx as nx
import osmnx as ox

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

class GTGeneratorNode(Node):
    def __init__(self):
        super().__init__('gt_generator')

        self.get_logger().info('GT Generator Node starting...')

        self.G = None
        self.index_to_osmid = {}
        self.osmid_to_index = {}
        
        # 현재 상태 저장
        self.latest_user_indices = []
        self.latest_gt_osmids = []
        self.prefer_path_function = None
        self.avoid_nodes = set()
        # 퍼블리셔: GT 경로를 map_viewer로 전송
        self.route_data_pub = self.create_publisher(String, 'route_data', 10)

        # 서브스크라이버: path_generator가 계산한 요약 로그 수신
        self.summary_sub = self.create_subscription(
            String, 
            '/generated_path_summary', 
            self.path_summary_callback, 
            10
        )

        self.load_graph()

        # 터미널 입력을 위한 별도 스레드 시작
        self.cli_thread = threading.Thread(target=self.cli_loop, daemon=True)
        self.cli_thread.start()

    def path_summary_callback(self, msg):
        """path_generator에서 보낸 요약 문자열을 ROS 태그 없이 깔끔하게 출력"""
        print(msg.data)
        # 입력 프롬프트가 밀렸을 수 있으므로 다시 그려줌
        print("> ", end="", flush=True)

    def load_graph(self):
        try:
            share_dir = get_package_share_directory('bd_nav')
            assets_dir = os.path.join(share_dir, "assets")
            target_ext = ".graphml"
            file_names_list = [f for f in os.listdir(assets_dir) if f.endswith(target_ext)]
            
            if not file_names_list:
                self.get_logger().error("No .graphml file found in assets directory.")
                return

            local_graph_path = os.path.join(assets_dir, file_names_list[0])
            self.get_logger().info(f"Loading map: {local_graph_path}")
            self.G = ox.load_graphml(local_graph_path)

            # map_viewer와 동일한 방식으로 index 부여 (1부터 시작)
            for idx, node_id in enumerate(self.G.nodes, start=1):
                self.index_to_osmid[idx] = node_id
                self.osmid_to_index[node_id] = idx

            self.get_logger().info(f"Graph loaded successfully. Total nodes: {len(self.G.nodes)}")

        except Exception as e:
            self.get_logger().error(f"Failed to load graph: {e}")

    def normalize_edge_data(self, data):
        """OSM edge 속성이 리스트형일 경우 첫 번째 요소로 정규화"""
        out = {}
        for k, v in (data or {}).items():
            if isinstance(v, list):
                out[k] = v[0] if len(v) > 0 else None
            else:
                out[k] = v
        return out

    def safe_weight(self, u, v, d):
        """사용자 정의 weight 함수 안전 호출 래퍼"""
        if self.prefer_path_function is None:
            if isinstance(d, list) and len(d) > 0:
                l = d[0].get('length', 0.0)
            elif isinstance(d, dict):
                l = d.get('length', 0.0)
            else:
                l = 0.0
            return float(l[0] if isinstance(l, list) else l)

        try:
            u_int = self.osmid_to_index.get(u, -1)
            v_int = self.osmid_to_index.get(v, -1)

            if isinstance(d, list) and len(d) > 0 and isinstance(d[0], dict):
                norm_d = [self.normalize_edge_data(d[0])]
            elif isinstance(d, dict):
                norm_d = [self.normalize_edge_data(d)]
            else:
                norm_d = [d]

            return self.prefer_path_function(u_int, v_int, norm_d)
        except Exception as e:
            self.get_logger().error(f"Weight function error: {e}")
            return float('inf')

    def generate_gt_route(self, user_indices):
        if not self.G:
            print("[오류] 그래프가 로드되지 않았습니다.")
            return

        invalid_indices = [idx for idx in user_indices if idx not in self.index_to_osmid]
        if invalid_indices:
            print(f"[오류] 그래프에 존재하지 않는 인덱스입니다: {invalid_indices}")
            return

        osmid_seq = [self.index_to_osmid[idx] for idx in user_indices]
        full_route_osmids = []

        try:
            for i in range(len(osmid_seq) - 1):
                source = osmid_seq[i]
                target = osmid_seq[i + 1]
                
                sub_route = nx.shortest_path(self.G, source=source, target=target, weight='length')
                
                if i > 0:
                    sub_route = sub_route[1:]
                full_route_osmids.extend(sub_route)

            self.latest_user_indices = user_indices
            self.latest_gt_osmids = full_route_osmids
            
            self.print_route_summary()
            self.publish_route_data()

        except nx.NetworkXNoPath:
            print(f"[오류] 노드 {source} 와 {target} 사이에 연결된 경로가 없습니다.")
        except Exception as e:
            print(f"[오류] 경로 생성 중 문제 발생: {e}")

    def publish_route_data(self):
        """생성된 GT 경로를 /route_data 토픽으로 발행하여 map_viewer에서 렌더링"""
        if not self.latest_gt_osmids:
            return
            
        try:
            waypoints_osmids = [self.index_to_osmid[idx] for idx in self.latest_user_indices]

            route_data = {
                "default_route": [],
                "route": self.latest_gt_osmids,
                "waypoints": waypoints_osmids
            }
            
            msg = String()
            msg.data = json.dumps(route_data)
            self.route_data_pub.publish(msg)
            
            print("🚀 [성공] GT 경로 시각화 데이터가 /route_data 토픽으로 전송되었습니다.")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish route data: {e}")

    def print_route_summary(self):
        if not self.latest_gt_osmids:
            return

        total_length = 0.0
        total_cost = 0.0

        for i in range(len(self.latest_gt_osmids) - 1):
            u = self.latest_gt_osmids[i]
            v = self.latest_gt_osmids[i + 1]

            if self.G.is_multigraph():
                min_l = 0.0
                min_c = float('inf')
                for key, edge_data in self.G[u][v].items():
                    c = self.safe_weight(u, v, edge_data)
                    l_val = edge_data.get('length', 0.0)
                    l = float(l_val[0] if isinstance(l_val, list) else l_val)
                    
                    if c < min_c:
                        min_c = c
                        min_l = l
                total_cost += min_c
                total_length += min_l
            else:
                edge_data = self.G[u][v]
                c = self.safe_weight(u, v, edge_data)
                l_val = edge_data.get('length', 0.0)
                l = float(l_val[0] if isinstance(l_val, list) else l_val)
                total_cost += c
                total_length += l

        idx_seq_str = " -> ".join(str(self.osmid_to_index.get(n, "?")) for n in self.latest_gt_osmids)
        osm_seq_str = " -> ".join(str(n) for n in self.latest_gt_osmids)

        cost_label = "Weight Cost" if self.prefer_path_function else "Default Cost (Length)"

        print("\n" + "="*80)
        print("🎯  [GROUND TRUTH PATH SUMMARY]")
        print(f"    📍 Input Indices  : {self.latest_user_indices}")
        print(f"    📏 Total Distance : {total_length:.2f} m")
        print(f"    ⚖️  {cost_label:<14} : {total_cost:.4f}")
        print("-" * 80)
        print(f"    [Node Index Sequence] ({len(self.latest_gt_osmids)} nodes)\n    {idx_seq_str}\n")
        print(f"    [OSM ID Sequence]\n    {osm_seq_str}")
        print("="*80 + "\n")

    def input_weight_function(self):
        print("\n[Weight Function 입력]")
        print("파이썬 코드를 붙여넣기 하세요. 입력을 마치려면 빈 줄에서 'EOF'를 입력하고 엔터를 누르세요.")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == 'EOF':
                    break
                lines.append(line)
            except EOFError:
                break

        # 입력받은 여러 줄의 코드를 하나로 합칩니다.
        code = "\n".join(lines)
        
        # [수정됨] 텍스트랩을 사용해 공통된 앞쪽 공백(들여쓰기)을 모두 제거합니다.
        code = textwrap.dedent(code).strip()
        
        if not code:
            print("입력된 코드가 없어 기존 상태를 유지합니다.\n")
            return

        try:
            exec_globals = {"avoid_nodes": self.avoid_nodes}
            exec(code, exec_globals)
            if "prefer_path" in exec_globals:
                self.prefer_path_function = exec_globals["prefer_path"]
                print("✅ 성공적으로 prefer_path 함수가 등록되었습니다.\n")
                
                if self.latest_gt_osmids:
                    print("🔄 새로운 Weight 함수를 적용하여 최근 GT 경로의 Cost를 재계산합니다...")
                    self.print_route_summary()
            else:
                print("❌ 코드 안에 'prefer_path' 함수가 정의되어 있지 않습니다.\n")
        except Exception as e:
            print(f"❌ 함수 컴파일/등록 중 오류가 발생했습니다:\n{traceback.format_exc()}\n")

    def cli_loop(self):
        import time
        while self.G is None:
            time.sleep(0.5)

        while True:
            try:
                print("=========================================")
                print("  GT Generator Menu")
                print("  1. 노드 인덱스 시퀀스 입력 (예: 1 5 12 18)")
                print("  2. Weight Function 코드 입력")
                print("  3. 종료")
                print("=========================================")
                choice = input("선택 (1/2/3): ").strip()

                if choice == '1':
                    idx_input = input("경유할 노드 인덱스들을 스페이스(공백)로 구분하여 입력하세요:\n> ").strip()
                    if not idx_input:
                        continue
                    try:
                        user_indices = [int(x) for x in idx_input.split()]
                        self.generate_gt_route(user_indices)
                    except ValueError:
                        print("❌ 숫자만 입력해주세요 (예: 1 5 12)")
                elif choice == '2':
                    self.input_weight_function()
                elif choice == '3':
                    print("GT Generator를 종료합니다.")
                    rclpy.shutdown()
                    break
                else:
                    print("잘못된 입력입니다. 1, 2, 3 중에서 선택하세요.")
            except (KeyboardInterrupt, EOFError):
                rclpy.shutdown()
                break

def main(args=None):
    rclpy.init(args=args)
    node = GTGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down GT Generator Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()