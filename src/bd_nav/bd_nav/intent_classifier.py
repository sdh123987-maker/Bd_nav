# Standard library
import os
import re
import json
import copy

# Third-party libraries
import openai
import osmnx as ox

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

class IntentClassifierNode(Node):
    def __init__(self, intent_prompt_template, path_prompt_template):
        super().__init__('intent_classifier')
        self.intent_prompt_template = intent_prompt_template
        self.path_prompt_template = path_prompt_template
        self.last_place_with_features = []  # place_with_features로 변경
        self.last_user_output = None

        self.user_input_subscription = self.create_subscription(
            String, 'user_input', self.user_input_callback, 10)

        self.path_reply_subscription = self.create_subscription(
            String, 'path_reply', self.path_reply_callback, 10)

        self.user_output_publisher = self.create_publisher(String, 'user_output', 10)
        self.get_logger().info('Intent classifier Node has been started.')

        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.map_loaded = False

        
    def parse_gpt_output(self, gpt_text: str):

        RE_YES  = re.compile(r"^\s*yes\s*$", re.IGNORECASE)
        RE_WP   = re.compile(r"^\s*waypoint\s*:\s*(\d+)\s*(?:-|–|—|->|→)\s*(\d+)\s*$", re.IGNORECASE)
        RE_AVO  = re.compile(r"^\s*avoid\s*:\s*(\d+)\s*$", re.IGNORECASE)
        RE_DEST = re.compile(r"^\s*dest(?:ination)?\s*:\s*(\d+)\s*$", re.IGNORECASE)

        def _strip_quotes(s: str) -> str:
            s = s.strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1].strip()
            return s

        def _parse_list(text: str):
            text = text.strip()
            if not text:
                return []
            if text.startswith('[') and not text.endswith(']'):
                text += ']'
            try:
                val = json.loads(text)
                if isinstance(val, list):
                    return [_strip_quotes(str(x)) for x in val]
            except Exception:
                pass
            items = [it for it in (t.strip() for t in text.strip('[]').split(',')) if it]
            return [_strip_quotes(it) for it in items]

        def _first_nonempty_line(t: str) -> str:
            t = (t or "").strip()
            if t.startswith("```") and t.endswith("```"):
                t = t[3:-3].strip()
            for ln in t.splitlines():
                ln = ln.strip()
                if ln:
                    return ln
            return ""
        
        line = _first_nonempty_line(gpt_text)
        if RE_YES.fullmatch(line):
            return {}
        m = RE_WP.fullmatch(line)
        if m:
            return {"waypoints": [f"{m.group(1)}-{m.group(2)}"]}
        m = RE_AVO.fullmatch(line)
        if m:
            return {"avoidance": [m.group(1)]}
        m = RE_DEST.fullmatch(line)
        if m:
            return {"destination": m.group(1)}
        try:
            if '[user_output]' in gpt_text:
                gpt_text = gpt_text.split('[user_output]', 1)[1]
            m = re.search(r'\{.*\}', gpt_text, flags=re.DOTALL)
            if m:
                obj = json.loads(m.group(0))

                if 'waypoint' in obj and 'waypoints' not in obj:
                    obj['waypoints'] = obj.pop('waypoint')
                if 'avoid' in obj and 'avoidance' not in obj:
                    obj['avoidance'] = obj.pop('avoid')
                if 'condition' in obj and 'conditions' not in obj:
                    cond = obj.pop('condition')
                    obj['conditions'] = [] if (isinstance(cond, str) and cond.strip().lower() == 'no condition') \
                                        else ([cond] if isinstance(cond, str) else (cond or []))
                out = {}

                # destination
                if 'destination' in obj:
                    dest = obj['destination']
                    if isinstance(dest, list):
                        dest_list = [str(_strip_quotes(str(x))) for x in dest if str(x).strip()]
                        if dest_list:
                            out['ranked_destinations'] = dest_list
                            out['destination'] = dest_list[0]
                    else:
                        dest_str = str(dest).strip()
                        if dest_str:
                            out['destination'] = dest_str

                # ranked_destinations
                if 'ranked_destinations' in obj and not out.get('ranked_destinations'):
                    rd = obj['ranked_destinations'] or []
                    rd = [str(_strip_quotes(str(x))).strip() for x in rd if str(x).strip()]
                    if rd:
                        out['ranked_destinations'] = rd
                        if 'destination' not in out:
                            out['destination'] = rd[0]

                # waypoints
                if 'waypoints' in obj:
                    wps = obj['waypoints']
                    if isinstance(wps, str):
                        wps = [wps]
                    wps = [str(_strip_quotes(str(x))).strip() for x in (wps or []) if str(x).strip()]
                    if wps:
                        out['waypoints'] = wps

                # avoidance
                if 'avoidance' in obj:
                    avs = obj['avoidance']
                    if isinstance(avs, str):
                        avs = [avs]
                    avs = [str(_strip_quotes(str(x))).strip() for x in (avs or []) if str(x).strip()]
                    if avs:
                        out['avoidance'] = avs

                # conditions
                if 'conditions' in obj:
                    conds = obj['conditions']
                    if isinstance(conds, str):
                        conds = [conds]
                    conds = [str(_strip_quotes(str(x))).strip() for x in (conds or []) if str(x).strip() and str(x).strip().lower() != 'no condition']
                    if conds:
                        out['conditions'] = conds

                return out
        except Exception as e:
            self.get_logger().warn(f"Failed to parse JSON block: {e}")
        out = {}

        for raw_line in gpt_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            low = line.lower()

            if low.startswith("destination:"):
                dest_str = line.split(":", 1)[1].strip()
                ranked = _parse_list(dest_str)
                ranked = [s for s in ranked if s]
                if ranked:
                    out["ranked_destinations"] = ranked
                    out["destination"] = ranked[0]
                else:
                    if dest_str:
                        out["destination"] = dest_str

            elif low.startswith("waypoints:") or low.startswith("waypoint:"):
                content = line.split(":", 1)[1].strip()
                wps = [s for s in _parse_list(content) if s]
                if wps:
                    out["waypoints"] = wps

            elif low.startswith("avoidance:"):
                content = line.split(":", 1)[1].strip()
                avs = [s for s in _parse_list(content) if s]
                if avs:
                    out["avoidance"] = avs

            elif low.startswith("condition:"):
                cond = line.split(":", 1)[1].strip()
                if cond and cond.lower() != "no condition":
                    conds = [c.strip() for c in cond.split(",") if c.strip()]
                    if conds:
                        out["conditions"] = conds

        return out

    def timer_callback(self):
        if not self.map_loaded:
            try:
                # 2. Load graph
                share_dir = get_package_share_directory('bd_nav')
                assets_dir = os.path.join(share_dir, "assets")
                target_ext = ".graphml"
                file_names_list = [f for f in os.listdir(assets_dir) if f.endswith(target_ext)]

                map_file_name = file_names_list[0]
                map_path = os.path.join(assets_dir, map_file_name)

                self.index_to_node_id = {}
                try:
                    G = ox.load_graphml(map_path)
                    for node in G.nodes:
                        idx_raw = G.nodes[node].get('index')
                        try:
                            idx = int(idx_raw)
                            self.index_to_node_id[idx] = node
                        except (ValueError, TypeError):
                            self.get_logger().warn(f"Invalid index for node {node}: {idx_raw}")
                    self.get_logger().info(f"Loaded graph with {len(self.index_to_node_id)} indexed nodes.")
                except Exception as e:
                    self.get_logger().error(f"Failed to load : {e}")
                
                self.map_loaded = True
            except Exception as e:
                self.get_logger().error(f"map_viewer error: {e}") 

    def user_input_callback(self, msg):
        try:
            self.get_logger().info(f"Received msg:\n{msg.data}")
            data = json.loads(msg.data)
            user_instruction = data.get("user_instruction", "")
            self.last_place_with_features = data.get("place_with_features", [])

            prompt = f"""{self.intent_prompt_template}

            [instruction]
            "{user_instruction}"
            [place_with_features]
            {json.dumps(self.last_place_with_features, ensure_ascii=False)}

            Now complete the following task:"""

            self.get_logger().info("GPT Processing with structured input...")

            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_completion_tokens=500
            )
            
            reply = response['choices'][0]['message']['content'].strip()
            self.get_logger().info("Raw GPT Output:")
            self.get_logger().info(reply)
            structured_data = self.parse_gpt_output(reply)
            self.last_user_output = structured_data

            self.get_logger().info("Parsed GPT Output:")
            self.get_logger().info(json.dumps(structured_data, indent=2, ensure_ascii=False))

            output_msg = String()
            output_msg.data = json.dumps(structured_data, ensure_ascii=False)
            self.user_output_publisher.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f'Intent classifier - GPT failed: {e}')

    def convert_index_fields(self, user_output):
        def convert_list(lst):
            result = []
            for item in lst:
                if isinstance(item, int) or (isinstance(item, str) and str(item).isdigit()):
                    idx = int(item)
                    if idx in self.index_to_node_id:
                        result.append(self.index_to_node_id[idx])
                        self.get_logger().info(f"[waypoints/avoidance] index {idx} → node_id {self.index_to_node_id[idx]}")
                    else:
                        result.append(item)
                        self.get_logger().warn(f"index {idx} not in map. left as-is.")
                else:
                    result.append(item)
            return result

        output = copy.deepcopy(user_output)
        output["waypoints"] = convert_list(output.get("waypoints", []))
        output["avoidance"] = convert_list(output.get("avoidance", []))

        dest = output.get("destination")
        if isinstance(dest, int) or (isinstance(dest, str) and str(dest).isdigit()):
            idx = int(dest)
            if idx in self.index_to_node_id:
                output["destination"] = self.index_to_node_id[idx]
                self.get_logger().info(f"[destination] index {idx} > node_id {self.index_to_node_id[idx]}")
            else:
                self.get_logger().warn(f"[destination] index {idx} not found. left as-is.")

        return output

    def path_reply_callback(self, msg):
        try:
            reply_text = msg.data.strip().lower()
            if reply_text == 'yes':
                self.get_logger().info("Route confirmed. Clearing stored user_output and place_with_features.")
                self.last_user_output = None
                self.last_place_with_features = []
                return

            if not self.last_user_output:
                self.get_logger().warn("No previous user_output available. Skipping.")
                return

            place_with_features = self.last_place_with_features
            user_output_json = json.dumps(self.last_user_output, ensure_ascii=False)

            prompt = f"""{self.path_prompt_template}

    The following is the last structured user output:
    [user_output]
    {user_output_json}

    The user has replied with the following update:
    [path_reply]
    "{reply_text}"

    Only use place names from the list below:
    [place_with_features]
    {json.dumps(place_with_features, ensure_ascii=False)}

    Now generate the revised user_output."""

            self.get_logger().info("GPT Processing (updating based on path_reply)...")

            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_completion_tokens=300
            )
            reply = response['choices'][0]['message']['content'].strip()
            self.get_logger().info("Raw GPT Output:")
            self.get_logger().info(reply)
            structured_data = self.parse_gpt_output(reply)
            converted_data = self.convert_index_fields(structured_data)

            self.last_user_output = converted_data
            self.get_logger().info("Parsed updated user_output:")
            self.get_logger().info(json.dumps(converted_data, indent=2, ensure_ascii=False))

            output_msg = String()
            output_msg.data = json.dumps(converted_data, ensure_ascii=False)
            self.user_output_publisher.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f'path_reply_callback error: {e}')


def initial_setting():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    lang_package_name = 'bd_nav'
    intent_prompt_file = 'assets/intent_classifier.txt'
    path_prompt_file = 'assets/path_classifier.txt'
    lang_package_share_dir = get_package_share_directory(lang_package_name)

    with open(os.path.join(lang_package_share_dir, intent_prompt_file), encoding='utf-8-sig') as f:
        intent_prompt_template = f.read()
    with open(os.path.join(lang_package_share_dir, path_prompt_file), encoding='utf-8-sig') as f:
        path_prompt_template = f.read()
    return intent_prompt_template, path_prompt_template


def main(args=None):
    intent_prompt_template, path_prompt_template = initial_setting()
    rclpy.init(args=args)
    node = IntentClassifierNode(intent_prompt_template, path_prompt_template)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Intent Classifier Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
