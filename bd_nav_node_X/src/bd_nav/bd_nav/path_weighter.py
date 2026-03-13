# Standard library
import os
import json
import re

# Third-party libraries
import openai

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


class PathWeighterNode(Node):
    def __init__(self, prompt_template):
        super().__init__('path_weighter')
        self.prompt_template = prompt_template

        self.subscription = self.create_subscription(
            String, 'user_output', self.listener_callback, 10)
        
        self.subscription = self.create_subscription(
            String, 'path_features', self.path_features_callback, 10)

        self.path_function_publisher = self.create_publisher(
            String, 'prefer_path_function', 10)

        self.get_logger().info('Path Weighter Node has been started.')

    def path_features_callback(self, msg):
        try:
            payload = json.loads(msg.data) if msg.data and msg.data.strip().startswith('{') else {}
            raw = payload.get("path_features", "")
            self.latest_path_features_raw = raw

            # Extract tokens inside quotes from {"k=v","k=v2"}
            tokens = re.findall(r'"([^"]+)"', raw) if raw else []
            self.latest_path_features_set = set(t.strip() for t in tokens if t.strip())
            self.get_logger().info(f"[path_features] {len(self.latest_path_features_set)} tokens loaded.")
        except Exception as e:
            self.get_logger().error(f"path_features parse error: {e}")
            self.latest_path_features_raw = ""
            self.latest_path_features_set = set()

    def listener_callback(self, msg):
        try:
            self.get_logger().info(f"Received /user_output:{msg.data}")
            data = json.loads(msg.data)

            # 1. Collect user conditions
            conditions = data.get("conditions", [])
            if not conditions:
                conditions = ["No condition"]
            condition_text = "\n".join([f"- {cond}" for cond in conditions])

            # 2. Prepare path_features injection
            MAX_FEATS = 150 
            feats = sorted(list(self.latest_path_features_set))[:MAX_FEATS]
            if feats:
                feats_block = "{ " + ", ".join(f'"{t}"' for t in feats) + " }"
                feats_note = "Use only attributes that actually exist in d[0]."
            else:
                feats_block = "{}"
                feats_note = (
                    "No map features were received; write robust code that checks keys in d[0] "
                    "and uses safe defaults."
                )

            # 3. Prompt substitution
            # Inject {feats_block}, {feats_note}, {condition_text} from the txt template here
            try:
                prompt = self.prompt_template.format(
                    feats_block=feats_block,
                    feats_note=feats_note,
                    condition_text=condition_text
                )
            except Exception as e:
                # Defend against placeholder mismatch in template, etc.
                self.get_logger().error(f"Prompt format error: {e}")
                # Minimal safe prompt fallback
                prompt = (
                    f"{self.prompt_template}\n\n"
                    f"Map Features (deduplicated):\n{feats_block}\n{feats_note}\n\n"
                    f"User Conditions:\n{condition_text}\n"
                )
            self.get_logger().info("Calling GPT to generate prefer_path function...")

            response = openai.ChatCompletion.create(
                model='gpt-5.2',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                reasoning_effort="none",
                max_completion_tokens=700
            )
            reply = response['choices'][0]['message']['content'].strip()
            # Inject u,v -> int so "u in avoid_nodes" never sees unhashable list (path_generator may use old install)
            inject = "    u = int(u) if isinstance(u, (int, float)) else -1\n    v = int(v) if isinstance(v, (int, float)) else -1\n"
            if "def prefer_path(" in reply:
                reply = re.sub(r"(def prefer_path\s*\([^)]+\)\s*:)", r"\1\n" + inject, reply, count=1)
            self.get_logger().info("Generated Python Function:")
            self.get_logger().info(reply)

            # 4. Publish to /prefer_path_function
            output_msg = String()
            output_msg.data = reply
            self.path_function_publisher.publish(output_msg)
            self.get_logger().info("Published prefer_path_function to /prefer_path_function topic.")

        except Exception as e:
            self.get_logger().error(f'Path Weighter - GPT failed: {e}')


def initial_setting():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    lang_package_name = 'bd_nav'
    prompts_file = 'assets/path_weighter.txt'
    lang_package_share_dir = get_package_share_directory(lang_package_name)
    prompt_txt_file_path = os.path.join(lang_package_share_dir, prompts_file)

    with open(prompt_txt_file_path, encoding='utf-8-sig') as f:
        prompt_template = f.read()

    return prompt_template


def main(args=None):
    prompt_template = initial_setting()

    rclpy.init(args=args)
    node = PathWeighterNode(prompt_template)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Path Weighter Node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
