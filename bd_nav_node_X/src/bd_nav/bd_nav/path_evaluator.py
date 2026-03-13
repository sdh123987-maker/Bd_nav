# Standard library
import os
import re
import json

# Third-party libraries
import openai

# ROS 2 libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


class PathEvaluator(Node):
    def __init__(self, prompt_template):
        super().__init__('path_evaluator')
        self.prompt_template = prompt_template

        self.latest_user_input: str = ""

        # MapViewer Image & User Input Subscriber
        self.create_subscription(String, '/map_image', self.map_image_callback, 10)
        self.create_subscription(String, '/user_input', self.user_input_callback, 10)

        # Output Text Publisher
        self.eval_output_publisher = self.create_publisher(String, 'eval_output', 10)

        self.get_logger().info("PathEvaluator node is ready.")
        self.eval_stage = -1
        self._confirm_target = "eval_output"

    def user_input_callback(self, msg: String):
        """Cache the most recent user input."""
        self.latest_user_input = msg.data.strip()
        self.eval_stage = 0
        self.get_logger().info("User input updated.")

    def map_image_callback(self, msg: String):
        """Handle base64 PNG from MapViewer according to the state machine."""
        image_b64 = msg.data

        # Ignore if locked
        if self.eval_stage == -1:
            return

        # First image: trigger GPT once
        if self.eval_stage == 0:
            self.eval_stage = 1
            self.get_logger().info("Map image received (stage=0). Triggering GPT once.")
            self.listener_callback(image_b64)
            return

        # Second image: publish 'yes' to eval_output to confirm the route
        if self.eval_stage == 1:
            if getattr(self, "latest_route", None):
                from std_msgs.msg import String as RosString
                yes_msg = RosString()
                yes_msg.data = "yes"  # In eval_output_callback, treat as empty dict or "yes"
                self.eval_output_publisher.publish(yes_msg)
                self.get_logger().info("[auto-confirm] Published 'yes' to eval_output.")
                # End cycle: lock until a new user_input arrives
                self.eval_stage = -1
                self.get_logger().info("Evaluation cycle finished. eval_stage=-1 (locked)")
            else:
                # If the route hasn't been created/stored yet, wait
                self.get_logger().info("[auto-confirm] latest_route is None; waiting for next image.")
            return

    def listener_callback(self, image_b64: str):
        try:
            # Pass as a data URL scheme
            data_url = f"data:image/png;base64,{image_b64}"

            self.get_logger().info("Submitting image to OpenAI…")

            resp = openai.ChatCompletion.create(
                model="gpt-5.2",
                max_completion_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt_template},
                            {
                                "type": "text",
                                "text": f"[USER_INPUT]\n{self.latest_user_input or '(no recent user input)'}"
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ]
            )

            reply_text = resp["choices"][0]["message"]["content"].strip()
            parsed_reply = self.parse_gpt_output(reply_text)
            self.get_logger().info(f"Raw GPT Output: {reply_text}")

            msg_out = String()
            msg_out.data = json.dumps(parsed_reply, ensure_ascii=False)
            self.eval_output_publisher.publish(msg_out)
            self.get_logger().info(f"Published parsed reply to 'eval_output': {parsed_reply}")
        except Exception as e:
            self.get_logger().error(f"OpenAI request failed: {e}")

    def parse_gpt_output(self, gpt_text: str) -> dict:
        """
        Convert PathEvaluator GPT outputs (3 forms) into a user_output JSON.
        - yes            -> {}   (no change)
        - waypoint: x-y  -> {"waypoints": ["x-y"]}
        - destination: x -> {"destination": "x"}
        """
        def _first_nonempty_line(text: str) -> str:
            t = (text or "").strip()
            # Remove code fence if present
            if t.startswith("```") and t.endswith("```"):
                t = t[3:-3].strip()
            for ln in t.splitlines():
                ln = ln.strip()
                if ln:
                    return ln
            return ""
        RE_YES  = re.compile(r"^\s*yes\s*$", re.IGNORECASE)
        RE_WP   = re.compile(r"^\s*waypoint\s*:\s*(\d+)\s*(?:-|–|—|->|→)\s*(\d+)\s*$", re.IGNORECASE)
        RE_DEST = re.compile(r"^\s*dest(?:ination)?\s*:\s*(\d+)\s*$", re.IGNORECASE)

        line = _first_nonempty_line(gpt_text)

        if RE_YES.fullmatch(line):
            return {}

        m = RE_WP.fullmatch(line)
        if m:
            u, v = m.group(1), m.group(2)
            return {"waypoints": [f"{u}-{v}"]}

        m = RE_DEST.fullmatch(line)
        if m:
            node = m.group(1)
            return {"destination": node}

        # Safety net: if a JSON block is present, lightly normalize keys
        try:
            m = re.search(r'\{.*\}', gpt_text, flags=re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                out = {}
                if "destination" in obj and obj["destination"] not in (None, ""):
                    out["destination"] = str(obj["destination"]).strip()
                if "waypoint" in obj and "waypoints" not in obj:
                    obj["waypoints"] = obj.pop("waypoint")
                if "waypoints" in obj and obj["waypoints"]:
                    # Accept "x-y" or ["x-y", ...]
                    if isinstance(obj["waypoints"], str):
                        out["waypoints"] = [obj["waypoints"].strip()]
                    else:
                        out["waypoints"] = [str(w).strip() for w in obj["waypoints"] if str(w).strip()]
                if "avoid" in obj and "avoidance" not in obj:
                    obj["avoidance"] = obj.pop("avoid")
                if "avoidance" in obj and obj["avoidance"]:
                    if isinstance(obj["avoidance"], str):
                        out["avoidance"] = [obj["avoidance"].strip()]
                    else:
                        out["avoidance"] = [str(a).strip() for a in obj["avoidance"] if str(a).strip()]
                return out
        except Exception:
            pass

        # If parsing fails, return no change
        return {}


def initial_setting():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    package_name = 'bd_nav'
    prompts_file = 'assets/path_evaluator.txt'
    share_dir = get_package_share_directory(package_name)
    prompt_txt_file_path = os.path.join(share_dir, prompts_file)

    with open(prompt_txt_file_path, encoding='utf-8-sig') as f:
        prompt_template = f.read()

    return prompt_template


def main(args=None):
    prompt_template = initial_setting()

    rclpy.init(args=args)
    node = PathEvaluator(prompt_template)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down PathEvaluator...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
