#!/usr/bin/env python3
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class StampSet:
    t_user_input: Optional[float] = None
    t_user_output: Optional[float] = None
    t_prefer_path: Optional[float] = None
    t_path_request: Optional[float] = None

    # path_evaluator
    t_map_image: Optional[float] = None
    t_eval_output: Optional[float] = None

    def clear_pipeline(self):
        self.t_user_input = None
        self.t_user_output = None
        self.t_prefer_path = None
        self.t_path_request = None

    def clear_eval(self):
        self.t_map_image = None
        self.t_eval_output = None

    def clear_all(self):
        self.clear_pipeline()
        self.clear_eval()

    def pipeline_ready(self) -> bool:
        return all(
            t is not None
            for t in (self.t_user_input, self.t_user_output, self.t_prefer_path, self.t_path_request)
        )

    def eval_ready(self) -> bool:
        return self.t_map_image is not None and self.t_eval_output is not None


class PipelineProfiler(Node):
    """
    Stage latencies by topic arrival times.

    - intent_classifier: /user_input -> /user_output
    - path_weighter:     /user_output -> /prefer_path_function
    - path_generator:    /prefer_path_function -> /path_request
    - total(planning):   /user_input -> /path_request

    - path_evaluator:    /map_image -> /eval_output
    - total(+eval):      /user_input -> /eval_output  (if both exist in same cycle)
    """

    def __init__(self):
        super().__init__("pipeline_profiler")

        self.s = StampSet()

        self.create_subscription(String, "/user_input", self.cb_user_input, 50)
        self.create_subscription(String, "/user_output", self.cb_user_output, 50)
        self.create_subscription(String, "/prefer_path_function", self.cb_prefer_path, 50)
        self.create_subscription(String, "/path_request", self.cb_path_request, 50)

        # path_evaluator hooks
        self.create_subscription(String, "/map_image", self.cb_map_image, 50)
        self.create_subscription(String, "/eval_output", self.cb_eval_output, 50)

        self.get_logger().info("pipeline_profiler started.")

    def now(self) -> float:
        return time.perf_counter()

    # ---------- Planning pipeline ----------
    def cb_user_input(self, msg: String):
        # New planning cycle
        self.s.clear_all()
        self.s.t_user_input = self.now()
        self.get_logger().info("Cycle start: /user_input received.")

    def cb_user_output(self, msg: String):
        if self.s.t_user_input is None:
            return
        if self.s.t_user_output is None:
            self.s.t_user_output = self.now()

    def cb_prefer_path(self, msg: String):
        if self.s.t_user_output is None:
            return
        if self.s.t_prefer_path is None:
            self.s.t_prefer_path = self.now()

    def cb_path_request(self, msg: String):
        if self.s.t_prefer_path is None:
            return
        if self.s.t_path_request is None:
            self.s.t_path_request = self.now()

        if self.s.pipeline_ready():
            self.report_planning()

    # ---------- Evaluator ----------
    def cb_map_image(self, msg: String):
        # record first map image timestamp after planning begins
        if self.s.t_user_input is None:
            return
        if self.s.t_map_image is None:
            self.s.t_map_image = self.now()

    def cb_eval_output(self, msg: String):
        if self.s.t_map_image is None:
            return
        if self.s.t_eval_output is None:
            self.s.t_eval_output = self.now()

        if self.s.eval_ready():
            self.report_eval_and_reset()

    # ---------- Reports ----------
    def report_planning(self):
        t0, t1, t2, t3 = (
            self.s.t_user_input,
            self.s.t_user_output,
            self.s.t_prefer_path,
            self.s.t_path_request,
        )
        if None in (t0, t1, t2, t3):
            return

        intent_ms = (t1 - t0) * 1000.0
        weighter_ms = (t2 - t1) * 1000.0
        generator_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        self.get_logger().info(
            "Planning Latency(ms) | intent: %.1f | weighter: %.1f | generator: %.1f | total: %.1f"
            % (intent_ms, weighter_ms, generator_ms, total_ms)
        )

    def report_eval_and_reset(self):
        tm, te = self.s.t_map_image, self.s.t_eval_output
        if tm is None or te is None:
            return

        eval_ms = (te - tm) * 1000.0
        self.get_logger().info("Eval Latency(ms) | path_evaluator: %.1f" % (eval_ms))

        # optional end-to-end including eval, if same cycle
        if self.s.t_user_input is not None:
            total_with_eval_ms = (te - self.s.t_user_input) * 1000.0
            self.get_logger().info("Total Latency(ms) | /user_input -> /eval_output: %.1f" % total_with_eval_ms)

        # reset for next cycle
        self.s.clear_all()


def main():
    rclpy.init()
    node = PipelineProfiler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
