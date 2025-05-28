import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import time
import math

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_position_publisher')

        # Publishers
        self.pub_x_apple = self.create_publisher(TransformStamped, '/x_position_apple', 10)
        self.pub_x_gripper = self.create_publisher(TransformStamped, '/x_position_gripper', 10)
        self.pub_y_apple = self.create_publisher(TransformStamped, '/y_position_apple', 10)
        self.pub_y_gripper = self.create_publisher(TransformStamped, '/y_position_gripper', 10)

        # Timer to publish messages at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_data)
        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]

    def create_msg(self, x=0.0, y=0.0):
        msg = TransformStamped()
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()
        msg.transform.translation.x = x
        msg.transform.translation.y = y
        return msg

    def publish_data(self):
        t = self.get_clock().now().seconds_nanoseconds()[0] - self.start_time
        # Generate some wave-based fake positions
        x_apple = 1.
        x_gripper = 2.
        y_apple = 3.
        y_gripper = 4.

        self.pub_x_apple.publish(self.create_msg(x=x_apple))
        self.pub_x_gripper.publish(self.create_msg(x=x_gripper))
        self.pub_y_apple.publish(self.create_msg(y=y_apple))
        self.pub_y_gripper.publish(self.create_msg(y=y_gripper))

def main():
    rclpy.init()
    node = TestPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
