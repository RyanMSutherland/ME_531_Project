# this file just gets data from the desired topics, doesn't do any kalman filter stuff. may be helpful

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray, Int32

class FlexToFListener(Node):
    def __init__(self):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()

        # Subscribe to flex sensor data
        self.flex_subscriber = self.create_subscription(
            Float32MultiArray,
            '/flex_sensor_data',
            self.flex_callback,
            10,
            callback_group=self.cbgroup
        )

        # Subscribe to ToF sensor data
        self.tof_subscriber = self.create_subscription(
            Int32,
            '/tof_sensor_data',
            self.tof_callback,
            10,
            callback_group=self.cbgroup
        )

        self.get_logger().info('FlexToFListener node has been started.')

    def flex_callback(self, msg: Float32MultiArray):
        values = list(msg.data)  # Convert array('f', [...]) to a regular Python list
        print(f"[Flex Sensor Data] Values: {['{:.2f}'.format(v) for v in values]}")
        # return values

    def tof_callback(self, msg: Int32):
        print(f"[ToF Sensor Data] Distance: {msg.data} mm")
        # return msg.data


def main():
    rclpy.init()
    node = FlexToFListener()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()