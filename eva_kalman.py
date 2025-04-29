# this is my kalman filter file, doesn't 100% work but it sort of works
#  - definitely could tweak things to make improvements -Eva

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np

# UP RIGHT DOWN LEFT

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

        # Kalman Filter Initialization
        self.state = np.zeros(2)  # [x-position, y-position]
        self.P = np.eye(2)  # Initial error covariance
        self.Q = np.eye(2) * 1e-4  # Process noise covariance CHANGE THIS TO OPTIMIZE?
        self.R = np.eye(4) * 0.05  # Measurement noise covariance CHANGE THIS TO OPTIMIZE?
        self.H = np.zeros((4, 2))  # Measurement matrix
        self.H[0, 1] = 1  # Whisker 1 influences y-position (positive)
        self.H[1, 0] = 1  # Whisker 2 influences x-position (positive)
        self.H[2, 1] = -1 # Whisker 3 influences y-position (negative)
        self.H[3, 0] = -1 # Whisker 4 influences x-position (negative)

    def flex_callback(self, msg: Float32MultiArray):
        values = list(msg.data)  # Convert array('f', [...]) to a regular Python list
        print(f"[Flex Sensor Data] Values: {['{:.2f}'.format(v) for v in values]}")
        
        # Create a measurement vector that corresponds to changes in position.
        measurement = np.array([values[0], values[1], values[2], values[3]])  # [x1, y1, x2, y2]

        # Perform the Kalman Filter update
        self.kalman_update(measurement)

        # Use the state estimate (position) to determine the apple's position
        apple_position_x = self.state[0]
        apple_position_y = self.state[1]

        if apple_position_y > 1.0:
            vertical_position = "Up"
        elif apple_position_y < -1.0:
            vertical_position = "Down"
        else:
            vertical_position = ""

        if apple_position_x > 1.0:
            horizontal_position = "Right"
        elif apple_position_x < -1.0:
            horizontal_position = "Left"
        else:
            horizontal_position = ""
        
        print(f"Apple Position: {vertical_position} {horizontal_position} ({apple_position_x:.2f}, {apple_position_y:.2f})")

    def tof_callback(self, msg: Int32):
        # print(f"[ToF Sensor Data] Distance: {msg.data} mm")
        # this isn't needed right now, can implement later
        pass

    def kalman_update(self, measurement):
        # Prediction step
        A = np.array([[1, 0], [0, 1]])
        self.state = np.dot(A, self.state)
        self.P = np.dot(np.dot(A, self.P), A.T) + self.Q

        # Normalize the measurements
        max_value = np.max(measurement)
        if max_value > 0:
            norm_measurement = measurement / max_value
        else:
            norm_measurement = measurement  # Avoid division by zero

        # Mask out the three lowest flex values
        lowest_indices = np.argsort(norm_measurement)[:3]
        norm_measurement[lowest_indices] = 0.0

        # Re-scale the measurement back
        filtered_measurement = norm_measurement * max_value

        # Decay towards zero if maximum flex is small CHANGE THESE TO OPTIMIZE?
        decay_factor = 1.0
        if max_value < 5.0:
            self.state *= (1 - decay_factor)

        # Measurement update
        y = filtered_measurement - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = np.dot(np.eye(2) - np.dot(K, self.H), self.P)





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
