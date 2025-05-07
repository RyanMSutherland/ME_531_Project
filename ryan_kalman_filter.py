import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np

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

        self.data_publisher = self.create_publisher(Float32MultiArray, '/kalman_data', 10)

        self.get_logger().info('FlexToFListener node has been started.')

        # Kalman Filter Initialization
        n = 2 # Number of outputs/state variables
        m = 4 # Number of input measurements
        self.z = np.zeros((m, 1)) #[x, y] - measurements
        self.x = np.zeros((n, 1)) # Output estimate of state variables
        #self.R = np.ones((m, m)) # Measurement covariance matrix - Input
        self.R = np.eye(m) * 0.05
        self.P = np.ones((n, n)) # Estimate covariance matrix
        self.H = np.zeros((m, n)) # State to measurement matrix - System Model
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 0] = -1
        self.H[3, 1] = -1 
        # self.H = np.eye(n)
        # self.A = np.ones((n, n)) # State transition matrix - system model
        self.A = np.eye(n)
        #self.Q = np.ones((n, n)) # Process noise covariance matrix - system model
        self.Q = np.eye(n) * 0.05

    def flex_callback(self, msg: Float32MultiArray):
        values = list(msg.data)  # Convert array('f', [...]) to a regular Python list
        print(f"[Flex Sensor Data] Values: {['{:.2f}'.format(v) for v in values]}")
        
        # Create a measurement vector that corresponds to changes in position.
        measurement = np.matrix([values[0], values[1], values[2], values[3]]).transpose() # [x1, y1, x2, y2]
        # Perform the Kalman Filter update
        self.kalman_update(measurement)

        print(f'Predicted values: {self.x}')
        msg = Float32MultiArray()
        msg.data = self.x
        self.data_publisher.publish(msg)

        # Use the state estimate (position) to determine the apple's position
        # apple_position_x = self.x[0]
        # apple_position_y = self.x[1]

        # if apple_position_y > 1.0:
        #     vertical_position = "Up"
        # elif apple_position_y < -1.0:
        #     vertical_position = "Down"
        # else:
        #     vertical_position = ""

        # if apple_position_x > 1.0:
        #     horizontal_position = "Right"
        # elif apple_position_x < -1.0:
        #     horizontal_position = "Left"
        # else:
        #     horizontal_position = ""
        
        # print(f"Apple Position: {vertical_position} {horizontal_position} ({apple_position_x:.2f}, {apple_position_y:.2f})")

    def tof_callback(self, msg: Int32):
        # print(f"[ToF Sensor Data] Distance: {msg.data} mm")
        # this isn't needed right now, can implement later
        pass

    def kalman_update(self, measurement):
        # Need to check what measurements look like, ensure as expected
        self.z = measurement
        # Initial update and covariance

        #[nx1] = [nxn][nx1]
        self.x_p = np.dot(self.A, self.x)
        self.P_p = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Q

        # Compute Kalman Gain
        self.K = np.dot(np.dot(self.P, self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.transpose()) + self.R))
        
        self.x = self.x_p + np.dot(self.K, (self.z - np.dot(self.H, self.x_p)))
        self.P = self.P_p - np.dot(np.dot(self.K, self.H), self.P_p)


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
