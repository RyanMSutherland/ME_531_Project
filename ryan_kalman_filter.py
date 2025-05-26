import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
import pandas as pd

class FlexToFListener(Node):
    def __init__(self, calibrate = True):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()
        self.calibrate = calibrate

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
        self.position_publisher = self.create_publisher(Float32MultiArray, '/position_data', 10)

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
        # self.H[4, 2] = 1
        # self.H = np.eye(n)
        # self.A = np.ones((n, n)) # State transition matrix - system model
        self.A = np.eye(n)
        #self.Q = np.ones((n, n)) # Process noise covariance matrix - system model
        self.Q = np.eye(n) * 0.05

        self.current_x, self.current_y = 0, 0
        self.current_x_vel, self.current_y_vel = 0, 0
        self.K_p = 1
        self.K_i = 0
        self.K_d = 0
        self.dt = 0.05 #Find real frequency
        self.prev_x_error, self.prev_y_error = 0, 0
        self.max_vel = 2

        if self.calibrate:
            self.all_data = np.zeros([1, 4]) # 4 needs to be changed to m -- test
            print(f'Init: {self.all_data}')

    def flex_callback(self, msg: Float32MultiArray):
        values = list(msg.data)  # Convert array('f', [...]) to a regular Python list
        print(f"[Flex Sensor Data] Values: {['{:.2f}'.format(v) for v in values]}")
        
        # Create a measurement vector that corresponds to changes in position.
        measurement = np.matrix([values[0], values[1], values[2], values[3]]).transpose() # [x1, y1, x2, y2]
        
        if self.calibrate:
            # Add new measurement to array
            mes = np.array([values[0], values[1], values[2], values[3]])
            self.all_data = np.append(self.all_data, [mes], axis = 0)

            #Current data just removes the first row that was needed to initialize the matrix
            current_data = self.all_data[1:]
        
            df = pd.DataFrame(current_data)
            df.to_csv("Calibration_data.csv")

            sigma = np.cov(current_data.T)

            print(f'Predicted covariance matrix: {sigma}')

        # Perform the Kalman Filter update
        self.kalman_update(measurement)

        # PID controller
        self.pid_controller(self.x)

        print(f'Predicted values: {self.x}')
        msg = Float32MultiArray()
        msg.data = self.x
        self.data_publisher.publish(msg)


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

    def pid_controller(self, predicted_location):
        error_x = predicted_location[0] - self.current_x
        error_y = predicted_location[1] - self.current_y

        derivative_x = (error_x - self.prev_x_error)/self.dt
        self.current_x_vel = self.K_p * error_x + self.K_d * derivative_x

        if abs(self.current_x_vel) > self.vel_max:
            if self.current_x_vel > 0:
                self.current_x_vel = self.vel_max
            else:
                self.current_x_vel = -self.vel_max

        derivative_y = (error_y - self.prev_y_error)/self.dt
        self.current_y_vel = self.K_p * error_y + self.K_d * derivative_y

        if abs(self.current_y_vel) > self.vel_max:
            if self.current_y_vel > 0:
                self.current_y_vel = self.vel_max
            else:
                self.current_y_vel = -self.vel_max

        self.current_x += self.current_x_vel * self.dt
        self.current_y += self.current_y_vel * self.dt



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
