import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
import pandas as pd

class FlexToFListener(Node):
    def __init__(self, calibrate=False):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()
        self.calibrate = calibrate

        self.flex_subscriber = self.create_subscription(
            Float32MultiArray,
            '/flex_sensor_data',
            self.flex_callback,
            10,
            callback_group=self.cbgroup
        )

        self.tof_subscriber = self.create_subscription(
            Int32,
            '/tof_sensor_data',
            self.tof_callback,
            10,
            callback_group=self.cbgroup
        )

        self.apple_publisher = self.create_publisher(Float32MultiArray, '/position_apple', 10)
        self.gripper_publisher = self.create_publisher(Float32MultiArray, '/position_gripper', 10)

        self.get_logger().info('FlexToFListener node has been started.')

        # Kalman filter setup
        n = 2
        m = 4
        self.z = np.zeros((m, 1))
        self.x = np.zeros((n, 1))
        self.R = np.eye(m) * 0.05
        self.P = np.ones((n, n))
        self.H = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])
        self.A = np.eye(n)
        self.Q = np.eye(n) * 0.05

        # Controller and state
        self.current_x, self.current_y = 0.0, 0.0
        self.current_x_vel, self.current_y_vel = 0.0, 0.0
        self.smoothed_x, self.smoothed_y = 0.0, 0.0
        self.K_p = 0.5
        self.K_i = 0.0
        self.K_d = 0.0
        self.dt = 0.01
        self.prev_time = self.get_clock().now().nanoseconds * 1e-9
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_x_error = 0.0
        self.prev_y_error = 0.0
        self.vel_max = 2.0
        self.scale = 4.0
        self.alpha = 0.5  # increased smoothing alpha
        self.deadband = 0.05  # increased deadband threshold

        if self.calibrate:
            self.all_data = np.zeros([1, 4])

    def flex_callback(self, msg: Float32MultiArray):
        values = list(msg.data)
        measurement = np.array([
            values[0]/self.scale,
            values[1]/self.scale,
            values[2]/self.scale,
            values[3]/self.scale
        ]).reshape((4, 1))

        if self.calibrate:
            mes = np.array([values[0], values[1], values[2], values[3]])
            self.all_data = np.append(self.all_data, [mes], axis=0)
            current_data = self.all_data[1:]
            df = pd.DataFrame(current_data)
            df.to_csv("Calibration_data.csv")
            sigma = np.cov(current_data.T)
            print(f'Predicted covariance matrix: {sigma}')

        self.kalman_update(measurement)
        self.pid_controller(self.x)

        # Publish apple position
        apple_msg = Float32MultiArray()
        apple_msg.data = [float(self.x[1]), float(self.x[0])]
        self.apple_publisher.publish(apple_msg)

        # Publish gripper velocity
        gripper_msg = Float32MultiArray()
        gripper_msg.data = [float(self.current_x_vel), float(self.current_y_vel)]
        self.gripper_publisher.publish(gripper_msg)

    def tof_callback(self, msg: Int32):
        pass  # Reserved for future use

    def kalman_update(self, measurement):
        self.z = measurement
        self.x_p = np.dot(self.A, self.x)
        self.P_p = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        K = np.dot(np.dot(self.P_p, self.H.T),
                   np.linalg.inv(np.dot(self.H, np.dot(self.P_p, self.H.T)) + self.R))
        self.x = self.x_p + np.dot(K, self.z - np.dot(self.H, self.x_p))
        self.P = self.P_p - np.dot(np.dot(K, self.H), self.P_p)



    def pid_controller(self, predicted_location):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self.prev_time
        dt = np.clip(dt, 0.001, 0.02)  # clamp dt between 1ms and 20ms
        if dt <= 0:
            dt = self.dt
        self.prev_time = now

        # Exponential smoothing of predicted position with higher alpha
        self.smoothed_x = self.alpha * predicted_location[1, 0] + (1 - self.alpha) * self.smoothed_x
        self.smoothed_y = self.alpha * predicted_location[0, 0] + (1 - self.alpha) * self.smoothed_y

        # Error calculation with increased deadband
        error_x = self.smoothed_x - self.current_x
        error_y = self.smoothed_y - self.current_y

        if abs(error_x) < self.deadband:
            error_x = 0.0
        if abs(error_y) < self.deadband:
            error_y = 0.0

        # Integral update
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt

        # Derivative (zero since K_d=0)
        derivative_x = (error_x - self.prev_x_error) / dt
        derivative_y = (error_y - self.prev_y_error) / dt

        # PID output
        x_vel_cmd = self.K_p * error_x + self.K_i * self.integral_x + self.K_d * derivative_x
        y_vel_cmd = self.K_p * error_y + self.K_i * self.integral_y + self.K_d * derivative_y

        # Clip to max velocities
        x_vel_cmd = np.clip(x_vel_cmd, -self.vel_max, self.vel_max)
        y_vel_cmd = np.clip(y_vel_cmd, -self.vel_max, self.vel_max)

        # Store velocities
        self.current_x_vel = x_vel_cmd
        self.current_y_vel = y_vel_cmd

        # Update position by integrating velocity commands
        self.current_x += self.current_x_vel * dt
        self.current_y += self.current_y_vel * dt

        self.prev_x_error = error_x
        self.prev_y_error = error_y


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
