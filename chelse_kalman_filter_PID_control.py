import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt


playbagfile = True
bag_path = '/home/chelse/ros2_ws/src/ME531_project_chelse/ME531_project_chelse/flex_data/flex_data/flex_data.db3'

class Me531Project(Node):
    def __init__(self):
        super().__init__('me531_project')

        self.x_history = []
        self.y_history = []
        self.distance_history = []
        self.direction_history = []
        self.raw_flex_history = []    # Store raw flex readings
        self.filtered_flex_history = []  # Store filtered flex readings (Kalman state)

        # Add matplotlib figure and axis placeholders
        self.fig = None
        self.ax = None

        if playbagfile: 
            self.bag_path = bag_path
        else:
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
        n = 4 # Number of outputs/state variables
        m = 4 # Number of input measurements
        self.z = np.zeros((m, 1)) #[x, y] - measurements
        self.x = np.zeros((n, 1)) # Output estimate of state variables
        #self.R = np.ones((m, m)) # Measurement covariance matrix - Input
        self.R = np.eye(m) *0.05
        self.P = np.ones((n, n)) # Estimate covariance matrix
        #self.H = np.ones((m, n)) # State to measurement matrix - System Model
        self.H = np.eye(n)
        # self.A = np.ones((n, n)) # State transition matrix - system model
        self.A = np.eye(n)
        #self.Q = np.ones((n, n)) # Process noise covariance matrix - system model
        self.Q = np.eye(n) * 0.05

    def kalman_update(self, measurement):
        # Need to check what measurements look like, ensure as expected
        self.z = measurement
        # Initial update and covariance

        #[nx1] = [nxn][nx1]
        self.x_p = np.dot(self.A, self.x)
        self.P_p = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # Compute Kalman Gain
        self.K = np.dot(np.dot(self.P, self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.transpose()) + self.R))
        
        self.x = self.x_p + np.dot(self.K, (self.z - np.dot(self.H, self.x_p)))
        self.P = self.P_p - np.dot(np.dot(self.K, self.H), self.P_p)

    def process_flex_data(self, values):
        measurement = np.array(values).reshape((4, 1))
        self.kalman_update(measurement)

        # Save positions for plotting
        self.x_history.append(float(self.x[0]))
        self.y_history.append(float(self.x[1]))

        # Use the state estimate (position) to determine the apple's position
        apple_position_x = float(self.x[0])
        apple_position_y = float(self.x[1])

        # Calculate direction and distance to origin (0, 0)
        dx = -apple_position_x
        dy = -apple_position_y
        distance = np.sqrt(dx**2 + dy**2)

        # Determine cardinal direction
        angle_rad = np.arctan2(dy, dx)
        angle_deg = (np.degrees(angle_rad) + 360) % 360

        direction = self.get_direction_from_angle(angle_deg)

        print(f"[Kalman Position] x = {apple_position_x:.2f}, y = {apple_position_y:.2f}")
        print(f"➡ Move {direction} by {distance:.2f} units\n")

        # Update the matplotlib figure with the new info
        self.update_info_figure(raw_values=values,
                                filtered_values=self.x.flatten().tolist(),
                                gripper_x=apple_position_x,
                                gripper_y=apple_position_y,
                                direction=direction,
                                distance=distance)

        if not playbagfile:
            msg = Float32MultiArray()
            msg.data = self.x.flatten().tolist()
            self.data_publisher.publish(msg)

    def get_direction_from_angle(self, angle_deg):
        # Define sectors for cardinal directions (8-way)
        directions = [
            ("East", 0),
            ("Northeast", 45),
            ("North", 90),
            ("Northwest", 135),
            ("West", 180),
            ("Southwest", 225),
            ("South", 270),
            ("Southeast", 315)
        ]
        for i in range(len(directions)):
            name, center = directions[i]
            start = (center - 22.5) % 360
            end = (center + 22.5) % 360
            if start < end:
                if start <= angle_deg < end:
                    return name
            else:  # wrap around 0 degrees
                if angle_deg >= start or angle_deg < end:
                    return name
        return "Unknown"


    def read_bag_and_process(self):
        self.get_logger().info(f"Opening bag file: {self.bag_path}")
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        topic_type_dict = {topic.name: topic.type for topic in topic_types}
        flex_topic = '/flex_sensor_data'

        if topic_type_dict.get(flex_topic) != 'std_msgs/msg/Float32MultiArray':
            self.get_logger().error(f"Unexpected topic type for {flex_topic}")
            return

        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic == flex_topic:
                msg = deserialize_message(data, Float32MultiArray)
                self.process_flex_data(list(msg.data))

        self.plot_positions()


    def flex_callback(self, msg: Float32MultiArray):
        self.process_flex_data(list(msg.data))

    def tof_callback(self, msg: Int32):
        # Placeholder: implement this if needed later
        pass

    def update_info_figure(self, raw_values, filtered_values, gripper_x, gripper_y, direction, distance):
        # Create figure and axis on first call
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 4))
            plt.ion()  # interactive mode on
            self.fig.show()

        self.ax.clear()

        # Compose text to display
        text = (
            f"Raw values: {np.array2string(np.array(raw_values), precision=3, separator=', ')}\n"
            f"Filtered values: {np.array2string(np.array(filtered_values), precision=3, separator=', ')}\n"
            f"Gripper x position: {gripper_x:.3f}\n"
            f"Gripper y position: {gripper_y:.3f}\n"
            f"------------------------------\n"
            f"Direction to move: {direction}\n"
            f"Distance to move: {distance:.3f}"
        )

        self.ax.text(0.1, 0.5, text, fontsize=12, family='monospace', va='center', ha='left')
        self.ax.axis('off')  # hide axes

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_positions(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.x_history, label='x position')
        plt.plot(self.y_history, label='y position')
        plt.xlabel('Time step')
        plt.ylabel('Position')
        plt.title('Kalman-Filtered x and y Positions Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)  # show plot but don't block

        print("Press spacebar or enter to close the plot...")
        plt.waitforbuttonpress()  # wait for key press (True if key pressed, False if mouse click)
        plt.close()  # close plot window after key press



def main(args=None):
    rclpy.init(args=args)
    try:
        node = Me531Project()
        if playbagfile:
            node.read_bag_and_process()
        else:
            # your ROS subscriber spin code here
            executor = MultiThreadedExecutor()
            executor.add_node(node)
            executor.spin()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
