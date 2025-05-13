#!/usr/bin/env python3
# To run code open terminal 
# cd ros2_ws
# source /opt/ros/humble/setup.bash
# colcon build
# source install/setup.bash
# ros2 run ME531_project_chelse simple_node


import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
import matplotlib

# Use non-interactive backend for easier stepping
matplotlib.use("TkAgg")  # Or use "Qt5Agg" if you prefer

noise_threshold = 2
max_flex_val = 60

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class BagPlotter(Node):
    def __init__(self):
        super().__init__('bag_plotter')

        bag_path = '/home/chelse/ros2_ws/src/ME531_project_chelse/ME531_project_chelse/flex_data/flex_data/flex_data.db3'
        self.flex_data, self.flex_timestamps = self.load_flex_data(bag_path)

        # Remove index 0 from each message
        self.flex_data = [data[1:] for data in self.flex_data if len(data) >= 4]

        self.flex_array = np.array(self.flex_data)
        self.time_array = np.array(self.flex_timestamps)

        self.get_logger().info(f'Loaded {len(self.flex_array)} flex messages.')
        self.filtered_array = self.apply_kalman_filter(self.flex_array)

        # Initialize one PID per sensor
        self.pids = [PIDController(kp=1.0) for _ in range(4)]

        self.run_pid_direction_display()
        
        #self.plot_data()  # Only one plot now

    def deserialize_float32multiarray(self, serialized_bytes):
        data = serialized_bytes[4:]  # skip CDR header
        dim_count = int.from_bytes(data[0:4], 'little')
        offset = 4 + dim_count * 12 + 4  # layout and offset
        float_count = (len(data) - offset) // 4
        return list(struct.unpack('<' + 'f' * float_count, data[offset:]))

    def load_flex_data(self, bag_path):
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        flex_data = []
        timestamps = []

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic == '/flex_sensor_data':
                array = self.deserialize_float32multiarray(data)
                flex_data.append(array)
                timestamps.append(timestamp * 1e-9)  # convert ns to seconds

        # Normalize time (start at 0)
        if timestamps:
            t0 = timestamps[0]
            timestamps = [t - t0 for t in timestamps]

        return flex_data, timestamps

    def apply_kalman_filter(self, data_array):
        """
        Applies a multi-dimensional Kalman filter to the sensor data.
        Each row of data_array is a measurement vector.
        """
        n_samples, n_sensors = data_array.shape

        # Kalman filter matrices (copied from FlexToFListener)
        n = n_sensors
        m = n_sensors
        A = np.eye(n)
        H = np.eye(m)
        Q = np.eye(n) * 0.05
        R = np.eye(m) * 0.05
        P = np.ones((n, n))
        x = np.zeros((n, 1))

        filtered = np.zeros_like(data_array)

        for t in range(n_samples):
            z = data_array[t, :].reshape((m, 1))  # Measurement vector

            # Predict
            x_p = A @ x
            P_p = A @ P @ A.T + Q

            # Kalman Gain
            K = P_p @ H.T @ np.linalg.inv(H @ P_p @ H.T + R)

            # Update
            x = x_p + K @ (z - H @ x_p)
            P = P_p - K @ H @ P_p

            filtered[t, :] = x.flatten()

        return filtered

    def direction_from_flex(self, active):
        # Map combinations to directions
        combinations = {
            (1, 0, 0, 0): 'South',
            (0, 1, 0, 0): 'West',
            (0, 0, 1, 0): 'North',
            (0, 0, 0, 1): 'East',
            (1, 1, 0, 0): 'Southwest',
            (0, 1, 1, 0): 'Northwest',
            (0, 0, 1, 1): 'Northeast',
            (1, 0, 0, 1): 'Southeast',
            (1, 1, 1, 0): 'West',
            (1, 1, 0, 1): 'South',
            (0, 1, 1, 1): 'North',
            (1, 0, 1, 1): 'East',
            (1, 1, 1, 1): 'STOP'
        }
        key = tuple(int(a > noise_threshold) for a in active)
        return combinations.get(key, 'Unknown')

    def plot_direction(self, direction, raw_vals, filtered_vals, timestep):
        fig, (ax_dir, ax_data) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Time {self.time_array[timestep]:.2f}s - Direction: {direction}", fontsize=14)

        # --- Direction plot ---
        ax_dir.set_xlim(-1.5, 1.5)
        ax_dir.set_ylim(-1.5, 1.5)
        ax_dir.axis('off')

        speed = 0  # Default

        if direction == 'STOP':
            hexagon = RegularPolygon((0, 0), numVertices=6, radius=1, color='red')
            ax_dir.add_patch(hexagon)
            ax_dir.text(0, 0, 'STOP', color='white', fontsize=20, ha='center', va='center')
        elif direction != 'Unknown':
            direction_vectors = {
                'North': (0, 1),
                'South': (0, -1),
                'East': (1, 0),
                'West': (-1, 0),
                'Northeast': (1, 1),
                'Northwest': (-1, 1),
                'Southeast': (1, -1),
                'Southwest': (-1, -1)
            }

            dx_raw, dy_raw = direction_vectors.get(direction, (0, 0))

            # Determine active sensors above noise threshold
            active_flex = [val for val in raw_vals if val > noise_threshold]

            if len(active_flex) == 1:
                speed = int(active_flex[0] // 10)
            elif len(active_flex) > 1:
                avg_flex = sum(active_flex) / len(active_flex)
                speed = int(avg_flex // 10)

            # Arrow length is now 0.5 × speed for better visibility
            arrow_len = 0.5 * speed

            # Normalize direction vector and scale to arrow_len
            norm = np.hypot(dx_raw, dy_raw)
            if norm == 0 or speed == 0:
                dx, dy = 0, 0
            else:
                dx = dx_raw / norm * arrow_len
                dy = dy_raw / norm * arrow_len

            arrow = FancyArrow(0, 0, dx, dy, width=0.1, length_includes_head=True, color='blue')
            ax_dir.add_patch(arrow)

            # Show speed *under the title*, above the origin
            ax_dir.text(0, 1.2, f"Speed: {speed:.0f} cm/s", ha='center', va='bottom', fontsize=12, color='black')

        ax_dir.set_title("Direction")

        # --- Sensor value subplot ---
        sensor_indices = np.arange(1, 5)

        ax_data.plot(sensor_indices, raw_vals, 'o', color='orange', label='Raw')
        ax_data.plot(sensor_indices, filtered_vals, 's', color='blue', label='Filtered')

        ax_data.set_xticks(sensor_indices)
        ax_data.set_xticklabels([f"Sensor {i}" for i in sensor_indices])
        ax_data.set_ylabel("Sensor Reading")
        ax_data.set_ylim(0, max(max(raw_vals), max(filtered_vals)) + 5)
        ax_data.legend()
        ax_data.grid(True)
        ax_data.set_title("Flex Sensor Readings")

        plt.tight_layout()
        plt.pause(0.01)
        input("Press Enter to continue...")
        plt.close()


    def run_pid_direction_display(self):
        print("Running PID + Direction display...")

        for i in range(1, len(self.filtered_array)):
            dt = self.time_array[i] - self.time_array[i - 1]

            pid_outputs = []
            for j in range(4):
                error = self.filtered_array[i, j]  # desired = 0, so error = value
                control = self.pids[j].update(error, dt)
                pid_outputs.append(control)

            if i <= 20:
                print(f"Time {self.time_array[i]:.2f}s - PID outputs: {pid_outputs}")

            direction = self.direction_from_flex(pid_outputs)

            raw_vals = self.flex_array[i, :]
            filtered_vals = self.filtered_array[i, :]

            self.plot_direction(direction, raw_vals, filtered_vals, i)

        plt.close()



    '''
    def plot_data(self):
        if self.flex_array.size == 0:
            self.get_logger().warn("No flex data to plot.")
            return

        plt.figure(figsize=(10, 6))
        n_sensors = self.flex_array.shape[1]

        for i in range(n_sensors):
            plt.plot(self.time_array, self.flex_array[:, i], linestyle=':', label=f'Raw Sensor {i + 1}')
            plt.plot(self.time_array, self.filtered_array[:, i], linestyle='-', label=f'Filtered Sensor {i + 1}')

        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.title('Flex Sensor Readings (Raw and Filtered)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    '''

def main(args=None):
    rclpy.init(args=args)
    node = BagPlotter()
    rclpy.shutdown()

if __name__ == '__main__':
    main()