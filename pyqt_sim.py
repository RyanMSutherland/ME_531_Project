import sys
import math
import rclpy
from rclpy.node import Node
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Int32
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from threading import Thread
from time import sleep

class PlotSubscriber(Node):
    def __init__(self):
        super().__init__('plot_subscriber')

        self.time_reference = None
        self.max_time_window = 60.0

        self.times = []

        self.x_apple = []
        self.x_gripper = []
        self.y_apple = []
        self.y_gripper = []
        self.z_apple = []  # NEW: z-apple

        self.create_subscription(TransformStamped, '/x_position_apple', self.x_apple_callback, 10)
        self.create_subscription(TransformStamped, '/x_position_gripper', self.x_gripper_callback, 10)
        self.create_subscription(TransformStamped, '/y_position_apple', self.y_apple_callback, 10)
        self.create_subscription(TransformStamped, '/y_position_gripper', self.y_gripper_callback, 10)
        self.create_subscription(Int32, '/tof_sensor_data', self.z_apple_callback, 10)

    def process_msg(self, msg, target_list):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.time_reference is None:
            self.time_reference = timestamp
        t = timestamp - self.time_reference

        if len(self.times) == 0 or t > self.times[-1]:
            self.times.append(t)
            # Default to previous values or 0.0
            self.x_apple.append(self.x_apple[-1] if self.x_apple else 0.0)
            self.x_gripper.append(self.x_gripper[-1] if self.x_gripper else 0.0)
            self.y_apple.append(self.y_apple[-1] if self.y_apple else 0.0)
            self.y_gripper.append(self.y_gripper[-1] if self.y_gripper else 0.0)
            self.z_apple.append(self.z_apple[-1] if self.z_apple else 0.0)

            # Update target list
            if target_list is self.x_apple:
                self.x_apple[-1] = msg.transform.translation.x
            elif target_list is self.x_gripper:
                self.x_gripper[-1] = msg.transform.translation.x
            elif target_list is self.y_apple:
                self.y_apple[-1] = msg.transform.translation.y
            elif target_list is self.y_gripper:
                self.y_gripper[-1] = msg.transform.translation.y
            elif target_list is self.z_apple:
                self.z_apple[-1] = msg.transform.translation.z

            # Trim history
            while self.times and self.times[0] < t - self.max_time_window:
                self.times.pop(0)
                self.x_apple.pop(0)
                self.x_gripper.pop(0)
                self.y_apple.pop(0)
                self.y_gripper.pop(0)
                self.z_apple.pop(0)

    def x_apple_callback(self, msg): self.process_msg(msg, self.x_apple)
    def x_gripper_callback(self, msg): self.process_msg(msg, self.x_gripper)
    def y_apple_callback(self, msg): self.process_msg(msg, self.y_apple)
    def y_gripper_callback(self, msg): self.process_msg(msg, self.y_gripper)
    def z_apple_callback(self, msg):
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        if self.time_reference is None:
            self.time_reference = timestamp
        t = timestamp - self.time_reference

        if len(self.times) == 0 or t > self.times[-1]:
            self.times.append(t)
            self.x_apple.append(self.x_apple[-1] if self.x_apple else 0.0)
            self.x_gripper.append(self.x_gripper[-1] if self.x_gripper else 0.0)
            self.y_apple.append(self.y_apple[-1] if self.y_apple else 0.0)
            self.y_gripper.append(self.y_gripper[-1] if self.y_gripper else 0.0)
            self.z_apple.append(msg.data)
            # print(f"Z_DATA: {msg.data}")

            # Trim history
            while self.times and self.times[0] < t - self.max_time_window:
                self.times.pop(0)
                self.x_apple.pop(0)
                self.x_gripper.pop(0)
                self.y_apple.pop(0)
                self.y_gripper.pop(0)
                self.z_apple.pop(0)


class PlotWindow(QWidget):
    def __init__(self, node: PlotSubscriber):
        super().__init__()
        self.node = node
        self.setWindowTitle("Live Position Plot")
        self.setGeometry(100, 100, 1000, 900)  # Slightly taller for 3 plots

        layout = QVBoxLayout()

        self.fig = Figure(figsize=(8, 8))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)  # NEW

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.timer_thread = Thread(target=self.update_loop, daemon=True)
        self.timer_thread.start()
        self.z_threshold = 45


    def update_loop(self):
        while rclpy.ok():
            sleep(0.1)
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()  # NEW

            times = self.node.times
            if times:
                self.ax1.plot(times, self.node.x_apple, label="X Apple", color='red', linewidth=2)
                self.ax1.plot(times, self.node.x_gripper, label="X Gripper", color='blue', linewidth=2)
                self.ax1.set_ylabel("X Position", fontsize=20)
                self.ax1.legend(fontsize=12)
                self.ax1.tick_params(axis='both', labelsize=12)

                self.ax2.plot(times, self.node.y_apple, label="Y Apple", color='red', linewidth=2)
                self.ax2.plot(times, self.node.y_gripper, label="Y Gripper", color='blue', linewidth=2)
                self.ax2.set_ylabel("Y Position", fontsize=20)
                self.ax2.legend(fontsize=12)
                self.ax2.tick_params(axis='both', labelsize=12)

                self.ax3.plot(times, self.node.z_apple, label="Z Apple", color='red', linewidth=2)
                self.ax3.set_ylabel("Z Position", fontsize=20)
                self.ax3.set_xlabel("Time (s)", fontsize=20)
                self.ax3.axhline(y=self.z_threshold, color='green', linestyle='--', label=f"Threshold ({self.z_threshold} mm)")
                self.ax3.legend(fontsize=12)
                self.ax3.tick_params(axis='both', labelsize=12)

                # Set time range
                self.ax1.set_xlim(0, 60)
                self.ax2.set_xlim(0, 60)
                self.ax3.set_xlim(0, 60)

            self.fig.suptitle("Live Apple and Gripper Position Tracking", fontsize=30)

            self.canvas.draw()

def ros_spin(node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

def main():
    rclpy.init()
    node = PlotSubscriber()

    # Start ROS spin thread
    ros_thread = Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    window = PlotWindow(node)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
