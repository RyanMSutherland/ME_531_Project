import sys
import math
import rclpy
from rclpy.node import Node
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from geometry_msgs.msg import TransformStamped
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from threading import Thread
from time import sleep

class PlotSubscriber(Node):
    def __init__(self):
        super().__init__('plot_subscriber')

        self.time_reference = None
        self.max_time_window = 50.0

        self.times = []

        self.x_apple = []
        self.x_gripper = []
        self.y_apple = []
        self.y_gripper = []

        self.create_subscription(TransformStamped, '/x_position_apple', self.x_apple_callback, 10)
        self.create_subscription(TransformStamped, '/x_position_gripper', self.x_gripper_callback, 10)
        self.create_subscription(TransformStamped, '/y_position_apple', self.y_apple_callback, 10)
        self.create_subscription(TransformStamped, '/y_position_gripper', self.y_gripper_callback, 10)

    def process_msg(self, msg, target_list):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.time_reference is None:
            self.time_reference = timestamp
        t = timestamp - self.time_reference

        if len(self.times) == 0 or t > self.times[-1]:
            self.times.append(t)
            if target_list is self.x_apple:
                self.x_apple.append(msg.transform.translation.x)
                self.x_gripper.append(self.x_gripper[-1] if self.x_gripper else 0.0)
                self.y_apple.append(self.y_apple[-1] if self.y_apple else 0.0)
                self.y_gripper.append(self.y_gripper[-1] if self.y_gripper else 0.0)
            elif target_list is self.x_gripper:
                self.x_apple.append(self.x_apple[-1] if self.x_apple else 0.0)
                self.x_gripper.append(msg.transform.translation.x)
                self.y_apple.append(self.y_apple[-1] if self.y_apple else 0.0)
                self.y_gripper.append(self.y_gripper[-1] if self.y_gripper else 0.0)
            elif target_list is self.y_apple:
                self.x_apple.append(self.x_apple[-1] if self.x_apple else 0.0)
                self.x_gripper.append(self.x_gripper[-1] if self.x_gripper else 0.0)
                self.y_apple.append(msg.transform.translation.y)
                self.y_gripper.append(self.y_gripper[-1] if self.y_gripper else 0.0)
            elif target_list is self.y_gripper:
                self.x_apple.append(self.x_apple[-1] if self.x_apple else 0.0)
                self.x_gripper.append(self.x_gripper[-1] if self.x_gripper else 0.0)
                self.y_apple.append(self.y_apple[-1] if self.y_apple else 0.0)
                self.y_gripper.append(msg.transform.translation.y)

            # Trim history
            while self.times and self.times[0] < t - self.max_time_window:
                self.times.pop(0)
                self.x_apple.pop(0)
                self.x_gripper.pop(0)
                self.y_apple.pop(0)
                self.y_gripper.pop(0)

    def x_apple_callback(self, msg): self.process_msg(msg, self.x_apple)
    def x_gripper_callback(self, msg): self.process_msg(msg, self.x_gripper)
    def y_apple_callback(self, msg): self.process_msg(msg, self.y_apple)
    def y_gripper_callback(self, msg): self.process_msg(msg, self.y_gripper)

class PlotWindow(QWidget):
    def __init__(self, node: PlotSubscriber):
        super().__init__()
        self.node = node
        self.setWindowTitle("Live Position Plot")
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()

        self.fig = Figure(figsize=(8, 6))
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.timer_thread = Thread(target=self.update_loop, daemon=True)
        self.timer_thread.start()

    def update_loop(self):
        while rclpy.ok():
            sleep(0.1)
            self.ax1.clear()
            self.ax2.clear()

            times = self.node.times
            if times:
                self.ax1.plot(times, self.node.x_apple, label="X Apple", color='red', linewidth=2)
                self.ax1.plot(times, self.node.x_gripper, label="X Gripper", color='blue', linewidth=2)
                self.ax1.set_ylabel("X Position", fontsize=25)
                self.ax1.set_xlabel("Time (s)", fontsize=25)
                self.ax1.legend(fontsize=16)
                self.ax1.tick_params(axis='both', labelsize=14)

                self.ax2.plot(times, self.node.y_apple, label="Y Apple", color='red', linewidth=2)
                self.ax2.plot(times, self.node.y_gripper, label="Y Gripper", color='blue', linewidth=2)
                self.ax2.set_ylabel("Y Position", fontsize=25)
                self.ax2.set_xlabel("Time (s)", fontsize=25)
                self.ax2.legend(fontsize=16)
                self.ax2.tick_params(axis='both', labelsize=14)

                self.ax1.set_xlim(0, 60)
                self.ax2.set_xlim(0, 60)

            self.fig.suptitle("Live Apple and Gripper Position Tracking", fontsize=40)

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
