import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

POSE_CONNECTIONS_3D = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
]

mp_pose = mp.solutions.pose


class PoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D (Top) + 3D (Bottom) Pose Viewer")
        self.resize(800, 1000)

        # 2D 카메라 뷰
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        # 3D 그래프용 matplotlib 캔버스
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")

        # 레이아웃 (세로 정렬)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 카메라, Pose, Timer 설정
        self.cap = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(model_complexity=1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        h, w, _ = frame.shape

        # 2D 포즈 랜더링
        if results.pose_landmarks:
            for con in mp_pose.POSE_CONNECTIONS:
                s, e = con
                x1, y1 = int(results.pose_landmarks.landmark[s].x * w), int(
                    results.pose_landmarks.landmark[s].y * h
                )
                x2, y2 = int(results.pose_landmarks.landmark[e].x * w), int(
                    results.pose_landmarks.landmark[e].y * h
                )
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # QLabel로 2D 이미지 표시
        qt_image = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        # 3D 포즈 렌더링
        if results.pose_world_landmarks:
            xs = np.array([lm.x for lm in results.pose_world_landmarks.landmark])
            ys = np.array([lm.y for lm in results.pose_world_landmarks.landmark])
            zs = np.array([lm.z for lm in results.pose_world_landmarks.landmark])

            xs -= np.mean(xs)
            ys -= np.mean(ys)
            zs -= np.mean(zs)

            self.ax.clear()
            self.ax.set_title("3D Pose")
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Z")
            self.ax.set_zlabel("Y")

            self.ax.scatter(xs, zs, -ys, c="r")
            for s, e in POSE_CONNECTIONS_3D:
                self.ax.plot([xs[s], xs[e]], [zs[s], zs[e]], [-ys[s], -ys[e]], c="b")

            self.canvas.draw()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PoseApp()
    win.show()
    sys.exit(app.exec_())
