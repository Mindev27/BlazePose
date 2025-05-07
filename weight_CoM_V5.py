import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 3D 연결 정보
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
mp_drawing = mp.solutions.drawing_utils


class PoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D + 3D Pose Viewer with Correct CoM")
        self.resize(800, 1000)

        # 2D 뷰
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        # 3D 뷰
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")

        # 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 카메라 & Pose
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다.")
        self.pose = mp_pose.Pose(model_complexity=1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # --- 1) 2D 리깅 & 2D CoM 계산 ---
        if results.pose_landmarks:
            # draw skeleton
            for s, e in mp_pose.POSE_CONNECTIONS:
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

            # normalized 이미지 CoM
            body_parts = {
                "head": {"idx": [0], "w": 8.26},
                "torso": {"idx": [11, 12, 23, 24], "w": 50.0},
                "upper_arm": {"idx": [11, 12], "w": 5.0},
                "forearm": {"idx": [13, 14], "w": 3.0},
                "hand": {"idx": [15, 16], "w": 1.0},
                "thigh": {"idx": [23, 24], "w": 10.0},
                "shank": {"idx": [25, 26], "w": 4.5},
                "foot": {"idx": [27, 28], "w": 1.5},
            }
            sum_w = 0.0
            com2d_x = com2d_y = 0.0
            for part in body_parts.values():
                ids, wt = part["idx"], part["w"]
                avg_x = np.mean([results.pose_landmarks.landmark[i].x for i in ids])
                avg_y = np.mean([results.pose_landmarks.landmark[i].y for i in ids])
                com2d_x += avg_x * wt
                com2d_y += avg_y * wt
                sum_w += wt
            com2d_x /= sum_w
            com2d_y /= sum_w

            # 화면 픽셀로
            px = int(com2d_x * w)
            py = int(com2d_y * h)
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"CoM",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # --- 2) 3D PoSE & 3D CoM 계산 & 시각화 ---
        self.ax.clear()
        self.ax.set_title("3D Pose + CoM")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")
        if results.pose_world_landmarks:
            # raw world coords
            xs = np.array([lm.x for lm in results.pose_world_landmarks.landmark])
            ys = np.array([lm.y for lm in results.pose_world_landmarks.landmark])
            zs = np.array([lm.z for lm in results.pose_world_landmarks.landmark])

            # world CoM (on world coords)
            sum_w = 0.0
            com3d = np.zeros(3)
            for part in body_parts.values():
                ids, wt = part["idx"], part["w"]
                p3 = np.array(
                    [
                        np.mean(
                            [results.pose_world_landmarks.landmark[i].x for i in ids]
                        ),
                        np.mean(
                            [results.pose_world_landmarks.landmark[i].y for i in ids]
                        ),
                        np.mean(
                            [results.pose_world_landmarks.landmark[i].z for i in ids]
                        ),
                    ]
                )
                com3d += p3 * wt
                sum_w += wt
            com3d /= sum_w

            # center everything by mean
            mx, my, mz = xs.mean(), ys.mean(), zs.mean()
            xs_c = xs - mx
            ys_c = ys - my
            zs_c = zs - mz
            comx_c = com3d[0] - mx
            comy_c = com3d[1] - my
            comz_c = com3d[2] - mz

            # draw skeleton in 3D
            for s, e in POSE_CONNECTIONS_3D:
                self.ax.plot(
                    [xs_c[s], xs_c[e]], [zs_c[s], zs_c[e]], [-ys_c[s], -ys_c[e]], c="b"
                )
            self.ax.scatter(xs_c, zs_c, -ys_c, c="r", s=20)

            # draw CoM in 3D
            self.ax.scatter(
                [comx_c], [comz_c], [-comy_c], c="yellow", s=80, label="CoM"
            )
            self.ax.legend(loc="upper right")
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)

        # 2D/3D 업데이트
        qt_img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        self.canvas.draw()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PoseApp()
    win.show()
    sys.exit(app.exec_())
