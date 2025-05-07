import sys
import cv2
import json
import numpy as np
import mediapipe as mp
import time
import math
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
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

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PoseLandmark = mp.solutions.pose.PoseLandmark

# CoM 가중치
BODY_PARTS = {
    "head": {"idx": [0], "w": 8.26},
    "torso": {"idx": [11, 12, 23, 24], "w": 50.0},
    "upper_arm": {"idx": [11, 12], "w": 5.0},
    "forearm": {"idx": [13, 14], "w": 3.0},
    "hand": {"idx": [15, 16], "w": 1.0},
    "thigh": {"idx": [23, 24], "w": 10.0},
    "shank": {"idx": [25, 26], "w": 4.5},
    "foot": {"idx": [27, 28], "w": 1.5},
}

# 관절 각도 계산을 위한 정의
JOINT_ANGLES = {
    "left_elbow": [11, 13, 15],  # 어깨-팔꿈치-손목 (좌)
    "right_elbow": [12, 14, 16],  # 어깨-팔꿈치-손목 (우)
    "left_knee": [23, 25, 27],  # 엉덩이-무릎-발목 (좌)
    "right_knee": [24, 26, 28],  # 엉덩이-무릎-발목 (우)
    "left_shoulder_flex": [23, 11, 13],  # 엉덩이-어깨-팔꿈치 (좌)
    "right_shoulder_flex": [24, 12, 14],  # 엉덩이-어깨-팔꿈치 (우)
}

# 거리 측정을 위한 정의
DISTANCE_PAIRS = {
    "foot_L_to_foot_R": [31, 32],  # 왼발-오른발
}


class PoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D + 3D Pose Viewer with CoM & Timed JSON Dump")
        self.resize(800, 1000)

        # 2D 뷰
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        # 3D 뷰
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 캡처 & Pose 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다.")
        self.pose = mp_pose.Pose(model_complexity=1)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 마지막 결과 저장
        self.last_results = None

        # 시간 및 속도 계산을 위한 변수
        self.start_time = time.time()  # 프로그램 시작 시간
        self.prev_time = self.start_time  # 이전 프레임 시간
        self.prev_com = None
        self.prev_landmarks = None

        # 전체 JSON 데이터 배열
        self.json_data = []

        # 초 단위 JSON 덤프 타이머 설정
        self.json_file = "pose_metrics.json"
        self.json_timer = QTimer()
        self.json_timer.timeout.connect(self.dump_json_to_file)
        self.json_timer.start(1000)  # 1000ms = 1초

        # 프로그램 종료 시 최종 JSON 저장
        self.final_save_timer = QTimer()
        self.final_save_timer.setSingleShot(True)
        self.final_save_timer.timeout.connect(self.save_final_json)

    def calculate_angle(self, a, b, c):
        """세 점 사이의 각도 계산 (도)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # 수치 오류 방지
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def calculate_distance(self, a, b):
        """두 점 사이의 유클리드 거리"""
        return np.linalg.norm(np.array(a) - np.array(b))

    def calculate_torso_orientation(self, landmarks):
        """몸통의 yaw, pitch 각도 계산"""
        # 어깨 중심점 계산
        left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # 엉덩이 중심점 계산
        left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
        right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
        hip_center = (left_hip + right_hip) / 2

        # 몸통 벡터 (엉덩이→어깨)
        torso_vector = shoulder_center - hip_center

        # 전방 벡터 (좌→우 엉덩이에 수직)
        hip_vector = right_hip - left_hip
        front_vector = np.cross(hip_vector, [0, 1, 0])
        front_vector = front_vector / np.linalg.norm(front_vector)

        # Yaw 계산 (수평면에서의 회전)
        torso_horizontal = np.array([torso_vector[0], 0, torso_vector[2]])
        if np.linalg.norm(torso_horizontal) > 0:
            torso_horizontal = torso_horizontal / np.linalg.norm(torso_horizontal)
            front_horizontal = np.array([front_vector[0], 0, front_vector[2]])
            front_horizontal = front_horizontal / np.linalg.norm(front_horizontal)

            cos_yaw = np.dot(torso_horizontal, front_horizontal)
            cos_yaw = np.clip(cos_yaw, -1.0, 1.0)
            yaw = np.degrees(np.arccos(cos_yaw))

            # 왼쪽/오른쪽 방향 결정
            cross_product = np.cross(front_horizontal, torso_horizontal)
            if cross_product[1] < 0:
                yaw = -yaw
        else:
            yaw = 0

        # Pitch 계산 (전후방 기울기)
        vertical = np.array([0, 1, 0])
        torso_norm = np.linalg.norm(torso_vector)
        if torso_norm > 0:
            torso_vector = torso_vector / torso_norm
            cos_pitch = np.dot(torso_vector, vertical)
            cos_pitch = np.clip(cos_pitch, -1.0, 1.0)
            pitch = 90 - np.degrees(np.arccos(cos_pitch))
        else:
            pitch = 0

        return yaw, pitch

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        self.last_results = results

        # 2D 리깅 & CoM
        if results.pose_landmarks:
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
            # 2D CoM
            sum_w = 0.0
            com2d = np.zeros(2)
            for part in BODY_PARTS.values():
                ids, wt = part["idx"], part["w"]
                pts = np.array(
                    [
                        [
                            results.pose_landmarks.landmark[i].x,
                            results.pose_landmarks.landmark[i].y,
                        ]
                        for i in ids
                    ]
                )
                com2d += pts.mean(axis=0) * wt
                sum_w += wt
            com2d /= sum_w
            px, py = int(com2d[0] * w), int(com2d[1] * h)
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(
                frame,
                "CoM",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        # 3D 렌더링
        self.ax.clear()
        self.ax.set_title("3D Pose + CoM")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")

        if results.pose_world_landmarks:
            ws = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
            )
            xs, ys, zs = ws[:, 0], ws[:, 1], ws[:, 2]
            # World CoM
            sum_w = 0.0
            com3d = np.zeros(3)
            for part in BODY_PARTS.values():
                ids, wt = part["idx"], part["w"]
                pts = ws[ids]
                com3d += pts.mean(axis=0) * wt
                sum_w += wt
            com3d /= sum_w
            # 중앙정렬
            mean_pt = ws.mean(axis=0)
            xs_c, ys_c, zs_c = xs - mean_pt[0], ys - mean_pt[1], zs - mean_pt[2]
            com_c = com3d - mean_pt
            for s, e in POSE_CONNECTIONS_3D:
                self.ax.plot(
                    [xs_c[s], xs_c[e]], [zs_c[s], zs_c[e]], [-ys_c[s], -ys_c[e]], c="b"
                )
            self.ax.scatter(xs_c, zs_c, -ys_c, c="r", s=20)
            self.ax.scatter(
                [com_c[0]], [com_c[2]], [-com_c[1]], c="yellow", s=80, label="CoM"
            )
            self.ax.legend(loc="upper right")
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)

        qt_img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        self.canvas.draw()

    def dump_json_to_file(self):
        res = self.last_results
        if not res or not (res.pose_landmarks and res.pose_world_landmarks):
            return

        # 현재 시간 정보
        current_time = time.time()
        time_delta = current_time - self.prev_time
        elapsed_time = current_time - self.start_time  # 시작부터 경과한 시간

        # 기본 JSON 구조 생성
        frame_data = {
            "t": round(
                elapsed_time, 2
            ),  # 초 단위 타임스탬프 (시작 시간부터의 경과 시간)
            "landmarks": {},
            "metrics": {
                "com": {},
                "torso": {},
                "joint_angles": {},
                "velocities": {},
                "distances": {},
            },
        }

        # 1. 3D 랜드마크 정보 (필요한 주요 관절만)
        landmarks_3d = res.pose_world_landmarks.landmark
        key_landmarks = {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_foot": 31,
            "right_foot": 32,
        }

        for name, idx in key_landmarks.items():
            lm = landmarks_3d[idx]
            frame_data["landmarks"][name] = {"x": lm.x, "y": lm.y, "z": lm.z}

        # 2. CoM 계산
        com3d = np.zeros(3)
        sum_w = 0.0
        for part in BODY_PARTS.values():
            ids, wt = part["idx"], part["w"]
            pts = np.array(
                [[landmarks_3d[i].x, landmarks_3d[i].y, landmarks_3d[i].z] for i in ids]
            )
            com3d += pts.mean(axis=0) * wt
            sum_w += wt
        com3d /= sum_w

        frame_data["metrics"]["com"] = {
            "x": float(com3d[0]),
            "y": float(com3d[1]),
            "z": float(com3d[2]),
        }

        # 3. 몸통 방향 (yaw, pitch)
        yaw, pitch = self.calculate_torso_orientation(landmarks_3d)
        frame_data["metrics"]["torso"] = {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
        }

        # 4. 관절 각도
        for name, points in JOINT_ANGLES.items():
            p1 = [
                landmarks_3d[points[0]].x,
                landmarks_3d[points[0]].y,
                landmarks_3d[points[0]].z,
            ]
            p2 = [
                landmarks_3d[points[1]].x,
                landmarks_3d[points[1]].y,
                landmarks_3d[points[1]].z,
            ]
            p3 = [
                landmarks_3d[points[2]].x,
                landmarks_3d[points[2]].y,
                landmarks_3d[points[2]].z,
            ]
            angle = self.calculate_angle(p1, p2, p3)
            frame_data["metrics"]["joint_angles"][name] = round(angle, 1)

        # 5. 속도 계산
        if self.prev_com is not None and time_delta > 0:
            # CoM 수직 속도
            com_y_velocity = (com3d[1] - self.prev_com[1]) / time_delta
            frame_data["metrics"]["velocities"]["com_y"] = round(com_y_velocity, 2)

            # 손 속도
            if self.prev_landmarks is not None:
                # 왼손 속도
                prev_hand_l = [
                    self.prev_landmarks[15].x,
                    self.prev_landmarks[15].y,
                    self.prev_landmarks[15].z,
                ]
                curr_hand_l = [
                    landmarks_3d[15].x,
                    landmarks_3d[15].y,
                    landmarks_3d[15].z,
                ]
                dist_l = self.calculate_distance(prev_hand_l, curr_hand_l)
                hand_l_velocity = dist_l / time_delta
                frame_data["metrics"]["velocities"]["hand_L"] = round(
                    hand_l_velocity, 2
                )

                # 오른손 속도
                prev_hand_r = [
                    self.prev_landmarks[16].x,
                    self.prev_landmarks[16].y,
                    self.prev_landmarks[16].z,
                ]
                curr_hand_r = [
                    landmarks_3d[16].x,
                    landmarks_3d[16].y,
                    landmarks_3d[16].z,
                ]
                dist_r = self.calculate_distance(prev_hand_r, curr_hand_r)
                hand_r_velocity = dist_r / time_delta
                frame_data["metrics"]["velocities"]["hand_R"] = round(
                    hand_r_velocity, 2
                )
        else:
            # 초기값
            frame_data["metrics"]["velocities"]["com_y"] = 0.0
            frame_data["metrics"]["velocities"]["hand_L"] = 0.0
            frame_data["metrics"]["velocities"]["hand_R"] = 0.0

        # 6. 주요 거리
        # 발 간 거리
        left_foot = [landmarks_3d[31].x, landmarks_3d[31].y, landmarks_3d[31].z]
        right_foot = [landmarks_3d[32].x, landmarks_3d[32].y, landmarks_3d[32].z]
        foot_distance = self.calculate_distance(left_foot, right_foot)
        frame_data["metrics"]["distances"]["foot_L_to_foot_R"] = round(foot_distance, 2)

        # 손과 가상 홀드 간 거리 (예시 - 실제로는 홀드 위치 필요)
        # 여기서는 간단한 예시로 가상의 고정된 홀드 위치 사용
        virtual_hold = [0.2, 0.2, 0.2]  # 임의의 고정된 홀드 위치
        right_hand = [landmarks_3d[16].x, landmarks_3d[16].y, landmarks_3d[16].z]
        hold_distance = self.calculate_distance(right_hand, virtual_hold)
        frame_data["metrics"]["distances"]["hand_R_to_hold"] = round(hold_distance, 2)

        # 데이터 추가
        self.json_data.append(frame_data)

        # 이전 상태 업데이트
        self.prev_time = current_time
        self.prev_com = com3d
        self.prev_landmarks = landmarks_3d

        # 콘솔에 현재 프레임 수 출력
        print(f"Captured frame: {len(self.json_data)}, t={frame_data['t']:.2f}s")

    def save_final_json(self):
        """최종 JSON 파일 저장"""
        if self.json_data:
            # JSON 데이터를 시간순으로 정렬
            self.json_data.sort(key=lambda x: x["t"])

            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.json_data)} frames to {self.json_file}")

    def keyPressEvent(self, event):
        # ESC로 종료
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        # 종료 전 JSON 저장
        self.save_final_json()
        self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PoseApp()
    win.show()
    sys.exit(app.exec_())
