import os
import sys
# Qt 플러그인 경로 설정 (macOS 문제 해결)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/opt/homebrew/lib/python3.12/site-packages/PyQt5/Qt5/plugins'
import cv2
import json
import numpy as np
import mediapipe as mp
import time
import zlib
import base64
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import copy

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

# 관절 이름 매핑
JOINT_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "shoulder_left",
    12: "shoulder_right",
    13: "elbow_left",
    14: "elbow_right",
    15: "wrist_left",
    16: "wrist_right",
    17: "pinky_left",
    18: "pinky_right",
    19: "index_left",
    20: "index_right",
    21: "thumb_left",
    22: "thumb_right",
    23: "hip_left",
    24: "hip_right",
    25: "knee_left",
    26: "knee_right",
    27: "ankle_left",
    28: "ankle_right",
    29: "heel_left",
    30: "heel_right",
    31: "foot_left",
    32: "foot_right"
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

        # 재생/일시정지 상태
        self.paused = False
        
        # 비디오 탐색용 변수
        self.total_frames = 0

        # 캡처 & Pose 초기화
        # self.cap = cv2.VideoCapture(0)
        # read input.mp4 instead of webcam
        self.cap = cv2.VideoCapture("input.mp4")
        if not self.cap.isOpened():
            raise RuntimeError("영상을 열 수 없습니다.")
            
        # 총 프레임 수 가져오기
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"총 프레임 수: {self.total_frames}")
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
        self.prev_joint_positions = {}

        # 전체 JSON 데이터 배열
        self.json_data = {
            "meta": {
                "source": "BlazePose MediaPipe",
                "frame_rate": 10,  # 30fps에서 10fps로 변경
                "joints_count": 33,
                "timestamp_unit": "seconds"
            },
            "frames": []
        }

        # 초 단위 JSON 덤프 타이머 설정
        self.json_file = "pose_metrics.json"
        self.compressed_json_file = "pose_metrics_compressed.json"
        self.ultracompressed_json_file = "pose_metrics_ultracompressed.json"
        self.json_timer = QTimer()
        self.json_timer.timeout.connect(self.dump_json_to_file)
        self.json_timer.start(100)  # 100ms 간격으로 체크

        # 프로그램 종료 시 최종 JSON 저장
        self.final_save_timer = QTimer()
        self.final_save_timer.setSingleShot(True)
        self.final_save_timer.timeout.connect(self.save_final_json)
        
        # 프레임 카운터 (10fps 조절용)
        self.frame_counter = 0
        self.sample_interval = 3  # 3프레임마다 1개 샘플링 (30fps -> 10fps)

        # 움직임 감지를 위한 변수
        self.prev_key_pose = None
        self.motion_threshold = 0.05  # 움직임 감지 임계값

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

    def step_one_frame(self):
        """비디오가 일시정지된 상태에서 한 프레임만 진행"""
        if self.paused and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.process_frame(frame)
                print("한 프레임 진행")
            else:
                print("더 이상 프레임이 없습니다")

    def process_frame(self, frame):
        """프레임 처리 (update_frame 메서드에서 공통 부분 추출)"""
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        self.last_results = results

        # 프레임 카운터 증가
        self.frame_counter += 1

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

        # 현재 프레임 번호 표시
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(
            frame,
            f"Frame: {current_frame}/{self.total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # 일시정지 상태 표시
        if self.paused:
            cv2.putText(
                frame,
                "PAUSED",
                (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            
        # 키 안내 표시
        cv2.putText(
            frame,
            "Space: 재생/정지 | ←/→: 1프레임 이동 | PgUp/PgDn: 10프레임 이동 | ESC: 종료",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 3D 렌더링
        self.ax.clear()
        self.ax.set_title("3D Pose + CoM (Camera as Origin)")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")

        if results.pose_world_landmarks:
            # 카메라를 원점으로 사용 (중앙정렬하지 않음)
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
            
            # 3D 표시 (원본 좌표 사용)
            for s, e in POSE_CONNECTIONS_3D:
                self.ax.plot(
                    [xs[s], xs[e]], [zs[s], zs[e]], [-ys[s], -ys[e]], c="b"
                )
            self.ax.scatter(xs, zs, -ys, c="r", s=20)
            self.ax.scatter(
                [com3d[0]], [com3d[2]], [-com3d[1]], c="yellow", s=80, label="CoM"
            )
            self.ax.legend(loc="upper right")
            
            # 적절한 뷰 범위 설정
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(-ys), max(-ys)
            min_z, max_z = min(zs), max(zs)
            
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = (min_z + max_z) / 2
            
            max_range = max(max_x - min_x, max_y - min_y, max_z - min_z)
            padding = max_range * 0.5
            
            self.ax.set_xlim(center_x - padding, center_x + padding)
            self.ax.set_ylim(center_z - padding, center_z + padding)
            self.ax.set_zlim(center_y - padding, center_y + padding)

        qt_img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        self.canvas.draw()
        
        # 일시 정지 상태에서도 JSON 데이터는 기록
        self.dump_json_to_file()

    def update_frame(self):
        """타이머에 의해 주기적으로 호출되는 프레임 업데이트 메서드"""
        ret, frame = self.cap.read()
        if not ret:
            return
        
        self.process_frame(frame)

    def dump_json_to_file(self):
        res = self.last_results
        if not res or not (res.pose_landmarks and res.pose_world_landmarks):
            return
            
        # 10fps 샘플링을 위한 프레임 건너뛰기
        if (self.frame_counter % self.sample_interval) != 0:
            return

        # 현재 시간 정보
        current_time = time.time()
        time_delta = current_time - self.prev_time
        elapsed_time = current_time - self.start_time  # 시작부터 경과한 시간

        # 새로운 JSON 포맷에 맞게 프레임 데이터 생성
        frame_data = {
            "time": round(elapsed_time, 3),
            "joints": {},
            "features": {}
        }

        # 모든 관절 좌표 추가
        landmarks_3d = res.pose_world_landmarks.landmark
        for idx, name in JOINT_NAMES.items():
            if idx < len(landmarks_3d):
                lm = landmarks_3d[idx]
                frame_data["joints"][name] = [lm.x, lm.y, lm.z]
        
        # 엉덩이 중심점 계산
        left_hip = np.array([landmarks_3d[23].x, landmarks_3d[23].y, landmarks_3d[23].z])
        right_hip = np.array([landmarks_3d[24].x, landmarks_3d[24].y, landmarks_3d[24].z])
        hip_center = (left_hip + right_hip) / 2
        frame_data["joints"]["hip_center"] = hip_center.tolist()

        # 관절 각도
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
            frame_data["features"][f"{name}_angle"] = round(angle, 1)

        # CoM 계산
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
        frame_data["features"]["center_of_mass"] = com3d.tolist()

        # 몸통 방향 (yaw, pitch)
        yaw, pitch = self.calculate_torso_orientation(landmarks_3d)
        frame_data["features"]["torso_yaw"] = round(yaw, 1)
        frame_data["features"]["torso_pitch"] = round(pitch, 1)

        # 속도 계산
        if self.prev_com is not None and time_delta > 0:
            # CoM 속도
            com_velocity = (com3d - self.prev_com) / time_delta
            frame_data["features"]["center_of_mass_velocity"] = com_velocity.tolist()
            
            # 관절별 속도 계산
            for joint_name, position in frame_data["joints"].items():
                if joint_name in self.prev_joint_positions and time_delta > 0:
                    prev_pos = np.array(self.prev_joint_positions[joint_name])
                    curr_pos = np.array(position)
                    velocity = (curr_pos - prev_pos) / time_delta
                    frame_data["features"][f"{joint_name}_velocity"] = velocity.tolist()
            
        else:
            # 초기 속도값
            frame_data["features"]["center_of_mass_velocity"] = [0.0, 0.0, 0.0]

        # 데이터 추가
        self.json_data["frames"].append(frame_data)

        # 이전 상태 업데이트
        self.prev_time = current_time
        self.prev_com = com3d
        self.prev_landmarks = landmarks_3d
        self.prev_joint_positions = {name: pos for name, pos in frame_data["joints"].items()}

        # 콘솔에 현재 프레임 수 출력
        print(f"Captured frame: {len(self.json_data['frames'])}, t={frame_data['time']:.3f}s")

    def create_compressed_json(self):
        """압축된 JSON 포맷 생성 (기존 압축 방식)"""
        if not self.json_data["frames"]:
            return
            
        # 메타데이터 설정
        compressed_data = {
            "meta": {
                "source": self.json_data["meta"]["source"],
                "frame_rate": self.json_data["meta"]["frame_rate"],
                "keyframe_interval": 1,  # 모든 프레임 사용
                "coordinate_quantization": 1000,  # 미터 단위를 밀리미터(mm)로 변환 (1000배)
                "angle_quantization": 100   # 각도를 센티도(centi-degree)로 변환 (100배)
            },
            "times": [],
            "joints": {
                "hip_center": [],
                "shoulder_left": [],
                "shoulder_right": [],
                "elbow_left": [],
                "elbow_right": [],
                "wrist_left": [],
                "wrist_right": [],
                "knee_left": [],
                "knee_right": [],
                "ankle_left": [],
                "ankle_right": []
            },
            "angles": {
                "elbow_left": [],
                "elbow_right": [],
                "knee_left": [],
                "knee_right": []
            },
            "velocity": {
                "hip_center": []
            }
        }
        
        # 시간 배열 추출
        compressed_data["times"] = [frame["time"] for frame in self.json_data["frames"]]
        
        # 좌표 압축 및 델타 인코딩
        prev_coords = {}
        for joint_name in compressed_data["joints"].keys():
            prev_coords[joint_name] = None
            
        # 관절 각도 압축 및 델타 인코딩
        prev_angles = {}
        for angle_name in compressed_data["angles"].keys():
            prev_angles[angle_name] = None
            
        # 속도 압축 및 델타 인코딩
        prev_velocity = None
        
        # 각 프레임 처리
        for i, frame in enumerate(self.json_data["frames"]):
            # 관절 좌표 압축
            for joint_name in compressed_data["joints"].keys():
                # 필요한 관절 이름 매핑
                mp_joint_name = joint_name
                if joint_name == "hip_center":
                    coords = frame["joints"]["hip_center"]
                elif joint_name == "shoulder_left":
                    coords = frame["joints"]["shoulder_left"]
                elif joint_name == "shoulder_right":
                    coords = frame["joints"]["shoulder_right"]
                elif joint_name == "elbow_left":
                    coords = frame["joints"]["elbow_left"]
                elif joint_name == "elbow_right":
                    coords = frame["joints"]["elbow_right"]
                elif joint_name == "wrist_left":
                    coords = frame["joints"]["wrist_left"]
                elif joint_name == "wrist_right":
                    coords = frame["joints"]["wrist_right"]
                elif joint_name == "knee_left":
                    coords = frame["joints"]["knee_left"]
                elif joint_name == "knee_right":
                    coords = frame["joints"]["knee_right"]
                elif joint_name == "ankle_left":
                    coords = frame["joints"]["ankle_left"]
                elif joint_name == "ankle_right":
                    coords = frame["joints"]["ankle_right"]
                
                # 좌표 양자화
                quantized = [
                    int(round(coords[0] * compressed_data["meta"]["coordinate_quantization"])),
                    int(round(coords[1] * compressed_data["meta"]["coordinate_quantization"])),
                    int(round(coords[2] * compressed_data["meta"]["coordinate_quantization"]))
                ]
                
                # 첫 프레임이거나 이전 좌표가 없는 경우 절대값 사용
                if i == 0 or prev_coords[joint_name] is None:
                    compressed_data["joints"][joint_name].extend(quantized)
                    prev_coords[joint_name] = quantized
                else:
                    # 델타 인코딩 (현재값 - 이전값)
                    delta = [
                        quantized[0] - prev_coords[joint_name][0],
                        quantized[1] - prev_coords[joint_name][1],
                        quantized[2] - prev_coords[joint_name][2]
                    ]
                    compressed_data["joints"][joint_name].extend(delta)
                    prev_coords[joint_name] = quantized
            
            # 관절 각도 압축
            for angle_name in compressed_data["angles"].keys():
                # 필요한 각도 이름 매핑
                feature_name = f"{angle_name}_angle"
                if feature_name in frame["features"]:
                    angle_value = frame["features"][feature_name]
                    
                    # 각도 양자화
                    quantized = int(round(angle_value * compressed_data["meta"]["angle_quantization"]))
                    
                    # 첫 프레임이거나 이전 각도가 없는 경우 절대값 사용
                    if i == 0 or prev_angles[angle_name] is None:
                        compressed_data["angles"][angle_name].append(quantized)
                        prev_angles[angle_name] = quantized
                    else:
                        # 델타 인코딩 (현재값 - 이전값)
                        delta = quantized - prev_angles[angle_name]
                        compressed_data["angles"][angle_name].append(delta)
                        prev_angles[angle_name] = quantized
            
            # 엉덩이 중심 속도 압축
            if "hip_center_velocity" in frame["features"] or "center_of_mass_velocity" in frame["features"]:
                velocity = frame["features"].get("hip_center_velocity", frame["features"].get("center_of_mass_velocity"))
                
                # 속도 양자화 (m/s -> mm/s로 변환)
                quantized = [
                    int(round(velocity[0] * 1000)),
                    int(round(velocity[1] * 1000)),
                    int(round(velocity[2] * 1000))
                ]
                
                # 첫 프레임이거나 이전 속도가 없는 경우 절대값 사용
                if i == 0 or prev_velocity is None:
                    compressed_data["velocity"]["hip_center"].extend(quantized)
                    prev_velocity = quantized
                else:
                    # 델타 인코딩 (현재값 - 이전값)
                    delta = [
                        quantized[0] - prev_velocity[0],
                        quantized[1] - prev_velocity[1],
                        quantized[2] - prev_velocity[2]
                    ]
                    compressed_data["velocity"]["hip_center"].extend(delta)
                    prev_velocity = quantized
                    
        return compressed_data

    def create_ultra_compressed_json(self):
        """5단계 압축 파이프라인 적용한 초압축 JSON 생성"""
        if not self.json_data["frames"]:
            return
            
        # 1. 관절/축소 선택 - 핵심 11 관절만 선택
        core_joints = [
            "hip_center",
            "shoulder_left", "shoulder_right",
            "elbow_left", "elbow_right",
            "wrist_left", "wrist_right",
            "knee_left", "knee_right",
            "ankle_left", "ankle_right"
        ]
        
        # 핵심 각도 선택
        core_angles = [
            "elbow_left", "elbow_right",
            "knee_left", "knee_right"
        ]
        
        # 메타데이터 설정
        ultra_compressed = {
            "meta": {
                "fps_in": 30,
                "fps_out": 10,
                "unit": "mm",
                "angle_unit": "centidegree"
            },
            "t": [],  # 타임스탬프 (ms)
        }
        
        # 핵심 관절 데이터 초기화
        for joint in core_joints:
            ultra_compressed[joint] = []
            
        # 핵심 각도 데이터 초기화
        for angle in core_angles:
            ultra_compressed[f"angle_{angle}"] = []
        
        # 속도 데이터 초기화
        ultra_compressed["com_v"] = []
        
        # 2. 시간 다운샘플링 / 키프레임 추출
        frames = self.json_data["frames"]
        keyframes = []
        last_keyframe_idx = -1
        
        for i, frame in enumerate(frames):
            # 정규 다운샘플링: 10fps (3프레임마다 한 번)
            is_regular_sample = (i % 3 == 0)
            
            # 움직임 기반 키프레임 선택
            is_motion_keyframe = False
            if i > 0 and last_keyframe_idx >= 0:
                # 이전 키프레임과 현재 프레임 사이의 포즈 변화 측정
                prev_pose = {}
                curr_pose = {}
                
                # hip_center와 어깨 위치를 기준으로 포즈 변화 측정
                for joint in ["hip_center", "shoulder_left", "shoulder_right"]:
                    if joint in frames[last_keyframe_idx]["joints"] and joint in frame["joints"]:
                        prev_pose[joint] = np.array(frames[last_keyframe_idx]["joints"][joint])
                        curr_pose[joint] = np.array(frame["joints"][joint])
                
                if prev_pose and curr_pose:
                    # 포즈 변화량 계산
                    pose_change = 0
                    for joint in prev_pose:
                        pose_change += np.linalg.norm(curr_pose[joint] - prev_pose[joint])
                    pose_change /= len(prev_pose)
                    
                    # 변화량이 임계값보다 크면 키프레임으로 선택
                    is_motion_keyframe = (pose_change > self.motion_threshold)
            
            # 정규 샘플링 또는 움직임 기반 키프레임이면 선택
            if is_regular_sample or is_motion_keyframe:
                keyframes.append(frame)
                last_keyframe_idx = i
        
        # 통계 데이터 계산을 위한 변수
        angle_stats = {angle: {"max": -float('inf'), "min": float('inf'), "values": []} for angle in core_angles}
        com_speeds = []
        events = []
        
        # 이전 프레임 데이터 (델타 인코딩용)
        prev_joint_pos = {joint: None for joint in core_joints}
        prev_angles = {angle: None for angle in core_angles}
        prev_com_v = None
        prev_time_ms = 0
        
        # 3. 델타 인코딩 + 정수화
        for i, frame in enumerate(keyframes):
            # 타임스탬프 (ms 단위)
            time_ms = int(round(frame["time"] * 1000))
            if i == 0:
                ultra_compressed["t"].append(time_ms)
            else:
                ultra_compressed["t"].append(time_ms - prev_time_ms)
            prev_time_ms = time_ms
            
            # 관절 좌표 처리
            for joint in core_joints:
                if joint in frame["joints"]:
                    # mm 단위로 변환하고 정수화
                    coords = np.array(frame["joints"][joint]) * 1000
                    coords = coords.round().astype(int)
                    
                    if i == 0 or prev_joint_pos[joint] is None:
                        # 첫 프레임은 절대 좌표
                        ultra_compressed[joint].extend(coords.tolist())
                    else:
                        # 이후 프레임은 델타값
                        delta = coords - prev_joint_pos[joint]
                        ultra_compressed[joint].extend(delta.tolist())
                    
                    prev_joint_pos[joint] = coords
            
            # 관절 각도 처리
            for angle in core_angles:
                angle_key = f"{angle}_angle"
                if angle_key in frame["features"]:
                    # 센티도(centidegree) 변환 (각도 * 100)
                    angle_value = frame["features"][angle_key]
                    angle_stats[angle]["values"].append(angle_value)
                    angle_stats[angle]["max"] = max(angle_stats[angle]["max"], angle_value)
                    angle_stats[angle]["min"] = min(angle_stats[angle]["min"], angle_value)
                    
                    # 각도가 150도 이상인 경우 이벤트 추가 (예: 팔꿈치 과신전)
                    if angle == "elbow_left" or angle == "elbow_right":
                        if angle_value > 150:
                            events.append({
                                "time": frame["time"],
                                "tag": f"{angle}_hyper"
                            })
                    
                    centidegree = int(round(angle_value * 100))
                    
                    if i == 0 or prev_angles[angle] is None:
                        # 첫 프레임은 절대값
                        ultra_compressed[f"angle_{angle}"].append(centidegree)
                    else:
                        # 이후 프레임은 델타값
                        delta = centidegree - prev_angles[angle]
                        ultra_compressed[f"angle_{angle}"].append(delta)
                    
                    prev_angles[angle] = centidegree
            
            # 속도 처리
            if "center_of_mass_velocity" in frame["features"]:
                velocity = np.array(frame["features"]["center_of_mass_velocity"])
                speed = np.linalg.norm(velocity)
                com_speeds.append(speed)
                
                # 속도가 1 m/s 이상인 경우 이벤트 추가
                if speed > 1.0:
                    events.append({
                        "time": frame["time"],
                        "tag": "com_spike"
                    })
                
                # mm/s 단위로 변환하고 정수화
                v_mm = (velocity * 1000).round().astype(int)
                
                if i == 0 or prev_com_v is None:
                    # 첫 프레임은 절대값
                    ultra_compressed["com_v"].extend(v_mm.tolist())
                else:
                    # 이후 프레임은 델타값
                    delta = v_mm - prev_com_v
                    ultra_compressed["com_v"].extend(delta.tolist())
                
                prev_com_v = v_mm
        
        # 4. 압축 및 Base64 인코딩
        # 문자열로 변환
        json_str = json.dumps(ultra_compressed, separators=(',', ':'))
        # zlib 압축
        compressed_data = zlib.compress(json_str.encode('utf-8'))
        # base64 인코딩
        b64_str = base64.b64encode(compressed_data).decode('ascii')
        
        # 5. 요약 통계 데이터 생성
        stats = {}
        
        # 각도 통계 데이터 추가 (무한대 값 확인)
        for angle in core_angles:
            # 값이 있는 경우에만 통계 추가
            if angle_stats[angle]["values"]:
                if angle_stats[angle]["max"] != -float('inf'):
                    stats[f"{angle}_max"] = int(round(angle_stats[angle]["max"] * 100))
                else:
                    stats[f"{angle}_max"] = 0  # 기본값 설정
                    
                if angle_stats[angle]["min"] != float('inf'):
                    stats[f"{angle}_min"] = int(round(angle_stats[angle]["min"] * 100))
                else:
                    stats[f"{angle}_min"] = 0  # 기본값 설정
            else:
                # 값이 없는 경우 기본값 설정
                stats[f"{angle}_max"] = 0
                stats[f"{angle}_min"] = 0
        
        # RMS 속도 추가
        if com_speeds:
            stats["com_speed_rms"] = round(np.sqrt(np.mean(np.array(com_speeds)**2)), 2)
        else:
            stats["com_speed_rms"] = 0
        
        # 이벤트 추가
        stats["events"] = events
        
        # 최종 초압축 데이터
        final_data = {
            "b64": b64_str,
            "stats": stats
        }
        
        return final_data

    def save_final_json(self):
        """최종 JSON 파일 저장"""
        if self.json_data["frames"]:
            # JSON 데이터를 시간순으로 정렬
            self.json_data["frames"].sort(key=lambda x: x["time"])

            # 일반 JSON 저장
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(self.json_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.json_data['frames'])} frames to {self.json_file}")
            
            # 압축 JSON 생성 및 저장
            compressed_data = self.create_compressed_json()
            if compressed_data:
                with open(self.compressed_json_file, "w", encoding="utf-8") as f:
                    json.dump(compressed_data, f, indent=2, ensure_ascii=False)
                print(f"Saved compressed data to {self.compressed_json_file}")
            
            # 초압축 JSON 생성 및 저장
            ultra_compressed_data = self.create_ultra_compressed_json()
            if ultra_compressed_data:
                with open(self.ultracompressed_json_file, "w", encoding="utf-8") as f:
                    json.dump(ultra_compressed_data, f, indent=2, ensure_ascii=False)
                print(f"Saved ultra-compressed data to {self.ultracompressed_json_file}")
                
                # LLM 프롬프트 예시 출력
                llm_prompt = f"""다음은 압축된 BlazePose 키프레임 데이터(base64 zlib)와 요약 통계입니다.
· meta.fps_out = 10fps
· 모든 좌표 mm, 각도 centidegree

JSON:
{json.dumps(ultra_compressed_data, indent=2)}

위 데이터를 기반으로
1) 비효율 자세
2) 부상 위험
3) 퍼포먼스 개선
을 분석해 주세요."""
                
                with open("llm_prompt.txt", "w", encoding="utf-8") as f:
                    f.write(llm_prompt)
                print("Generated LLM prompt in llm_prompt.txt")

    def keyPressEvent(self, event):
        # ESC로 종료
        if event.key() == Qt.Key_Escape:
            self.close()
        # 스페이스 키로 재생/일시정지 토글
        elif event.key() == Qt.Key_Space:
            self.paused = not self.paused
            if self.paused:
                self.timer.stop()
                print("영상 일시정지")
            else:
                self.timer.start(30)
                print("영상 재생")
        
        # 일시정지 상태에서의 탐색 기능
        if self.paused:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 오른쪽 화살표 키로 한 프레임 앞으로
            if event.key() == Qt.Key_Right:
                self.seek_to_frame(current_frame)
            
            # 왼쪽 화살표 키로 한 프레임 뒤로
            elif event.key() == Qt.Key_Left:
                self.seek_to_frame(current_frame - 2)  # 읽기 후 자동으로 +1 되므로 -2
            
            # Page Up 키로 10프레임 뒤로
            elif event.key() == Qt.Key_PageUp:
                self.seek_to_frame(current_frame - 11)  # 읽기 후 자동으로 +1 되므로 -11
            
            # Page Down 키로 10프레임 앞으로
            elif event.key() == Qt.Key_PageDown:
                self.seek_to_frame(current_frame + 9)  # 읽기 후 자동으로 +1 되므로 +9

    def closeEvent(self, event):
        # 종료 전 JSON 저장
        self.save_final_json()
        self.cap.release()
        super().closeEvent(event)

    def seek_to_frame(self, frame_number):
        """특정 프레임 번호로 이동"""
        # 프레임 범위 확인
        if frame_number < 0:
            frame_number = 0
        elif frame_number >= self.total_frames:
            frame_number = self.total_frames - 1

        # 프레임 설정
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # 해당 프레임 처리
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)
            print(f"프레임 {frame_number}로 이동")
        else:
            print("프레임 이동 실패")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PoseApp()
    win.show()
    sys.exit(app.exec_())
