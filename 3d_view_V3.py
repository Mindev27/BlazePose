import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# MediaPipe 포즈 연결 정보 (3D용)
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

# 실시간 matplotlib 초기화
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

# MediaPipe 초기화
mp_pose = mp.solutions.pose

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks and results.pose_world_landmarks:
            h, w, _ = frame.shape
            landmarks_2d = results.pose_landmarks.landmark
            landmarks_3d = results.pose_world_landmarks.landmark

            # 2D 뷰에 리깅 그리기
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                x1 = int(landmarks_2d[start_idx].x * w)
                y1 = int(landmarks_2d[start_idx].y * h)
                x2 = int(landmarks_2d[end_idx].x * w)
                y2 = int(landmarks_2d[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for lm in landmarks_2d:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # 3D 좌표 추출 및 중심 정렬 (발이 벽에 붙지 않도록 조정)
            xs = np.array([lm.x for lm in landmarks_3d])
            ys = np.array([lm.y for lm in landmarks_3d])
            zs = np.array([lm.z for lm in landmarks_3d])

            # 중앙 기준으로 정렬
            xs -= np.mean(xs)
            ys -= np.mean(ys)
            zs -= np.mean(zs)

            # 3D 시각화
            ax.clear()
            ax.set_title("3D Pose (Centered)")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.scatter(xs, zs, -ys, c="r", s=20)  # Z축은 깊이, Y는 위아래라 반전

            for start_idx, end_idx in POSE_CONNECTIONS_3D:
                ax.plot(
                    [xs[start_idx], xs[end_idx]],
                    [zs[start_idx], zs[end_idx]],
                    [-ys[start_idx], -ys[end_idx]],
                    c="b",
                )

            plt.draw()
            plt.pause(0.001)

        cv2.imshow("2D Pose View", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
