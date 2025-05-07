import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처 시작
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

        # RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 포즈 추정
        results = pose.process(image)

        # 다시 BGR로
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 결과 랜드마크 그리기 및 무게중심 계산
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # 랜드마크 추출
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # 부위별 중심 좌표 및 질량 비중
            body_parts = {
                "head": {"indices": [0], "weight": 8.26},
                "torso": {"indices": [11, 12, 23, 24], "weight": 50.00},
                "upper_arm": {"indices": [11, 12], "weight": 5.00},
                "forearm": {"indices": [13, 14], "weight": 3.00},
                "hand": {"indices": [15, 16], "weight": 1.00},
                "thigh": {"indices": [23, 24], "weight": 10.00},
                "shank": {"indices": [25, 26], "weight": 4.50},
                "foot": {"indices": [27, 28], "weight": 1.50},
            }

            total_weight = 0
            com_x = 0
            com_y = 0

            for part in body_parts.values():
                x = sum([landmarks[i].x for i in part["indices"]]) / len(
                    part["indices"]
                )
                y = sum([landmarks[i].y for i in part["indices"]]) / len(
                    part["indices"]
                )
                weight = part["weight"]
                com_x += x * weight
                com_y += y * weight
                total_weight += weight

            com_x /= total_weight
            com_y /= total_weight

            # 이미지 좌표로 변환
            cx, cy = int(com_x * w), int(com_y * h)

            # 무게중심 표시
            cv2.circle(image, (cx, cy), 10, (0, 255, 255), -1)
            cv2.putText(
                image,
                "C",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        cv2.imshow("BlazePose with Center of Mass", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
