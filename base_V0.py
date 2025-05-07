import cv2
import mediapipe as mp

# Pose Landmarker API
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    model_complexity=1,         # 0은 라이트, 1은 풀 모델
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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

        # 결과 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('BlazePose Demo', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
