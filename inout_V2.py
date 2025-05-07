import cv2
import mediapipe as mp

# 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 입력 비디오 열기
cap = cv2.VideoCapture("input_1.mp4")
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 정보 읽기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정 (mp4v)

# 출력 비디오 저장 객체 설정
out = cv2.VideoWriter("output_1.mp4", fourcc, fps, (width, height))

# 포즈 추정 모델 불러오기
with mp_pose.Pose(
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 종료 조건

        # RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 포즈 인식
        results = pose.process(image)

        # 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 관절 랜드마크 그리기
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # 출력 비디오에 프레임 저장
        out.write(image)

# 자원 정리
cap.release()
out.release()
cv2.destroyAllWindows()
print("완료! → 'test_video_output.mp4'로 저장되었습니다.")
