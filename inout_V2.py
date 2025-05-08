import cv2
import mediapipe as mp
import numpy as np
import json
import math

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

# JSON 데이터 구조 초기화
pose_data = {
    "meta": {
        "source": "DeepMotion Animate 3D",
        "frame_rate": fps,
        "joints_count": 33,  # MediaPipe Pose는 33개의 랜드마크를 가짐
        "timestamp_unit": "seconds"
    },
    "frames": []
}

# 관절 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# 이전 프레임의 위치를 저장하기 위한 변수
prev_positions = {}
frame_count = 0

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

        # 프레임 데이터
        frame_data = {
            "time": frame_count / fps,
            "joints": {},
            "features": {}
        }

        # 관절 랜드마크 그리기 및 데이터 추출
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # 주요 관절 위치 매핑
            joint_mapping = {
                "hip_center": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                "shoulder_left": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                "elbow_left": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                "wrist_left": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                "shoulder_right": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                "elbow_right": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                "wrist_right": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                "knee_left": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                "ankle_left": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                "knee_right": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                "ankle_right": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                "hip_left": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                "hip_right": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            }
            
            # 관절 위치 데이터 저장 (정규화된 좌표를 실제 거리로 변환)
            for joint_name, landmark in joint_mapping.items():
                frame_data["joints"][joint_name] = [
                    landmark.x, landmark.y, landmark.z if landmark.visibility > 0.5 else 0
                ]
            
            # 특징 계산 (각도, 속도 등)
            # 왼쪽 팔꿈치 각도 (어깨-팔꿈치-손목)
            if all(joint in joint_mapping for joint in ["shoulder_left", "elbow_left", "wrist_left"]):
                elbow_left_angle = calculate_angle(
                    [joint_mapping["shoulder_left"].x, joint_mapping["shoulder_left"].y, joint_mapping["shoulder_left"].z],
                    [joint_mapping["elbow_left"].x, joint_mapping["elbow_left"].y, joint_mapping["elbow_left"].z],
                    [joint_mapping["wrist_left"].x, joint_mapping["wrist_left"].y, joint_mapping["wrist_left"].z]
                )
                frame_data["features"]["elbow_left_angle"] = round(elbow_left_angle, 1)
            
            # 오른쪽 무릎 각도 (엉덩이-무릎-발목)
            if all(joint in joint_mapping for joint in ["hip_right", "knee_right", "ankle_right"]):
                knee_right_angle = calculate_angle(
                    [joint_mapping["hip_right"].x, joint_mapping["hip_right"].y, joint_mapping["hip_right"].z],
                    [joint_mapping["knee_right"].x, joint_mapping["knee_right"].y, joint_mapping["knee_right"].z],
                    [joint_mapping["ankle_right"].x, joint_mapping["ankle_right"].y, joint_mapping["ankle_right"].z]
                )
                frame_data["features"]["knee_right_angle"] = round(knee_right_angle, 1)
            
            # 무게 중심 계산 (간단히 주요 관절의 평균 위치로 추정)
            center_of_mass = [0, 0, 0]
            valid_joints = 0
            
            for joint_pos in frame_data["joints"].values():
                center_of_mass[0] += joint_pos[0]
                center_of_mass[1] += joint_pos[1]
                center_of_mass[2] += joint_pos[2]
                valid_joints += 1
            
            if valid_joints > 0:
                center_of_mass = [coord / valid_joints for coord in center_of_mass]
                frame_data["features"]["center_of_mass"] = [round(coord, 2) for coord in center_of_mass]
            
            # 속도 계산 (이전 프레임과 현재 프레임의 위치 차이)
            if frame_count > 0 and "hip_center" in prev_positions:
                hip_velocity = [
                    (frame_data["joints"]["hip_center"][0] - prev_positions["hip_center"][0]) * fps,
                    (frame_data["joints"]["hip_center"][1] - prev_positions["hip_center"][1]) * fps,
                    (frame_data["joints"]["hip_center"][2] - prev_positions["hip_center"][2]) * fps
                ]
                frame_data["features"]["hip_center_velocity"] = [round(v, 2) for v in hip_velocity]
            
            # 현재 위치 저장
            prev_positions = {joint_name: position.copy() for joint_name, position in frame_data["joints"].items()}
            
            # 프레임 데이터 추가
            pose_data["frames"].append(frame_data)

        # 출력 비디오에 프레임 저장
        out.write(image)
        frame_count += 1

# JSON 파일 저장
with open("pose_data.json", "w", encoding="utf-8") as json_file:
    json.dump(pose_data, json_file, indent=2)

# 자원 정리
cap.release()
out.release()
cv2.destroyAllWindows()
print("완료!")
print("→ 'output_1.mp4'로 영상이 저장되었습니다.")
print("→ 'pose_data.json'로 포즈 데이터가 저장되었습니다.")
