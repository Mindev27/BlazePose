import cv2
import numpy as np
import mediapipe as mp
import argparse
import sys


def process_video(input_path: str, output_path: str):
    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)

    # 비디오 열기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        sys.exit(1)

    # 원본 영상 속성 읽기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 비디오 설정 (MP4 / mp4v 코덱)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {input_path} → {output_path}")
    print(f"  resolution: {width}x{height}, fps: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝

        # BGR→RGB 변환 후 포즈 추정
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # 2D 랜드마크 & 연결선 그리기
        if results.pose_landmarks:
            # 연결선
            for s, e in mp_pose.POSE_CONNECTIONS:
                x1 = int(results.pose_landmarks.landmark[s].x * width)
                y1 = int(results.pose_landmarks.landmark[s].y * height)
                x2 = int(results.pose_landmarks.landmark[e].x * width)
                y2 = int(results.pose_landmarks.landmark[e].y * height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 관절점
            for lm in results.pose_landmarks.landmark:
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # 프레임 쓰기
        out.write(frame)

    # 정리
    cap.release()
    out.release()
    pose.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MP4 파일에서 Pose 추출한 뒤, 스켈레톤 그려서 새로운 MP4로 저장합니다."
    )
    parser.add_argument("input_mp4", help="입력 비디오 파일 경로 (mp4)")
    parser.add_argument("output_mp4", help="출력 비디오 파일 경로 (mp4)")
    args = parser.parse_args()

    process_video(args.input_mp4, args.output_mp4)
