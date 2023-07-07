import cv2
import mediapipe as mp
import csv
import datetime
import argparse
from argparse import RawTextHelpFormatter

dt_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

parser.add_argument('--input', help='Input video data', default='input_data/input_video.mp4')
parser.add_argument('--output', help='Output video data', default=f'output_data/{dt_now}.mp4')
parser.add_argument('--csv', help='Output csv data', default=f'output_data/{dt_now}.csv')

parser.usage = parser.format_help()

# 引数を解析する
args = parser.parse_args()

input_file = args.input

csv_file = args.csv

output_file = args.output

# 部位名のリスト
landmark_names = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(input_file)

# 出力動画の設定
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# MediapipeのPoseモデルを初期化
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # CSVファイルを書き込みモードでオープン
    csvfile = open(csv_file, mode='w', newline='')
    # CSVライターを作成
    csv_writer = csv.writer(csvfile)
    # ヘッダー行を書き込み
    csv_writer.writerow(['frame_number', 'landmark_name', 'landmark_x', 'landmark_y', 'landmark_z'])

    # フレームごとに処理を実行
    frame_number = 0
    while cap.isOpened():
        # フレームをキャプチャ
        ret, frame = cap.read()

        if not ret:
            break

        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        # 骨格情報を描画
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 骨格情報をCSVに書き込み
        if results.pose_landmarks:
            for index, landmark in enumerate(results.pose_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                z = landmark.z
                landmark_name = landmark_names[index]
                csv_writer.writerow([frame_number, landmark_name, x, y, z])

        out.write(annotated_frame)

        # ウィンドウに表示
        cv2.imshow('Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    csvfile.close()

cap.release()
out.release()
cv2.destroyAllWindows()
