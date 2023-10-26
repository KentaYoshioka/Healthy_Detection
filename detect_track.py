import cv2
from ultralytics import YOLO
import numpy as np

# 学習済みのモデルをロード
model = YOLO('yolov8n.pt')

# 動画ファイル(or カメラ)を開く
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 10)

size = 100000 # 配列のサイズを設定
pre_x1, pre_x2,  pre_y1,  pre_y2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int)
dis_x1, dis_x2,  dis_y1,  dis_y2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int)
pre_dis_x1, pre_dis_x2, pre_dis_y1, pre_dis_y2 = np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int)

# キーが押されるまでループ
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # YOLOv8でトラッキング
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")

        # キャプチャした画像サイズを取得
        imageWidth = results[0].orig_shape[0]
        imageHeight = results[0].orig_shape[1]
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes
        annotatedFrame = results[0].plot()

        # 結果を画像に変換
        annotated_frame = results[0].plot()

        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            if box.id is not None:
                ids = int(box.id[0])
            else:             continue  # または適切なデフォルト値を代入

            dis_x1[ids], dis_x2[ids], dis_y1[ids], dis_y2[ids] = abs(x1-pre_x1[ids]), abs(x2-pre_x2[ids]), abs(y1-pre_y1[ids]), abs(y2-pre_y2[ids])
            evaluation = abs(dis_x1[ids]-pre_dis_x1[ids]) + abs(dis_x2[ids]-pre_dis_x2[ids]) + abs(dis_y1[ids]-pre_dis_y1[ids]) + abs(dis_y2[ids]-pre_dis_y2[ids])
            pre_x1[ids], pre_x2[ids], pre_y1[ids], pre_y2[ids] = x1, x2, y1, y2
            pre_dis_x1[ids], pre_dis_x2[ids], pre_dis_y1[ids], pre_dis_y2[ids] = dis_x1[ids], dis_x2[ids], dis_y1[ids], dis_y2[ids]
            print(f"Object: {name} Coordinates: StartX={x1}, StartY={y1}, EndX={x2}, EndY={y2}")

            # バウンディングBOXの座標情報を書き込む
            if evaluation < 500:
                cv2.line(annotatedFrame, pt1=((x1, y2)),pt2=(x1 ,y2 - evaluation*4),color=(0,255,0),thickness=20,lineType=cv2.LINE_4, shift=0)
            cv2.putText(annotatedFrame, f"EVALUATION {evaluation}", (x1, y1 - 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2, cv2.LINE_AA)

        # プレビューウィンドウに画像出力
        cv2.imshow("YOLOv8 Inference", annotatedFrame)

        # アプリケーション終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 終了処理
cap.release()
cv2.destroyAllWindows()


