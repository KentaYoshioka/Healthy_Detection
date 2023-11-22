# 骨格情報の数値化，可視化
## 概要
動画データを読み込み，映った人物(現在は一人)の骨格情報を数値化し，CSVデータに出力する．
また，骨格情報を可視化した動画データも出力する．
また，デモ用としてカメラから動画を撮影し，ゲージとして可視化した活動量を動画として出力する機能も用意した．

## ファイル，ディレクトリ一覧
+ README.md
    + このファイル
+ draw_skelton_informations.py
    + 映った人物の骨格情報を数値化し，CSVデータに出力するプログラム
+ input_data/
    + 読み込む動画データを格納する．入力データに指定がない場合，このディレクトリ直下のinput_video.mp4が指定される．
+ output_data/
    + 動画データ，CSVデータが出力される．
+ detect_track.py(デモ用)
    + カメラから動画を撮影し，ゲージとして可視化した活動量を動画として送信するプログラム


### 使用方法
python3 draw_skelton_informations.py [-h] [--input INPUTDATA] [--output OUTPUTDATA] [--csv OUTPUTCSV][--box][--skelton]

+ optional arguments:
  + -h, --help
    + show this help message and exit
  + --input INPUTDATA
    + imput file (mp4_filename)
  + --output OUTPUTDATA
    + output file(mp4_filename)
  + --csv OUTPUTCSV
    + output csvfile(csv_filename)
  + --box
    + output only box
  + --skelton
    + output only skelton

### デモプログラム使用法
1. python3 のインストール
2. import のためのインストール
```pip3 install -r requirements.txt```
3. デモの実行
```python3 detect_track.py```
(設定)
検知したい人の最大数を変更する場合には，コードの変更をお願いします．

### 実行環境
+ 実行環境
    + Python 3.11.3
    + ライブラリ
      + cv2
      + mediapipe
      + csv
      + datetime
      + argparse
      + RawTextHelpFormatter
