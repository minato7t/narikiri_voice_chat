## 準備

以下のインストール

- sox

[こちら](https://ja.osdn.net/projects/sfnet_sox/downloads/sox/14.4.2/sox-14.4.2-win32.exe/)からダウンロードしてインストール

- tensorflow

```
pip install tensorflow-gpu==1.5.0
```

リポジトリをクローンし、リポジトリのフォルダ内へ移動

```
git clone https://github.com/NON906/NVC_train.git
cd NVC_train
```

## 実行

```
python train.py 音声が入ったディレクトリのパス 生成するnvzのパス
```

何らかの理由で途中で中断した場合は、以下から再開できる

```
python train.py 音声が入ったディレクトリのパス 生成するnvzのパス True
```
