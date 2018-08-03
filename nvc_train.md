## 0.準備

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

## 1. 音声から解析ファイルを生成

targetsフォルダを作成し、音声をその中に格納する

```
# nvmファイルの生成（nvzファイルに必要）
python make_nvm.py targets outputs/target.nvm
```

[こちら](https://github.com/NON906/NVC_train/releases/download/v0.1/voice_h5.zip)からダウンロードし、解凍したvoice.h5をリポジトリへ格納

```
# gen_targetsの生成（学習に必要）
python gen_targets.py targets gen_targets gen_targets.zip
```

## 2. 学習データtarget.h5の生成（GPU推奨）

```
python target_train.py gen_targets outputs/target.h5 20 -1 32
```

（備考：中断した場合は、以下で再開できる）

```
python target_retrain.py outputs/target.h5 gen_targets outputs/target.h5 20 -1 32
```

## 3. 学習データpitch.h5の生成（GPU推奨）

```
python target_pitch_train.py gen_targets outputs/pitch.h5 20 -1 32
```

（備考：中断した場合は、以下で再開できる）

```
python target_pitch_retrain.py outputs/pitch.h5 gen_targets outputs/pitch.h5 20 -1 32
```

## 4. pbファイルを作成し、nvzファイルに統合

```
python target_convert_h5_to_pb.py outputs/target.h5 outputs/target.pb
python target_pitch_convert_h5_to_pb.py outputs/pitch.h5 outputs/pitch.pb

python make_nvz.py outputs/target.nvz
```
