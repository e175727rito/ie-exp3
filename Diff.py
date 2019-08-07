import matplotlib
import numpy as np
import os
from keras.models import Sequential
from keras import backend as K
import cv2
from PIL import Image
matplotlib.use('PS')
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pyaudio
import time
import pickle


#学習済みモデルの復元
with open('model.pickle','rb') as fp:
    clf = pickle.load(fp)


#読み取り開始
def main(abc):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 # int16型
    CHANNELS = 2             # ステレオ
    RATE = 44100             # 441.kHz
    RECORD_SECONDS = 20 # 20秒録音

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=0,
                    frames_per_buffer=CHUNK)

    print("now recording...")

    frames = []
    count = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

        DATUM = 8800 # 音響データの長さ（切り取り後）
        result = translate_numpy(frames)
        ch1 = result[0::2] # 紙幣の音響データ
        ch2 = result[1::2] # パルスデータ

        # データを切り取る
        max_index = np.argmax(ch2) # index
        max_value = ch2[max_index] # value

        if max_value > 0.5:
            for j in range(17): # 追加で９回データを取る
                data = stream.read(CHUNK)
                frames.append(data)
            result = translate_numpy(frames)
            ch1 = result[0::2] # 紙幣の音響データ
            ch2 = result[1::2] # パルスデータ

            split_data = ch1[max_index:max_index+DATUM] # データを切り取る
            print("ok")
            Datas = np.array(split_data)#8800個のデータ

            pred = clf.predict(Datas)#predには予想ラベルが入っている
            print("pred:"+str(pred))


max_iter = input("iteration num: ")#入れる枚数の指定

for a in range(int(max_iter)):

    main(a)
