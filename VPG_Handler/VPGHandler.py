import asyncio
import multiprocessing as mp
import json
import os
import cv2
import numpy as np


from VPG_Analyzer.VPGAnalyzer import VPGAnalyzer
from VPG_Generator.VPGGenerator import VPGGenerator


class VPGHandler:
    @staticmethod
    def f(path):
        vpg_generator = VPGGenerator(os.path.join('VPG_Generator', 'shape_predictor_68_face_landmarks.dat'))
        vpg_analyzer = VPGAnalyzer()
        
        # Считываем файл
        frames = []
        cap = cv2.VideoCapture(path)
        print(int(cap.get(3)), int(cap.get(4)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        while (True):
            ret, frame = cap.read()
            # Если файл закончился выйти из цикла
            if ret == False:
                break
            #Добавляем frame
            frames.append(frame)
        cap.release()
        vpg = vpg_generator.get_report(frames)

        # Избавление от кадров без лица
        for i in range(len(vpg)):
            if vpg[i] is None:
                if i == 0:
                    vpg[i] = 0
                else:
                    vpg[i] = vpg[i - 1]

        # Нормолизуем сигнал
        vpg = (vpg - np.mean(vpg)) / np.std(vpg)
        # Фильтрация
        vpg_filt = vpg_analyzer.filt(vpg, fps)
        # Расчёт ЧСС
        hr = vpg_analyzer.get_report_hr(vpg_filt, fps)
        hr = hr['hr']
        # Избавление от None
        for i in range(len(hr)):
            if hr[i] is None:
                if i == 0:
                    hr[i] = 0
                else:
                    hr[i] = hr[i - 1]

        with open(path.split('.')[0] + '.json', 'w') as file:
            json.dump(round(np.mean(hr)), file)
        
        return None

    def __init__(self, path):
        self.__p = mp.Process(target=self.f, args=(path,))
        self.__path = path

    async def start(self):
        self.__p.start()

    async def join(self):
        while self.__p.is_alive():
            await asyncio.sleep(0)

        with open(self.__path.split('.')[0] + '.json', 'r') as file:
            ans = json.load(file)
        os.remove(self.__path.split('.')[0] + '.json')
        return ans
