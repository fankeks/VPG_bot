import os
import numpy as np
import json
import matplotlib.pyplot as plt

from VPG_Analyzer.VPGAnalyzer import VPGAnalyzer


PATH_DATA = 'Data'
FPS = 30


def test():
    for file_name in os.listdir(PATH_DATA):
        file_path = os.path.join(PATH_DATA, file_name)
        print(file_name)

        vpg_analyzer = VPGAnalyzer()

        with open(file_path, 'r') as file:
            vpg = json.load(file)
            # Избавление от кадров без лица
            for i in range(len(vpg)):
                if vpg[i] is None:
                    if i == 0:
                        vpg[i] = 0
                    else:
                        vpg[i] = vpg[i - 1]
        vpg = np.array(vpg)

        f, spec = vpg_analyzer.get_spec(vpg, FPS)
        plt.plot(f[1:], np.abs(spec)[1:])
        plt.grid()
        plt.show()

        t = np.array(list(range(len(vpg)))) * (1 / FPS)
        vpg = (vpg - np.mean(vpg)) / np.std(vpg)

        plt.plot(t, vpg)
        vpg = vpg_analyzer.filt(vpg, FPS)
        plt.plot(t, vpg)

        peaks = vpg_analyzer.find_peaks(vpg)
        plt.plot(t[peaks], vpg[peaks], "x")
        plt.show()

        print(f'SDANN: {vpg_analyzer.get_sdann(vpg, FPS)}')
        print(f'RMSSD: {vpg_analyzer.get_rmssd(vpg, FPS)}')
        print(f'NN50: {vpg_analyzer.get_nn50(vpg, FPS)}')
        print(f'ЧСС: {vpg_analyzer.get_hr_peak(vpg, FPS)}')

if __name__ == '__main__':
    test()