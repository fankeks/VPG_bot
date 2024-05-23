import numpy as np
from abc import ABC, abstractmethod


class IVPGAnalyzer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def filt(self, vpg: list, fd: float) -> list:
        """
        Метод фильтрации сигнала ВПГ
        :param vpg: Сигнал
        :param fd: Частота дискретизации
        :return: Отфильтрованный сигнал ВПГ
        """
        return vpg

    @abstractmethod
    def get_report_hr(self, vpg: list, fd: float) -> list:
        """
        Метод расчёта кривой изменения ЧСС
        :param vpg: Отфильтрованный сигнал ВПГ
        :param fd: Частота дискоетизации
        :return: Массив значений ЧСС
        """
        pass

    def get_report_hrv(self, vpg: list, fd: float, number: int, stride=1) -> dict:
        """
        Метод расчёта метрик вариабельности сердечного ритма
        :param vpg: Отфильтрованный сигнал ВПГ
        :param fd: Частота дискоетизации
        :param number: Число эллементов в срезе
        :param stride: Сдвиг окна для расчёта
        :return: Словарь с ключами: "sdann", "rmssd", "nn50". В каждом элементе хранится массив значений. (По аналогии с ЧСС)
        """
        pass