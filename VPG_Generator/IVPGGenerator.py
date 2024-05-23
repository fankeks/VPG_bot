import numpy as np
from abc import ABC, abstractmethod


class IVPGGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_vpg_discret(self, frame: np.ndarray) -> float:
        """
        Метод расчёта одного отсчёта ВПГ
        :param frame: Кадр
        :return: отсчёт ВПГ / если нет лица то вернёт None
        """
        return 0.0

    @abstractmethod
    def get_report(self, frames: list) -> list:
        """
        Метод формирования ВПГ сигнала
        :param frames: Список кадров
        :return: vpg - Сигнал ВПГ (Список значений)
        """
        vpg = []
        for frame in frames:
            vpg.append(0)
        return vpg
