import numpy as np
import cv2
import dlib

from VPG_Generator.IVPGGenerator import IVPGGenerator


class VPGGenerator(IVPGGenerator):
    def __init__(self, predictor_path='', cascade_path=''):
        """
        :param predictor_path:
        :param cascade_path: Путь к файлу с каскадами хаара
        """
        if len(predictor_path) == 0:
            predictor_path = 'shape_predictor_68_face_landmarks.dat'
        if len(cascade_path) == 0:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        self.__predictor = dlib.shape_predictor(predictor_path)
        self.__cascade = cv2.CascadeClassifier(cascade_path)

        self._points_pred = None

    def detect_face(self, frame: np.ndarray) -> tuple:
        """
        Метод выделения лица на изображении при помощи каскадов хаара
        :param frame: Изображение
        :return: (only_face, rectungle) - Кадр с лицом, координаты прямоугольника
        """
        faces = self.__cascade.detectMultiScale(frame,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(100, 100),
                                                maxSize=(550, 550))
        # Если есть лицо
        if len(faces) > 0:
            x = faces[0][0]         # Координата x прямоугольника
            y = faces[0][1]         # Координата y прямоугольника
            width = faces[0][2]     # Ширина прямоугольника
            height = faces[0][3]    # Высота прямоугольника

            only_face = frame[y:y + height,
                              x:x + width]

            rectangle = [x, y,
                         x + width - 1, y + height - 1]

            return only_face, np.array(rectangle)

        # Если нет лица
        return frame, np.array([])

    def get_landmarks(self, frame: np.ndarray, rectangle: list) -> np.matrix:
        """
        Метод выделения контрольных точек на лице
        :param frame: Изображение
        :param rectangle: Координаты прямоугольника с лицом
        :return: Массив координат [[x, y], ...]
        """
        rect = dlib.rectangle(0, 0, frame.shape[0], frame.shape[1])
        if len(rectangle) != 0:
            x1, y1, x2, y2 = rectangle
            rect = dlib.rectangle(int(x1), int(y1), int(x2 + 1), int(y2 + 1))

        if self._points_pred is None:
            self._points_pred = np.matrix([[p.x, p.y] for p in self.__predictor(frame, rect).parts()])
            return self._points_pred

        points = np.matrix([[p.x, p.y] for p in self.__predictor(frame, rect).parts()])
        for i in range(len(points)):
            delta = np.abs(points[i] - self._points_pred[i])
            if np.max(delta) > 2:
                self._points_pred[i] = (self._points_pred[i] + points[i]) // 2
        return self._points_pred

    def _get_segmented_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Метод для расчёта одного кадра ВПГ
        :param frame: Кадр
        :return: vpg_frame - Массив
        """
        # Ищем лицо
        _, rectangle = self.detect_face(frame)
        if len(rectangle) == 0:
            return np.array([])

        # Ищем контрольные точки
        points = self.get_landmarks(frame, rectangle)

        #Формируем области интереса
        ver = [50, 33, 30, 29]              # Точки на носу
        hor = [4, 5, 6, 7, 8, 9, 10, 11]    # Точки на подбородке

        frame = cv2.medianBlur(frame, 3)
        frame = cv2.GaussianBlur(frame, (3, 3), 5)

        channels = cv2.split(frame)
        one_frame_vpg = np.zeros(shape=(3, len(ver) - 1, len(hor) - 1))

        try:
            for i in range(len(hor) - 1):
                hl_x = points[hor[i]][0, 0]
                lr_x = points[hor[i + 1]][0, 0]

                for j in range(len(ver) - 1):
                    hl_y = points[ver[j + 1]][0, 1]
                    lr_y = points[ver[j]][0, 1]

                    if i != 3 or i != 4 and j != 2:
                        submats = np.asarray([x[hl_y:lr_y, hl_x:lr_x] for x in channels])

                        for k in range(len(channels)):
                            one_frame_vpg[k][len(ver) - j - 2][i] = np.mean(submats[k])
        except:
            return np.array([])

        return one_frame_vpg

    @staticmethod
    def _get_RGB(one_frame_vpg: np.ndarray) -> tuple:
        """
        Метод формирования каналов R G B
        :param one_frame_vpg: - Сигналы в областях интереса
        :return: R, G, B - Сигналы R G B
        """
        return np.mean(one_frame_vpg[2]), np.mean(one_frame_vpg[1]), np.mean(one_frame_vpg[0])

    @staticmethod
    def _vpg_func(r: float, g: float, b: float) -> float:
        """
        Метод преобразования каналов в отсчёт ВПГ
        :param r: Красный канал
        :param g: Зелёный канал
        :param b: Синий канал
        :return: ВПГ
        """
        return - 1 * g / (1 * r - 65 * b)

    def get_vpg_discret(self, frame: np.ndarray) -> float:
        """
        Метод расчёта одного отсчёта ВПГ
        :param frame: Кадр
        :return: отсчёт ВПГ / если нет лица то вернёт None
        """
        try:
            one_frame_vpg = self._get_segmented_frame(frame)
            # Проверка на наличие лица
            if len(one_frame_vpg) != 3:
                return None
            r, g, b = self._get_RGB(one_frame_vpg)
        except:
            return None
        return self._vpg_func(r, g, b)

    def get_vpg_discret_without_face(self, frame: np.ndarray) -> float:
        """
        Метод расчёта одного отсчёта ВПГ
        :param frame: Кадр
        :return: отсчёт ВПГ
        """
        channels = np.array(cv2.split(frame), np.float64)
        r, g, b = self._get_RGB(channels)
        return self._vpg_func(r, g, b)

    def get_report(self, frames: list) -> list:
        """
        Метод формирования ВПГ сигнала
        :param frames: Список кадров
        :return: vpg - Сигнал ВПГ (массив значений)
        """
        vpg = []
        for frame in frames:
            value = self.get_vpg_discret(frame)
            vpg.append(value)

        return vpg